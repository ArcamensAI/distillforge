use async_trait::async_trait;
use bytes::Bytes;
use distillforge::config::{load_config, AppConfig, ModelBackendConfig};
use distillforge::feedback::FeedbackWriter;
use distillforge::log_writer::{JsonlLogWriter, LogInput};
use distillforge::metrics::ProxyMetrics;
use distillforge::routing::{
    decide_route, extract_metadata, load_routing_snapshot, RequestMetadata, RoutingDecision,
    RoutingSnapshot,
};
use distillforge::shadow_log::{ShadowLogInput, ShadowLogWriter};
use log::{error, info, warn};
use pingora::http::ResponseHeader;
use pingora::prelude::*;
use pingora::proxy::{http_proxy_service, ProxyHttp, Session};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use std::{
    io::{self, Read, Write},
    net::{TcpStream, ToSocketAddrs},
    thread,
    time::Duration,
};

#[derive(Clone)]
struct DistillProxy {
    config: Arc<AppConfig>,
    logs: Arc<JsonlLogWriter>,
    shadow_logs: Arc<ShadowLogWriter>,
    feedback: Arc<FeedbackWriter>,
    metrics: Arc<ProxyMetrics>,
    routing_snapshot: Arc<RwLock<RoutingSnapshot>>,
}

#[derive(Debug, Clone)]
struct RequestContext {
    started_at: Instant,
    metadata: RequestMetadata,
    decision: Option<RoutingDecision>,
    task_id: String,
    selected_model: String,
    routing_decision: String,
    routing_reason: String,
    selected_backend: Option<ModelBackendConfig>,
    shadow_backend: Option<ModelBackendConfig>,
    fallback_attempted: bool,
    request_body_bytes: usize,
    response_body_bytes: usize,
    request_body_capture: Vec<u8>,
    response_body_capture: Vec<u8>,
    skip_interaction_log: bool,
}

#[async_trait]
impl ProxyHttp for DistillProxy {
    type CTX = RequestContext;

    fn new_ctx(&self) -> Self::CTX {
        RequestContext {
            started_at: Instant::now(),
            metadata: RequestMetadata {
                request_id: String::new(),
                client_id: None,
                task_id: None,
                cost_center: None,
                quality_mode: None,
                pii_level: None,
                no_train: false,
            },
            decision: None,
            task_id: "unknown_task".to_string(),
            selected_model: "none".to_string(),
            routing_decision: "none".to_string(),
            routing_reason: "not_routed".to_string(),
            selected_backend: None,
            shadow_backend: None,
            fallback_attempted: false,
            request_body_bytes: 0,
            response_body_bytes: 0,
            request_body_capture: Vec::new(),
            response_body_capture: Vec::new(),
            skip_interaction_log: false,
        }
    }

    async fn request_filter(&self, session: &mut Session, ctx: &mut Self::CTX) -> Result<bool> {
        self.metrics.inc_total();

        let path = session.req_header().uri.path();
        if path == "/health" {
            ctx.skip_interaction_log = true;
            respond_text(session, 200, "ok\n").await?;
            return Ok(true);
        }
        if path == "/metrics" {
            ctx.skip_interaction_log = true;
            respond_text(session, 200, &self.metrics.render_prometheus()).await?;
            return Ok(true);
        }
        if path == "/admin/reload-routing" {
            ctx.skip_interaction_log = true;
            let reload_response = match load_routing_snapshot(&self.config.routing.snapshot_path) {
                Ok(snapshot) => match self.routing_snapshot.write() {
                    Ok(mut active_snapshot) => {
                        let version = snapshot.version;
                        *active_snapshot = snapshot;
                        (
                            200,
                            format!("routing snapshot reloaded version={version}\n"),
                        )
                    }
                    Err(_) => (500, "routing snapshot lock poisoned\n".to_string()),
                },
                Err(err) => (400, format!("{err}\n")),
            };
            respond_text(session, reload_response.0, &reload_response.1).await?;
            return Ok(true);
        }
        if path == "/v1/feedback" {
            ctx.skip_interaction_log = true;
            self.metrics.inc_feedback();
            match read_limited_body(session, self.config.logging.max_capture_bytes).await {
                Ok(body) => match serde_json::from_slice::<serde_json::Value>(&body) {
                    Ok(value) => {
                        let headers = &session.req_header().headers;
                        self.feedback.write_value(
                            &value,
                            header_value(headers, "x-client-id"),
                            header_value(headers, "x-task-id"),
                        );
                        respond_text(session, 202, "accepted\n").await?;
                    }
                    Err(_) => {
                        self.metrics.inc_rejected();
                        respond_text(session, 400, "invalid feedback json\n").await?;
                    }
                },
                Err(_) => {
                    self.metrics.inc_rejected();
                    respond_text(session, 413, "feedback body too large\n").await?;
                }
            }
            return Ok(true);
        }

        if path != "/v1/chat/completions" && path != "/v1/completions" {
            self.metrics.inc_rejected();
            ctx.routing_reason = "unsupported_path".to_string();
            let _ = session.respond_error(404).await;
            return Ok(true);
        }

        ctx.metadata = extract_metadata(&session.req_header().headers);
        let snapshot = self
            .routing_snapshot
            .read()
            .map(|snapshot| snapshot.clone())
            .unwrap_or_default();
        let decision = decide_route(
            &ctx.metadata,
            self.config.routing.default_missing_task_behavior,
            &snapshot,
        );

        match &decision {
            RoutingDecision::Teacher { reason, task_id } => {
                self.metrics.inc_teacher();
                ctx.task_id = task_id.clone();
                ctx.routing_reason = reason.clone();
                ctx.routing_decision = "teacher".to_string();
                ctx.selected_model = self.config.teacher.name.clone();
                ctx.selected_backend = Some(self.config.teacher.clone());
                ctx.decision = Some(decision);
                Ok(false)
            }
            RoutingDecision::Student {
                reason,
                task_id,
                model_id,
            } => match student_backend(&self.config, model_id) {
                Some(backend) => {
                    self.metrics.inc_student();
                    ctx.task_id = task_id.clone();
                    ctx.routing_reason = reason.clone();
                    ctx.routing_decision = "student".to_string();
                    ctx.selected_model = backend.name.clone();
                    ctx.selected_backend = Some(backend);
                    ctx.decision = Some(decision);
                    Ok(false)
                }
                None => {
                    self.metrics.inc_teacher();
                    ctx.task_id = task_id.clone();
                    ctx.routing_reason = "student_backend_missing_teacher_fallback".to_string();
                    ctx.routing_decision = "teacher".to_string();
                    ctx.selected_model = self.config.teacher.name.clone();
                    ctx.selected_backend = Some(self.config.teacher.clone());
                    ctx.decision = Some(RoutingDecision::Teacher {
                        reason: ctx.routing_reason.clone(),
                        task_id: task_id.clone(),
                    });
                    Ok(false)
                }
            },
            RoutingDecision::Shadow {
                reason,
                task_id,
                model_id,
            } => {
                self.metrics.inc_teacher();
                ctx.task_id = task_id.clone();
                ctx.routing_reason = reason.clone();
                ctx.routing_decision = "teacher".to_string();
                ctx.selected_model = self.config.teacher.name.clone();
                ctx.selected_backend = Some(self.config.teacher.clone());
                ctx.shadow_backend = student_backend(&self.config, model_id);
                ctx.decision = Some(decision);
                Ok(false)
            }
            RoutingDecision::Reject { status, reason } => {
                self.metrics.inc_rejected();
                ctx.routing_reason = reason.clone();
                ctx.routing_decision = "reject".to_string();
                ctx.selected_model = "none".to_string();
                ctx.decision = Some(decision.clone());
                let _ = session.respond_error(*status).await;
                Ok(true)
            }
        }
    }

    async fn request_body_filter(
        &self,
        _session: &mut Session,
        body: &mut Option<Bytes>,
        _end_of_stream: bool,
        ctx: &mut Self::CTX,
    ) -> Result<()>
    where
        Self::CTX: Send + Sync,
    {
        if let Some(body) = body {
            ctx.request_body_bytes += body.len();
            append_capture(
                &mut ctx.request_body_capture,
                body,
                self.config.logging.max_capture_bytes,
            );
        }
        Ok(())
    }

    async fn upstream_peer(
        &self,
        _session: &mut Session,
        ctx: &mut Self::CTX,
    ) -> Result<Box<HttpPeer>> {
        let backend = ctx
            .selected_backend
            .as_ref()
            .unwrap_or(&self.config.teacher);
        let mut peer = Box::new(HttpPeer::new(
            backend.address.as_str(),
            backend.use_tls,
            backend.sni.clone(),
        ));
        apply_peer_timeouts(&mut peer, &self.config, ctx.routing_decision == "student");
        Ok(peer)
    }

    fn fail_to_connect(
        &self,
        _session: &mut Session,
        _peer: &HttpPeer,
        ctx: &mut Self::CTX,
        error: Box<pingora::Error>,
    ) -> Box<pingora::Error> {
        if ctx.routing_decision == "student" && !ctx.fallback_attempted {
            ctx.fallback_attempted = true;
            ctx.routing_decision = "teacher".to_string();
            ctx.routing_reason = "student_connect_error_teacher_fallback".to_string();
            ctx.selected_model = self.config.teacher.name.clone();
            ctx.selected_backend = Some(self.config.teacher.clone());
            self.metrics.inc_fallback();
            self.metrics.inc_teacher();
            warn!(
                "student upstream connection failed; retrying request_id={} on teacher",
                ctx.metadata.request_id
            );

            let mut retry_error = error;
            retry_error.set_retry(true);
            return retry_error;
        }

        error
    }

    async fn upstream_request_filter(
        &self,
        _session: &mut Session,
        upstream_request: &mut RequestHeader,
        ctx: &mut Self::CTX,
    ) -> Result<()> {
        let backend = ctx
            .selected_backend
            .as_ref()
            .unwrap_or(&self.config.teacher);
        if let Some(host) = &backend.host_header {
            upstream_request.insert_header("Host", host).unwrap();
        }
        upstream_request
            .insert_header("X-DistillForge-Routing", ctx.routing_decision.as_str())
            .unwrap();
        Ok(())
    }

    fn response_body_filter(
        &self,
        _session: &mut Session,
        body: &mut Option<Bytes>,
        _end_of_stream: bool,
        ctx: &mut Self::CTX,
    ) -> Result<Option<std::time::Duration>>
    where
        Self::CTX: Send + Sync,
    {
        if let Some(body) = body {
            ctx.response_body_bytes += body.len();
            append_capture(
                &mut ctx.response_body_capture,
                body,
                self.config.logging.max_capture_bytes,
            );
        }
        Ok(None)
    }

    async fn logging(
        &self,
        session: &mut Session,
        error: Option<&pingora::Error>,
        ctx: &mut Self::CTX,
    ) {
        if ctx.skip_interaction_log {
            return;
        }

        if error.is_some() {
            self.metrics.inc_upstream_error();
        }

        let status = session
            .response_written()
            .map_or(0, |response| response.status.as_u16());
        let request_summary = format!(
            "{} {}",
            session.req_header().method,
            session.req_header().uri.path()
        );
        let error_code = if error.is_some() {
            Some("proxy_error")
        } else if status >= 400 {
            Some(ctx.routing_reason.as_str())
        } else {
            None
        };
        let input_tokens = estimate_tokens(ctx.request_body_bytes);
        let output_tokens = estimate_tokens(ctx.response_body_bytes);
        let selected_backend = ctx.selected_backend.as_ref();
        let estimated_cost_usd = selected_backend
            .map(|backend| estimate_cost_usd(backend, input_tokens, output_tokens))
            .unwrap_or(0.0);
        let estimated_teacher_cost_usd =
            estimate_cost_usd(&self.config.teacher, input_tokens, output_tokens);

        self.logs.write(LogInput {
            metadata: &ctx.metadata,
            task_id: &ctx.task_id,
            teacher_model: &self.config.teacher.name,
            selected_model: &ctx.selected_model,
            routing_decision: &ctx.routing_decision,
            routing_reason: &ctx.routing_reason,
            latency: ctx.started_at.elapsed(),
            input_tokens,
            output_tokens,
            estimated_cost_usd,
            estimated_teacher_cost_usd,
            http_status: status,
            error_code,
            request_summary: &request_summary,
            request_body: &ctx.request_body_capture,
            response_body: &ctx.response_body_capture,
        });

        if error.is_none() && status < 500 {
            if let Some(shadow_backend) = ctx.shadow_backend.clone() {
                self.metrics.inc_shadow();
                spawn_shadow_request(
                    Arc::clone(&self.metrics),
                    Arc::clone(&self.shadow_logs),
                    shadow_backend,
                    session.req_header().uri.path().to_string(),
                    ctx.metadata.request_id.clone(),
                    ctx.task_id.clone(),
                    ctx.response_body_capture.clone(),
                    ctx.request_body_capture.clone(),
                    self.config.logging.max_capture_bytes,
                    self.config.timeouts.shadow_student_timeout_ms,
                );
            }
        }
    }
}

fn main() {
    env_logger::init();

    let config_path = std::env::var("DISTILLFORGE_CONFIG")
        .or_else(|_| std::env::var("DISTILL_PROXY_CONFIG"))
        .unwrap_or_else(|_| "config/example.yaml".to_string());
    let config = match load_config(&config_path) {
        Ok(config) => Arc::new(config),
        Err(err) => {
            error!("{err}");
            std::process::exit(1);
        }
    };

    let logs = match JsonlLogWriter::new(&config.logging.path, config.logging.mode) {
        Ok(logs) => Arc::new(logs),
        Err(err) => {
            error!("{err}");
            std::process::exit(1);
        }
    };
    let shadow_logs = match ShadowLogWriter::new(&config.logging.shadow_path, config.logging.mode) {
        Ok(logs) => Arc::new(logs),
        Err(err) => {
            error!("{err}");
            std::process::exit(1);
        }
    };
    let feedback = match FeedbackWriter::new(&config.logging.feedback_path) {
        Ok(feedback) => Arc::new(feedback),
        Err(err) => {
            error!("{err}");
            std::process::exit(1);
        }
    };
    let metrics = Arc::new(ProxyMetrics::default());
    let routing_snapshot = match load_routing_snapshot(&config.routing.snapshot_path) {
        Ok(snapshot) => {
            info!(
                "loaded routing snapshot {} version={}",
                config.routing.snapshot_path, snapshot.version
            );
            Arc::new(RwLock::new(snapshot))
        }
        Err(err) => {
            error!("{err}; falling back to teacher_only routing");
            Arc::new(RwLock::new(RoutingSnapshot::default()))
        }
    };

    let mut server = Server::new(None).expect("create pingora server");
    server.bootstrap();

    let proxy = DistillProxy {
        config: Arc::clone(&config),
        logs,
        shadow_logs,
        feedback,
        metrics,
        routing_snapshot,
    };
    let mut proxy_service = http_proxy_service(&server.configuration, proxy);
    proxy_service.add_tcp(&config.server.listen_addr);

    info!(
        "DistillForge proxy listening on {}",
        config.server.listen_addr
    );
    server.add_service(proxy_service);
    server.run_forever();
}

fn student_backend(config: &AppConfig, model_id: &str) -> Option<ModelBackendConfig> {
    config
        .students
        .iter()
        .find(|backend| backend.name == model_id)
        .cloned()
}

fn estimate_tokens(bytes: usize) -> u64 {
    if bytes == 0 {
        0
    } else {
        bytes.div_ceil(4) as u64
    }
}

fn estimate_cost_usd(backend: &ModelBackendConfig, input_tokens: u64, output_tokens: u64) -> f64 {
    let input_cost =
        (input_tokens as f64 / 1_000_000.0) * backend.input_cost_per_million_tokens_usd;
    let output_cost =
        (output_tokens as f64 / 1_000_000.0) * backend.output_cost_per_million_tokens_usd;
    input_cost + output_cost
}

fn apply_peer_timeouts(peer: &mut HttpPeer, config: &AppConfig, is_student: bool) {
    peer.options.connection_timeout =
        Some(duration_ms(config.timeouts.upstream_connection_timeout_ms));
    peer.options.read_timeout = Some(duration_ms(if is_student {
        config.timeouts.student_inference_timeout_ms
    } else {
        config.timeouts.teacher_inference_timeout_ms
    }));
    peer.options.write_timeout = Some(duration_ms(config.timeouts.upstream_write_timeout_ms));
}

fn duration_ms(value: u64) -> Duration {
    Duration::from_millis(value.max(1))
}

fn append_capture(capture: &mut Vec<u8>, chunk: &[u8], max_capture_bytes: usize) {
    if capture.len() >= max_capture_bytes {
        return;
    }
    let remaining = max_capture_bytes - capture.len();
    capture.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
}

fn spawn_shadow_request(
    metrics: Arc<ProxyMetrics>,
    shadow_logs: Arc<ShadowLogWriter>,
    backend: ModelBackendConfig,
    path: String,
    request_id: String,
    task_id: String,
    teacher_response_body: Vec<u8>,
    body: Vec<u8>,
    max_capture_bytes: usize,
    timeout_ms: u64,
) {
    thread::spawn(move || {
        let started_at = Instant::now();
        let timeout = duration_ms(timeout_ms);
        let result = if backend.use_tls {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "tls shadow requests are not supported yet",
            ))
        } else {
            send_shadow_request(&backend, &path, &body, max_capture_bytes, timeout)
        };

        match result {
            Ok(response) => {
                let error_code = if response.status < 400 {
                    None
                } else {
                    metrics.inc_shadow_error();
                    Some("shadow_http_status")
                };
                shadow_logs.write(ShadowLogInput {
                    request_id: &request_id,
                    task_id: &task_id,
                    student_model: &backend.name,
                    path: &path,
                    latency: started_at.elapsed(),
                    http_status: Some(response.status),
                    error_code,
                    teacher_response_body: &teacher_response_body,
                    student_response_body: &response.body,
                });
            }
            Err(_) => {
                metrics.inc_shadow_error();
                shadow_logs.write(ShadowLogInput {
                    request_id: &request_id,
                    task_id: &task_id,
                    student_model: &backend.name,
                    path: &path,
                    latency: started_at.elapsed(),
                    http_status: None,
                    error_code: Some("shadow_request_failed"),
                    teacher_response_body: &teacher_response_body,
                    student_response_body: &[],
                });
            }
        }
    });
}

struct ShadowHttpResponse {
    status: u16,
    body: Vec<u8>,
}

fn send_shadow_request(
    backend: &ModelBackendConfig,
    path: &str,
    body: &[u8],
    max_capture_bytes: usize,
    timeout: Duration,
) -> io::Result<ShadowHttpResponse> {
    let mut stream = connect_with_timeout(&backend.address, timeout)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let host = backend.host_header.as_deref().unwrap_or(&backend.address);
    write!(
        stream,
        "POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nX-DistillForge-Shadow: true\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    stream.flush()?;

    let mut response = Vec::new();
    let read_limit = (max_capture_bytes + 8192) as u64;
    stream.take(read_limit).read_to_end(&mut response)?;
    parse_shadow_response(&response, max_capture_bytes)
}

fn parse_shadow_response(
    response: &[u8],
    max_capture_bytes: usize,
) -> io::Result<ShadowHttpResponse> {
    let Some(header_end) = response.windows(4).position(|window| window == b"\r\n\r\n") else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "shadow response missing header terminator",
        ));
    };
    let header = String::from_utf8_lossy(&response[..header_end]);
    let status = header
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|status| status.parse::<u16>().ok())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "shadow response missing HTTP status",
            )
        })?;
    let body_start = header_end + 4;
    let body_end = response.len().min(body_start + max_capture_bytes);
    Ok(ShadowHttpResponse {
        status,
        body: response[body_start..body_end].to_vec(),
    })
}

fn connect_with_timeout(address: &str, timeout: Duration) -> io::Result<TcpStream> {
    let mut addrs = address.to_socket_addrs()?;
    let addr = addrs.next().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("backend address {address} resolved to no socket addresses"),
        )
    })?;
    TcpStream::connect_timeout(&addr, timeout)
}

async fn read_limited_body(session: &mut Session, max_bytes: usize) -> Result<Vec<u8>> {
    let mut body = Vec::new();
    while let Some(chunk) = session.as_downstream_mut().read_request_body().await? {
        if body.len() + chunk.len() > max_bytes {
            let _ = session.as_downstream_mut().drain_request_body().await;
            return Err(Error::because(
                ErrorType::HTTPStatus(413),
                "feedback body too large",
                "feedback body exceeds configured capture limit",
            ));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn header_value(headers: &http::HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

async fn respond_text(session: &mut Session, status: u16, body: &str) -> Result<()> {
    let mut response = ResponseHeader::build(status, Some(2))?;
    response.append_header("Content-Type", "text/plain; charset=utf-8")?;
    response.append_header("Content-Length", body.len().to_string())?;
    session
        .write_response_header(Box::new(response), false)
        .await?;
    session
        .write_response_body(Some(Bytes::copy_from_slice(body.as_bytes())), true)
        .await
}
