use async_trait::async_trait;
use bytes::Bytes;
use distillforge::config::{load_config, AppConfig, ModelBackendConfig};
use distillforge::log_writer::{JsonlLogWriter, LogInput};
use distillforge::metrics::ProxyMetrics;
use distillforge::routing::{
    decide_route, extract_metadata, load_routing_snapshot, RequestMetadata, RoutingDecision,
    RoutingSnapshot,
};
use log::{error, info};
use pingora::http::ResponseHeader;
use pingora::prelude::*;
use pingora::proxy::{http_proxy_service, ProxyHttp, Session};
use std::sync::{Arc, RwLock};
use std::time::Instant;

#[derive(Clone)]
struct DistillProxy {
    config: Arc<AppConfig>,
    logs: Arc<JsonlLogWriter>,
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

    async fn upstream_peer(
        &self,
        _session: &mut Session,
        ctx: &mut Self::CTX,
    ) -> Result<Box<HttpPeer>> {
        let backend = ctx
            .selected_backend
            .as_ref()
            .unwrap_or(&self.config.teacher);
        let peer = Box::new(HttpPeer::new(
            backend.address.as_str(),
            backend.use_tls,
            backend.sni.clone(),
        ));
        Ok(peer)
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

        self.logs.write(LogInput {
            metadata: &ctx.metadata,
            task_id: &ctx.task_id,
            teacher_model: &self.config.teacher.name,
            selected_model: &ctx.selected_model,
            routing_decision: &ctx.routing_decision,
            routing_reason: &ctx.routing_reason,
            latency: ctx.started_at.elapsed(),
            http_status: status,
            error_code,
            request_summary: &request_summary,
        });
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
