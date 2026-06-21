use async_trait::async_trait;
use distillforge::config::{load_config, AppConfig};
use distillforge::log_writer::{JsonlLogWriter, LogInput};
use distillforge::metrics::ProxyMetrics;
use distillforge::routing::{decide_route, extract_metadata, RequestMetadata, RoutingDecision};
use log::{error, info};
use pingora::prelude::*;
use pingora::proxy::{http_proxy_service, ProxyHttp, Session};
use pingora::http::ResponseHeader;
use bytes::Bytes;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone)]
struct DistillProxy {
    config: Arc<AppConfig>,
    logs: Arc<JsonlLogWriter>,
    metrics: Arc<ProxyMetrics>,
}

#[derive(Debug, Clone)]
struct RequestContext {
    started_at: Instant,
    metadata: RequestMetadata,
    decision: Option<RoutingDecision>,
    task_id: String,
    routing_reason: &'static str,
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
            routing_reason: "not_routed",
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
            respond_text(session, 501, "routing reload is not implemented in V1\n").await?;
            return Ok(true);
        }

        if path != "/v1/chat/completions" && path != "/v1/completions" {
            self.metrics.inc_rejected();
            ctx.routing_reason = "unsupported_path";
            let _ = session.respond_error(404).await;
            return Ok(true);
        }

        ctx.metadata = extract_metadata(&session.req_header().headers);
        let decision = decide_route(
            &ctx.metadata,
            self.config.routing.default_missing_task_behavior,
        );

        match &decision {
            RoutingDecision::Teacher { reason, task_id } => {
                self.metrics.inc_teacher();
                ctx.task_id = task_id.clone();
                ctx.routing_reason = reason;
                ctx.decision = Some(decision);
                Ok(false)
            }
            RoutingDecision::Reject { status, reason } => {
                self.metrics.inc_rejected();
                ctx.routing_reason = reason;
                ctx.decision = Some(decision.clone());
                let _ = session.respond_error(*status).await;
                Ok(true)
            }
        }
    }

    async fn upstream_peer(&self, _session: &mut Session, _ctx: &mut Self::CTX) -> Result<Box<HttpPeer>> {
        let teacher = &self.config.teacher;
        let peer = Box::new(HttpPeer::new(
            teacher.address.as_str(),
            teacher.use_tls,
            teacher.sni.clone(),
        ));
        Ok(peer)
    }

    async fn upstream_request_filter(
        &self,
        _session: &mut Session,
        upstream_request: &mut RequestHeader,
        _ctx: &mut Self::CTX,
    ) -> Result<()> {
        if let Some(host) = &self.config.teacher.host_header {
            upstream_request.insert_header("Host", host).unwrap();
        }
        upstream_request
            .insert_header("X-DistillForge-Routing", "teacher")
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
        let rejected = matches!(ctx.decision, Some(RoutingDecision::Reject { .. }));
        let routing_decision = if rejected { "reject" } else { "teacher" };
        let selected_model = if rejected {
            "none"
        } else {
            &self.config.teacher.name
        };
        let error_code = if error.is_some() {
            Some("proxy_error")
        } else if status >= 400 {
            Some(ctx.routing_reason)
        } else {
            None
        };

        self.logs.write(LogInput {
            metadata: &ctx.metadata,
            task_id: &ctx.task_id,
            teacher_model: &self.config.teacher.name,
            selected_model,
            routing_decision,
            routing_reason: ctx.routing_reason,
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

    let mut server = Server::new(None).expect("create pingora server");
    server.bootstrap();

    let proxy = DistillProxy {
        config: Arc::clone(&config),
        logs,
        metrics,
    };
    let mut proxy_service = http_proxy_service(&server.configuration, proxy);
    proxy_service.add_tcp(&config.server.listen_addr);

    info!("DistillForge proxy listening on {}", config.server.listen_addr);
    server.add_service(proxy_service);
    server.run_forever();
}

async fn respond_text(session: &mut Session, status: u16, body: &str) -> Result<()> {
    let mut response = ResponseHeader::build(status, Some(2))?;
    response.append_header("Content-Type", "text/plain; charset=utf-8")?;
    response.append_header("Content-Length", body.len().to_string())?;
    session.write_response_header(Box::new(response), false).await?;
    session
        .write_response_body(Some(Bytes::copy_from_slice(body.as_bytes())), true)
        .await
}
