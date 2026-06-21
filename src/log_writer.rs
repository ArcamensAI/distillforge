use crate::config::LogMode;
use crate::routing::RequestMetadata;
use chrono::{DateTime, Utc};
use regex::Regex;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum LogWriterError {
    #[error("failed to create log directory {path}: {source}")]
    CreateDir {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to open log file {path}: {source}")]
    Open {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

#[derive(Debug)]
pub struct JsonlLogWriter {
    mode: LogMode,
    writer: Mutex<BufWriter<File>>,
}

#[derive(Debug, Serialize)]
pub struct InteractionLog {
    timestamp: DateTime<Utc>,
    request_id: String,
    client_id: Option<String>,
    task_id: String,
    cost_center: Option<String>,
    quality_mode: Option<String>,
    input_hash: Option<String>,
    prompt_redacted: Option<String>,
    response_redacted: Option<String>,
    teacher_model: String,
    selected_model: String,
    routing_decision: String,
    routing_reason: String,
    latency_ms: u128,
    input_tokens: u64,
    output_tokens: u64,
    estimated_cost_usd: f64,
    estimated_teacher_cost_usd: f64,
    estimated_savings_usd: f64,
    status: String,
    http_status: u16,
    error_code: Option<String>,
    pii_level: Option<String>,
    training_eligible: bool,
}

pub struct LogInput<'a> {
    pub metadata: &'a RequestMetadata,
    pub task_id: &'a str,
    pub teacher_model: &'a str,
    pub selected_model: &'a str,
    pub routing_decision: &'a str,
    pub routing_reason: &'a str,
    pub latency: Duration,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub estimated_cost_usd: f64,
    pub estimated_teacher_cost_usd: f64,
    pub http_status: u16,
    pub error_code: Option<&'a str>,
    pub request_summary: &'a str,
    pub request_body: &'a [u8],
    pub response_body: &'a [u8],
}

impl JsonlLogWriter {
    pub fn new(path: impl AsRef<Path>, mode: LogMode) -> Result<Self, LogWriterError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| LogWriterError::CreateDir {
                path: parent.display().to_string(),
                source,
            })?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|source| LogWriterError::Open {
                path: path.display().to_string(),
                source,
            })?;

        Ok(Self {
            mode,
            writer: Mutex::new(BufWriter::new(file)),
        })
    }

    pub fn write(&self, input: LogInput<'_>) {
        if self.mode == LogMode::Disabled {
            return;
        }

        let entry = InteractionLog {
            timestamp: Utc::now(),
            request_id: input.metadata.request_id.clone(),
            client_id: input.metadata.client_id.clone(),
            task_id: input.task_id.to_string(),
            cost_center: input.metadata.cost_center.clone(),
            quality_mode: input.metadata.quality_mode.clone(),
            input_hash: content_hash(input.request_body, input.request_summary, self.mode),
            prompt_redacted: redacted_content(input.request_body, self.mode),
            response_redacted: redacted_content(input.response_body, self.mode),
            teacher_model: input.teacher_model.to_string(),
            selected_model: input.selected_model.to_string(),
            routing_decision: input.routing_decision.to_string(),
            routing_reason: input.routing_reason.to_string(),
            latency_ms: input.latency.as_millis(),
            input_tokens: input.input_tokens,
            output_tokens: input.output_tokens,
            estimated_cost_usd: input.estimated_cost_usd,
            estimated_teacher_cost_usd: input.estimated_teacher_cost_usd,
            estimated_savings_usd: (input.estimated_teacher_cost_usd - input.estimated_cost_usd)
                .max(0.0),
            status: if input.error_code.is_none() && input.http_status < 400 {
                "success".to_string()
            } else {
                "error".to_string()
            },
            http_status: input.http_status,
            error_code: input.error_code.map(ToOwned::to_owned),
            pii_level: input.metadata.pii_level.clone(),
            training_eligible: input.http_status < 400
                && !input.metadata.no_train
                && input.metadata.pii_level.as_deref() != Some("high")
                && matches!(self.mode, LogMode::Redacted | LogMode::FullEncrypted),
        };

        let Ok(line) = serde_json::to_string(&entry) else {
            return;
        };
        if let Ok(mut writer) = self.writer.lock() {
            let _ = writeln!(writer, "{line}");
            let _ = writer.flush();
        }
    }
}

fn content_hash(body: &[u8], fallback: &str, mode: LogMode) -> Option<String> {
    match mode {
        LogMode::MetadataOnly | LogMode::Disabled => None,
        LogMode::Redacted | LogMode::FullEncrypted => {
            let content = if body.is_empty() {
                fallback.as_bytes()
            } else {
                body
            };
            Some(format!("sha256:{}", sha256_hex(content)))
        }
    }
}

fn redacted_content(body: &[u8], mode: LogMode) -> Option<String> {
    if body.is_empty() || mode != LogMode::Redacted {
        return None;
    }

    let text = String::from_utf8_lossy(body);
    Some(redact_text(&text))
}

pub fn redact_text(text: &str) -> String {
    let email_re = Regex::new(r"(?i)[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}").unwrap();
    let secret_re = Regex::new(
        r#"(?i)("?(?:api[_-]?key|password|secret|token|authorization)"?\s*:\s*")([^"]*)(")"#,
    )
    .unwrap();
    let bearer_re = Regex::new(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+").unwrap();
    let openai_key_re = Regex::new(r"\bsk-[A-Za-z0-9_-]{12,}\b").unwrap();

    let text = email_re.replace_all(text, "[REDACTED_EMAIL]");
    let text = secret_re.replace_all(&text, "$1[REDACTED]$3");
    let text = bearer_re.replace_all(&text, "Bearer [REDACTED]");
    openai_key_re
        .replace_all(&text, "sk-[REDACTED]")
        .into_owned()
}

fn sha256_hex(input: &[u8]) -> String {
    let digest = Sha256::digest(input);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_jsonl_entry() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("proxy.jsonl");
        let writer = JsonlLogWriter::new(&path, LogMode::Redacted).unwrap();
        let metadata = RequestMetadata {
            request_id: "req_1".to_string(),
            client_id: Some("crm_backend".to_string()),
            task_id: Some("email_classification_v1".to_string()),
            cost_center: Some("support".to_string()),
            quality_mode: Some("balanced".to_string()),
            pii_level: Some("low".to_string()),
            no_train: false,
        };

        writer.write(LogInput {
            metadata: &metadata,
            task_id: "email_classification_v1",
            teacher_model: "teacher",
            selected_model: "teacher",
            routing_decision: "teacher",
            routing_reason: "teacher_only_v1",
            latency: Duration::from_millis(12),
            input_tokens: 16,
            output_tokens: 4,
            estimated_cost_usd: 0.001,
            estimated_teacher_cost_usd: 0.001,
            http_status: 200,
            error_code: None,
            request_summary: "POST /v1/chat/completions",
            request_body: br#"{"email":"alice@example.com","password":"secret"}"#,
            response_body: br#"{"result":"ok","token":"abc"}"#,
        });

        let contents = std::fs::read_to_string(path).unwrap();
        assert!(contents.contains("\"request_id\":\"req_1\""));
        assert!(contents.contains("\"training_eligible\":true"));
        assert!(contents.contains("\"input_tokens\":16"));
        assert!(contents.contains("\"estimated_cost_usd\":0.001"));
        assert!(contents.contains("[REDACTED_EMAIL]"));
        assert!(!contents.contains("alice@example.com"));
        assert!(!contents.contains("secret"));
    }

    #[test]
    fn metadata_only_does_not_store_content_or_hash() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("proxy.jsonl");
        let writer = JsonlLogWriter::new(&path, LogMode::MetadataOnly).unwrap();
        let metadata = RequestMetadata {
            request_id: "req_1".to_string(),
            client_id: Some("crm_backend".to_string()),
            task_id: Some("email_classification_v1".to_string()),
            cost_center: None,
            quality_mode: None,
            pii_level: None,
            no_train: false,
        };

        writer.write(LogInput {
            metadata: &metadata,
            task_id: "email_classification_v1",
            teacher_model: "teacher",
            selected_model: "teacher",
            routing_decision: "teacher",
            routing_reason: "teacher_only",
            latency: Duration::from_millis(12),
            input_tokens: 16,
            output_tokens: 4,
            estimated_cost_usd: 0.001,
            estimated_teacher_cost_usd: 0.001,
            http_status: 200,
            error_code: None,
            request_summary: "POST /v1/chat/completions",
            request_body: br#"{"email":"alice@example.com"}"#,
            response_body: br#"{"result":"ok"}"#,
        });

        let contents = std::fs::read_to_string(path).unwrap();
        assert!(contents.contains("\"input_hash\":null"));
        assert!(contents.contains("\"prompt_redacted\":null"));
        assert!(contents.contains("\"response_redacted\":null"));
    }
}
