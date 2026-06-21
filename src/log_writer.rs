use crate::config::LogMode;
use crate::routing::RequestMetadata;
use chrono::{DateTime, Utc};
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
    teacher_model: String,
    selected_model: String,
    routing_decision: String,
    routing_reason: String,
    latency_ms: u128,
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
    pub http_status: u16,
    pub error_code: Option<&'a str>,
    pub request_summary: &'a str,
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
            input_hash: match self.mode {
                LogMode::MetadataOnly | LogMode::Disabled => None,
                LogMode::Redacted | LogMode::FullEncrypted => Some(format!(
                    "sha256:{}",
                    sha256_hex(input.request_summary.as_bytes())
                )),
            },
            teacher_model: input.teacher_model.to_string(),
            selected_model: input.selected_model.to_string(),
            routing_decision: input.routing_decision.to_string(),
            routing_reason: input.routing_reason.to_string(),
            latency_ms: input.latency.as_millis(),
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
            http_status: 200,
            error_code: None,
            request_summary: "POST /v1/chat/completions",
        });

        let contents = std::fs::read_to_string(path).unwrap();
        assert!(contents.contains("\"request_id\":\"req_1\""));
        assert!(contents.contains("\"training_eligible\":true"));
    }
}
