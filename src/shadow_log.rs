use crate::config::LogMode;
use crate::log_writer::redact_text;
use chrono::{DateTime, Utc};
use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum ShadowLogWriterError {
    #[error("failed to create shadow log directory {path}: {source}")]
    CreateDir {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to open shadow log file {path}: {source}")]
    Open {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

#[derive(Debug)]
pub struct ShadowLogWriter {
    mode: LogMode,
    writer: Mutex<BufWriter<File>>,
}

#[derive(Debug, Serialize)]
struct ShadowProbeLog {
    timestamp: DateTime<Utc>,
    request_id: String,
    task_id: String,
    student_model: String,
    path: String,
    latency_ms: u128,
    http_status: Option<u16>,
    status: String,
    error_code: Option<String>,
    teacher_response_redacted: Option<String>,
    student_response_redacted: Option<String>,
    response_exact_match: Option<bool>,
}

pub struct ShadowLogInput<'a> {
    pub request_id: &'a str,
    pub task_id: &'a str,
    pub student_model: &'a str,
    pub path: &'a str,
    pub latency: Duration,
    pub http_status: Option<u16>,
    pub error_code: Option<&'a str>,
    pub teacher_response_body: &'a [u8],
    pub student_response_body: &'a [u8],
}

impl ShadowLogWriter {
    pub fn new(path: impl AsRef<Path>, mode: LogMode) -> Result<Self, ShadowLogWriterError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| ShadowLogWriterError::CreateDir {
                path: parent.display().to_string(),
                source,
            })?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|source| ShadowLogWriterError::Open {
                path: path.display().to_string(),
                source,
            })?;
        Ok(Self {
            mode,
            writer: Mutex::new(BufWriter::new(file)),
        })
    }

    pub fn write(&self, input: ShadowLogInput<'_>) {
        if self.mode == LogMode::Disabled {
            return;
        }

        let teacher_response_redacted = redacted_content(input.teacher_response_body, self.mode);
        let student_response_redacted = redacted_content(input.student_response_body, self.mode);
        let response_exact_match = teacher_response_redacted
            .as_ref()
            .zip(student_response_redacted.as_ref())
            .map(|(teacher, student)| teacher == student);
        let status = if input.error_code.is_none()
            && input
                .http_status
                .map(|status| status < 400)
                .unwrap_or(false)
        {
            "success"
        } else {
            "error"
        };

        let entry = ShadowProbeLog {
            timestamp: Utc::now(),
            request_id: input.request_id.to_string(),
            task_id: input.task_id.to_string(),
            student_model: input.student_model.to_string(),
            path: input.path.to_string(),
            latency_ms: input.latency.as_millis(),
            http_status: input.http_status,
            status: status.to_string(),
            error_code: input.error_code.map(ToOwned::to_owned),
            teacher_response_redacted,
            student_response_redacted,
            response_exact_match,
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

fn redacted_content(body: &[u8], mode: LogMode) -> Option<String> {
    if body.is_empty() || mode != LogMode::Redacted {
        return None;
    }

    let text = String::from_utf8_lossy(body);
    Some(redact_text(&text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_shadow_comparison_log() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("shadow.jsonl");
        let writer = ShadowLogWriter::new(&path, LogMode::Redacted).unwrap();

        writer.write(ShadowLogInput {
            request_id: "req_1",
            task_id: "email_classification_v1",
            student_model: "local_student",
            path: "/v1/chat/completions",
            latency: Duration::from_millis(8),
            http_status: Some(200),
            error_code: None,
            teacher_response_body: br#"{"result":"billing","email":"alice@example.com"}"#,
            student_response_body: br#"{"result":"billing","email":"alice@example.com"}"#,
        });

        let contents = std::fs::read_to_string(path).unwrap();
        assert!(contents.contains("\"request_id\":\"req_1\""));
        assert!(contents.contains("\"response_exact_match\":true"));
        assert!(contents.contains("[REDACTED_EMAIL]"));
        assert!(!contents.contains("alice@example.com"));
    }
}
