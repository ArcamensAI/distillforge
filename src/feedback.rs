use crate::log_writer::redact_text;
use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::Value;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;

#[derive(Debug, thiserror::Error)]
pub enum FeedbackWriterError {
    #[error("failed to create feedback log directory {path}: {source}")]
    CreateDir {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to open feedback log file {path}: {source}")]
    Open {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

#[derive(Debug)]
pub struct FeedbackWriter {
    writer: Mutex<BufWriter<File>>,
}

#[derive(Debug, Serialize)]
pub struct FeedbackLog {
    timestamp: DateTime<Utc>,
    request_id: Option<String>,
    client_id: Option<String>,
    task_id: Option<String>,
    rating: Option<String>,
    correct_output_redacted: Option<String>,
    comment_redacted: Option<String>,
}

impl FeedbackWriter {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, FeedbackWriterError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| FeedbackWriterError::CreateDir {
                path: parent.display().to_string(),
                source,
            })?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|source| FeedbackWriterError::Open {
                path: path.display().to_string(),
                source,
            })?;
        Ok(Self {
            writer: Mutex::new(BufWriter::new(file)),
        })
    }

    pub fn write_value(&self, value: &Value, client_id: Option<String>, task_id: Option<String>) {
        let entry = FeedbackLog {
            timestamp: Utc::now(),
            request_id: string_field(value, "request_id"),
            client_id,
            task_id,
            rating: string_field(value, "rating"),
            correct_output_redacted: redacted_string_field(value, "correct_output"),
            comment_redacted: redacted_string_field(value, "comment"),
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

fn string_field(value: &Value, field: &str) -> Option<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn redacted_string_field(value: &Value, field: &str) -> Option<String> {
    string_field(value, field).map(|text| redact_text(&text))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn writes_redacted_feedback() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("feedback.jsonl");
        let writer = FeedbackWriter::new(&path).unwrap();

        writer.write_value(
            &json!({
                "request_id": "req_1",
                "rating": "bad",
                "correct_output": "alice@example.com",
                "comment": "token sk-testSECRET123456"
            }),
            Some("crm_backend".to_string()),
            Some("email_classification_v1".to_string()),
        );

        let contents = std::fs::read_to_string(path).unwrap();
        assert!(contents.contains("\"request_id\":\"req_1\""));
        assert!(contents.contains("[REDACTED_EMAIL]"));
        assert!(contents.contains("sk-[REDACTED]"));
        assert!(!contents.contains("alice@example.com"));
        assert!(!contents.contains("sk-testSECRET123456"));
    }
}
