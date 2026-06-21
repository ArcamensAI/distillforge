use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub teacher: TeacherConfig,
    #[serde(default)]
    pub students: Vec<ModelBackendConfig>,
    #[serde(default)]
    pub timeouts: TimeoutsConfig,
    pub logging: LoggingConfig,
    pub routing: RoutingConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_listen_addr")]
    pub listen_addr: String,
    #[serde(default = "default_metrics_addr")]
    pub metrics_addr: String,
}

pub type TeacherConfig = ModelBackendConfig;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelBackendConfig {
    pub name: String,
    pub address: String,
    #[serde(default)]
    pub use_tls: bool,
    #[serde(default)]
    pub sni: String,
    #[serde(default)]
    pub host_header: Option<String>,
    #[serde(default)]
    pub input_cost_per_million_tokens_usd: f64,
    #[serde(default)]
    pub output_cost_per_million_tokens_usd: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LoggingConfig {
    pub path: String,
    #[serde(default = "default_feedback_path")]
    pub feedback_path: String,
    #[serde(default = "default_shadow_path")]
    pub shadow_path: String,
    #[serde(default)]
    pub mode: LogMode,
    #[serde(default = "default_max_capture_bytes")]
    pub max_capture_bytes: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TimeoutsConfig {
    #[serde(default = "default_upstream_connection_timeout_ms")]
    pub upstream_connection_timeout_ms: u64,
    #[serde(default = "default_teacher_inference_timeout_ms")]
    pub teacher_inference_timeout_ms: u64,
    #[serde(default = "default_student_inference_timeout_ms")]
    pub student_inference_timeout_ms: u64,
    #[serde(default = "default_upstream_write_timeout_ms")]
    pub upstream_write_timeout_ms: u64,
    #[serde(default = "default_shadow_student_timeout_ms")]
    pub shadow_student_timeout_ms: u64,
}

impl Default for TimeoutsConfig {
    fn default() -> Self {
        Self {
            upstream_connection_timeout_ms: default_upstream_connection_timeout_ms(),
            teacher_inference_timeout_ms: default_teacher_inference_timeout_ms(),
            student_inference_timeout_ms: default_student_inference_timeout_ms(),
            upstream_write_timeout_ms: default_upstream_write_timeout_ms(),
            shadow_student_timeout_ms: default_shadow_student_timeout_ms(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct RoutingConfig {
    #[serde(default)]
    pub default_missing_task_behavior: MissingTaskBehavior,
    #[serde(default = "default_snapshot_path")]
    pub snapshot_path: String,
}

#[derive(Debug, Clone, Copy, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum LogMode {
    MetadataOnly,
    Redacted,
    FullEncrypted,
    Disabled,
}

impl Default for LogMode {
    fn default() -> Self {
        Self::Redacted
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MissingTaskBehavior {
    Reject,
    TeacherFallback,
    UnknownTask,
}

impl Default for MissingTaskBehavior {
    fn default() -> Self {
        Self::TeacherFallback
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse config {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_yaml::Error,
    },
}

pub fn load_config(path: impl AsRef<Path>) -> Result<AppConfig, ConfigError> {
    let path = path.as_ref();
    let path_display = path.display().to_string();
    let contents = fs::read_to_string(path).map_err(|source| ConfigError::Read {
        path: path_display.clone(),
        source,
    })?;
    serde_yaml::from_str(&contents).map_err(|source| ConfigError::Parse {
        path: path_display,
        source,
    })
}

fn default_listen_addr() -> String {
    "127.0.0.1:6188".to_string()
}

fn default_metrics_addr() -> String {
    "127.0.0.1:6192".to_string()
}

fn default_snapshot_path() -> String {
    "config/routing_snapshot.json".to_string()
}

fn default_feedback_path() -> String {
    "data/logs/feedback.jsonl".to_string()
}

fn default_shadow_path() -> String {
    "data/logs/shadow.jsonl".to_string()
}

fn default_max_capture_bytes() -> usize {
    64 * 1024
}

fn default_upstream_connection_timeout_ms() -> u64 {
    2_000
}

fn default_teacher_inference_timeout_ms() -> u64 {
    30_000
}

fn default_student_inference_timeout_ms() -> u64 {
    2_000
}

fn default_upstream_write_timeout_ms() -> u64 {
    30_000
}

fn default_shadow_student_timeout_ms() -> u64 {
    5_000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_example_config() {
        let config = load_config("config/example.yaml").expect("example config should parse");

        assert_eq!(config.server.listen_addr, "127.0.0.1:6188");
        assert_eq!(config.teacher.address, "127.0.0.1:9000");
        assert_eq!(config.students.len(), 1);
        assert_eq!(config.timeouts.upstream_connection_timeout_ms, 2_000);
        assert_eq!(config.timeouts.teacher_inference_timeout_ms, 30_000);
        assert_eq!(config.timeouts.student_inference_timeout_ms, 2_000);
        assert_eq!(config.timeouts.shadow_student_timeout_ms, 5_000);
        assert_eq!(config.routing.snapshot_path, "config/routing_snapshot.json");
        assert_eq!(config.logging.mode, LogMode::Redacted);
        assert_eq!(config.logging.feedback_path, "data/logs/feedback.jsonl");
        assert_eq!(config.logging.shadow_path, "data/logs/shadow.jsonl");
        assert_eq!(config.logging.max_capture_bytes, 65_536);
        assert_eq!(
            config.routing.default_missing_task_behavior,
            MissingTaskBehavior::TeacherFallback
        );
    }
}
