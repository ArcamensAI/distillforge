use serde::Deserialize;
use std::collections::BTreeMap;
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
    #[serde(default)]
    pub rate_limits: RateLimitsConfig,
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

#[derive(Debug, Clone, Deserialize, Eq, PartialEq)]
pub struct RateLimitsConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_rate_limit_window_ms")]
    pub window_ms: u64,
    #[serde(default)]
    pub default_requests_per_window: Option<u64>,
    #[serde(default)]
    pub clients: BTreeMap<String, u64>,
    #[serde(default)]
    pub tasks: BTreeMap<String, u64>,
}

impl Default for RateLimitsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            window_ms: default_rate_limit_window_ms(),
            default_requests_per_window: None,
            clients: BTreeMap::new(),
            tasks: BTreeMap::new(),
        }
    }
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

fn default_rate_limit_window_ms() -> u64 {
    60_000
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
        assert!(config.rate_limits.enabled);
        assert_eq!(config.rate_limits.window_ms, 60_000);
        assert_eq!(config.rate_limits.default_requests_per_window, Some(120));
        assert_eq!(
            config.rate_limits.clients.get("crm_backend").copied(),
            Some(600)
        );
        assert_eq!(
            config
                .rate_limits
                .tasks
                .get("email_classification_v1")
                .copied(),
            Some(300)
        );
    }

    #[test]
    fn parses_groq_banking77_volume_config() {
        let config = load_config("examples/groq_banking77/config.volume.yaml")
            .expect("volume demo config should parse");

        assert_eq!(config.teacher.address, "127.0.0.1:9200");
        assert_eq!(
            config.logging.path,
            "examples/groq_banking77/data_volume/logs/proxy.jsonl"
        );
        assert_eq!(
            config.routing.snapshot_path,
            "examples/groq_banking77/routing_snapshot.volume.json"
        );
        assert_eq!(config.rate_limits.default_requests_per_window, Some(20));
    }

    #[test]
    fn parses_groq_banking77_local_10k_config() {
        let config = load_config("examples/groq_banking77/config.local_10k.yaml")
            .expect("local 10k demo config should parse");

        assert_eq!(config.teacher.name, "local_embedding_banking_teacher");
        assert_eq!(config.teacher.address, "127.0.0.1:9300");
        assert_eq!(
            config.logging.path,
            "examples/groq_banking77/data_local_10k/logs/proxy.jsonl"
        );
        assert_eq!(
            config.routing.snapshot_path,
            "examples/groq_banking77/routing_snapshot.local_10k.json"
        );
        assert!(!config.rate_limits.enabled);
        assert_eq!(config.rate_limits.default_requests_per_window, Some(1200));
    }

    #[test]
    fn parses_groq_banking77_local_llm_config() {
        let config = load_config("examples/groq_banking77/config.local_llm.yaml")
            .expect("local LLM demo config should parse");

        assert_eq!(config.teacher.name, "qwen3_8b_master");
        assert_eq!(config.teacher.address, "127.0.0.1:9400");
        assert_eq!(config.students[0].name, "qwen2_5_1_5b_student");
        assert_eq!(config.students[0].address, "127.0.0.1:9500");
        assert_eq!(
            config.routing.snapshot_path,
            "examples/groq_banking77/routing_snapshot.local_llm.json"
        );
        assert!(!config.rate_limits.enabled);
    }

    #[test]
    fn parses_cfpb_complaints_local_llm_config() {
        let config = load_config("examples/cfpb_complaints/config.local_llm.yaml")
            .expect("CFPB local LLM demo config should parse");

        assert_eq!(config.teacher.name, "qwen3_8b_cfpb_teacher");
        assert_eq!(config.teacher.address, "127.0.0.1:9600");
        assert_eq!(config.students[0].name, "cfpb_product_student_hybrid_bge_m3");
        assert_eq!(config.students[0].address, "127.0.0.1:9102");
        assert_eq!(
            config.routing.snapshot_path,
            "examples/cfpb_complaints/routing_snapshot.local_llm.json"
        );
        assert!(!config.rate_limits.enabled);
    }
}
