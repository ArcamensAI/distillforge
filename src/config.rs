use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub teacher: TeacherConfig,
    #[serde(default)]
    pub students: Vec<ModelBackendConfig>,
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
}

#[derive(Debug, Clone, Deserialize)]
pub struct LoggingConfig {
    pub path: String,
    #[serde(default)]
    pub mode: LogMode,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_example_config() {
        let config = load_config("config/example.yaml").expect("example config should parse");

        assert_eq!(config.server.listen_addr, "127.0.0.1:6188");
        assert_eq!(config.teacher.address, "127.0.0.1:9000");
        assert_eq!(config.students.len(), 1);
        assert_eq!(config.routing.snapshot_path, "config/routing_snapshot.json");
        assert_eq!(config.logging.mode, LogMode::Redacted);
        assert_eq!(
            config.routing.default_missing_task_behavior,
            MissingTaskBehavior::TeacherFallback
        );
    }
}
