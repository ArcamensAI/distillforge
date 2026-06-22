use crate::config::MissingTaskBehavior;
use http::HeaderMap;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;
use uuid::Uuid;

pub const HEADER_TASK_ID: &str = "x-task-id";
pub const HEADER_CLIENT_ID: &str = "x-client-id";
pub const HEADER_REQUEST_ID: &str = "x-request-id";
pub const HEADER_COST_CENTER: &str = "x-cost-center";
pub const HEADER_QUALITY_MODE: &str = "x-quality-mode";
pub const HEADER_NO_TRAIN: &str = "x-no-train";
pub const HEADER_PII_LEVEL: &str = "x-pii-level";

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum RoutingDecision {
    Teacher {
        reason: String,
        task_id: String,
    },
    Student {
        reason: String,
        task_id: String,
        model_id: String,
    },
    Shadow {
        reason: String,
        task_id: String,
        model_id: String,
    },
    Reject {
        status: u16,
        reason: String,
    },
}

#[derive(Debug, Clone, Deserialize, Eq, PartialEq)]
pub struct RoutingSnapshot {
    pub version: u64,
    #[serde(default)]
    pub default_mode: RoutingMode,
    #[serde(default)]
    pub tasks: BTreeMap<String, TaskRoute>,
}

impl Default for RoutingSnapshot {
    fn default() -> Self {
        Self {
            version: 0,
            default_mode: RoutingMode::TeacherOnly,
            tasks: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Eq, PartialEq)]
pub struct TaskRoute {
    #[serde(default)]
    pub mode: RoutingMode,
    #[serde(default)]
    pub student_model: Option<String>,
    #[serde(default)]
    pub student_traffic_percentage: Option<u8>,
    #[serde(default)]
    pub teacher_probe_percentage: Option<u8>,
}

#[derive(Debug, Clone, Copy, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RoutingMode {
    TeacherOnly,
    StudentOnly,
    Shadow,
    Canary,
    Bandit,
}

impl Default for RoutingMode {
    fn default() -> Self {
        Self::TeacherOnly
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RequestMetadata {
    pub request_id: String,
    pub client_id: Option<String>,
    pub task_id: Option<String>,
    pub cost_center: Option<String>,
    pub quality_mode: Option<String>,
    pub pii_level: Option<String>,
    pub no_train: bool,
}

pub fn extract_metadata(headers: &HeaderMap) -> RequestMetadata {
    RequestMetadata {
        request_id: header_value(headers, HEADER_REQUEST_ID)
            .unwrap_or_else(|| Uuid::new_v4().to_string()),
        client_id: header_value(headers, HEADER_CLIENT_ID),
        task_id: header_value(headers, HEADER_TASK_ID),
        cost_center: header_value(headers, HEADER_COST_CENTER),
        quality_mode: header_value(headers, HEADER_QUALITY_MODE),
        pii_level: header_value(headers, HEADER_PII_LEVEL),
        no_train: header_value(headers, HEADER_NO_TRAIN)
            .map(|value| value.eq_ignore_ascii_case("true"))
            .unwrap_or(false),
    }
}

pub fn decide_route(
    metadata: &RequestMetadata,
    missing_task_behavior: MissingTaskBehavior,
    snapshot: &RoutingSnapshot,
) -> RoutingDecision {
    if metadata.client_id.is_none() {
        return RoutingDecision::Reject {
            status: 400,
            reason: "missing_client_id".to_string(),
        };
    }

    let task_id = match metadata.task_id.as_deref() {
        Some(task_id) if !task_id.trim().is_empty() => task_id.to_string(),
        _ => match missing_task_behavior {
            MissingTaskBehavior::Reject => {
                return RoutingDecision::Reject {
                    status: 400,
                    reason: "missing_task_id".to_string(),
                };
            }
            MissingTaskBehavior::TeacherFallback | MissingTaskBehavior::UnknownTask => {
                "unknown_task".to_string()
            }
        },
    };

    if metadata.quality_mode.as_deref() == Some("strict") {
        return RoutingDecision::Teacher {
            reason: "quality_mode_strict".to_string(),
            task_id,
        };
    }

    let task_route = snapshot.tasks.get(&task_id);
    let mode = task_route
        .map(|route| route.mode)
        .unwrap_or(snapshot.default_mode);

    match mode {
        RoutingMode::TeacherOnly => RoutingDecision::Teacher {
            reason: if task_route.is_some() {
                "teacher_only"
            } else {
                "default_teacher_only"
            }
            .to_string(),
            task_id,
        },
        RoutingMode::StudentOnly => choose_student(task_id, task_route, "student_only"),
        RoutingMode::Shadow => choose_shadow(task_id, task_route),
        RoutingMode::Canary => {
            let percentage = task_route
                .and_then(|route| route.student_traffic_percentage)
                .unwrap_or(0)
                .min(100);
            if percentage > 0 && percentage_hit(&metadata.request_id, percentage) {
                choose_student(task_id, task_route, "canary_student")
            } else {
                RoutingDecision::Teacher {
                    reason: "canary_teacher".to_string(),
                    task_id,
                }
            }
        }
        RoutingMode::Bandit => {
            let teacher_probe_percentage = task_route
                .and_then(|route| route.teacher_probe_percentage)
                .unwrap_or(2)
                .min(100);
            if teacher_probe_percentage > 0
                && percentage_hit(&metadata.request_id, teacher_probe_percentage)
            {
                RoutingDecision::Teacher {
                    reason: "bandit_teacher_probe".to_string(),
                    task_id,
                }
            } else {
                choose_student(task_id, task_route, "bandit_student")
            }
        }
    }
}

pub fn load_routing_snapshot(
    path: impl AsRef<Path>,
) -> Result<RoutingSnapshot, RoutingSnapshotError> {
    let path = path.as_ref();
    let path_display = path.display().to_string();
    let contents = fs::read_to_string(path).map_err(|source| RoutingSnapshotError::Read {
        path: path_display.clone(),
        source,
    })?;
    serde_json::from_str(&contents).map_err(|source| RoutingSnapshotError::Parse {
        path: path_display,
        source,
    })
}

#[derive(Debug, thiserror::Error)]
pub enum RoutingSnapshotError {
    #[error("failed to read routing snapshot {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse routing snapshot {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_json::Error,
    },
}

fn choose_shadow(task_id: String, task_route: Option<&TaskRoute>) -> RoutingDecision {
    match task_route.and_then(|route| route.student_model.as_deref()) {
        Some(model_id) => RoutingDecision::Shadow {
            reason: "shadow_teacher".to_string(),
            task_id,
            model_id: model_id.to_string(),
        },
        None => RoutingDecision::Teacher {
            reason: "shadow_student_unavailable_teacher_fallback".to_string(),
            task_id,
        },
    }
}

fn choose_student(
    task_id: String,
    task_route: Option<&TaskRoute>,
    reason: &str,
) -> RoutingDecision {
    match task_route.and_then(|route| route.student_model.as_deref()) {
        Some(model_id) => RoutingDecision::Student {
            reason: reason.to_string(),
            task_id,
            model_id: model_id.to_string(),
        },
        None => RoutingDecision::Teacher {
            reason: "student_unavailable_teacher_fallback".to_string(),
            task_id,
        },
    }
}

fn percentage_hit(request_id: &str, percentage: u8) -> bool {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    request_id.hash(&mut hasher);
    (hasher.finish() % 100) < u64::from(percentage)
}

fn header_value(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::HeaderValue;

    #[test]
    fn extracts_headers_case_insensitively() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "X-Task-ID",
            HeaderValue::from_static("email_classification_v1"),
        );
        headers.insert("X-Client-ID", HeaderValue::from_static("crm_backend"));
        headers.insert("X-No-Train", HeaderValue::from_static("true"));

        let metadata = extract_metadata(&headers);

        assert_eq!(metadata.task_id.as_deref(), Some("email_classification_v1"));
        assert_eq!(metadata.client_id.as_deref(), Some("crm_backend"));
        assert!(metadata.no_train);
    }

    #[test]
    fn routes_to_teacher_for_known_task() {
        let metadata = RequestMetadata {
            request_id: "req_1".to_string(),
            client_id: Some("crm_backend".to_string()),
            task_id: Some("email_classification_v1".to_string()),
            cost_center: None,
            quality_mode: None,
            pii_level: None,
            no_train: false,
        };

        assert_eq!(
            decide_route(
                &metadata,
                MissingTaskBehavior::TeacherFallback,
                &RoutingSnapshot::default()
            ),
            RoutingDecision::Teacher {
                reason: "default_teacher_only".to_string(),
                task_id: "email_classification_v1".to_string()
            }
        );
    }

    #[test]
    fn rejects_missing_client_id() {
        let metadata = RequestMetadata {
            request_id: "req_1".to_string(),
            client_id: None,
            task_id: Some("email_classification_v1".to_string()),
            cost_center: None,
            quality_mode: None,
            pii_level: None,
            no_train: false,
        };

        assert_eq!(
            decide_route(
                &metadata,
                MissingTaskBehavior::TeacherFallback,
                &RoutingSnapshot::default()
            ),
            RoutingDecision::Reject {
                status: 400,
                reason: "missing_client_id".to_string()
            }
        );
    }

    #[test]
    fn routes_student_only_when_configured() {
        let mut tasks = BTreeMap::new();
        tasks.insert(
            "email_classification_v1".to_string(),
            TaskRoute {
                mode: RoutingMode::StudentOnly,
                student_model: Some("local_student".to_string()),
                student_traffic_percentage: None,
                teacher_probe_percentage: None,
            },
        );
        let snapshot = RoutingSnapshot {
            version: 1,
            default_mode: RoutingMode::TeacherOnly,
            tasks,
        };
        let metadata = RequestMetadata {
            request_id: "req_1".to_string(),
            client_id: Some("crm_backend".to_string()),
            task_id: Some("email_classification_v1".to_string()),
            cost_center: None,
            quality_mode: None,
            pii_level: None,
            no_train: false,
        };

        assert_eq!(
            decide_route(&metadata, MissingTaskBehavior::TeacherFallback, &snapshot),
            RoutingDecision::Student {
                reason: "student_only".to_string(),
                task_id: "email_classification_v1".to_string(),
                model_id: "local_student".to_string()
            }
        );
    }

    #[test]
    fn strict_quality_mode_forces_teacher() {
        let metadata = RequestMetadata {
            request_id: "req_1".to_string(),
            client_id: Some("crm_backend".to_string()),
            task_id: Some("email_classification_v1".to_string()),
            cost_center: None,
            quality_mode: Some("strict".to_string()),
            pii_level: None,
            no_train: false,
        };

        assert_eq!(
            decide_route(
                &metadata,
                MissingTaskBehavior::TeacherFallback,
                &RoutingSnapshot::default()
            ),
            RoutingDecision::Teacher {
                reason: "quality_mode_strict".to_string(),
                task_id: "email_classification_v1".to_string()
            }
        );
    }

    #[test]
    fn routes_shadow_to_teacher_with_student_probe() {
        let mut tasks = BTreeMap::new();
        tasks.insert(
            "email_classification_v1".to_string(),
            TaskRoute {
                mode: RoutingMode::Shadow,
                student_model: Some("local_student".to_string()),
                student_traffic_percentage: None,
                teacher_probe_percentage: None,
            },
        );
        let snapshot = RoutingSnapshot {
            version: 1,
            default_mode: RoutingMode::TeacherOnly,
            tasks,
        };
        let metadata = RequestMetadata {
            request_id: "req_1".to_string(),
            client_id: Some("crm_backend".to_string()),
            task_id: Some("email_classification_v1".to_string()),
            cost_center: None,
            quality_mode: None,
            pii_level: None,
            no_train: false,
        };

        assert_eq!(
            decide_route(&metadata, MissingTaskBehavior::TeacherFallback, &snapshot),
            RoutingDecision::Shadow {
                reason: "shadow_teacher".to_string(),
                task_id: "email_classification_v1".to_string(),
                model_id: "local_student".to_string()
            }
        );
    }

    #[test]
    fn routes_bandit_to_student_by_default() {
        let mut tasks = BTreeMap::new();
        tasks.insert(
            "email_classification_v1".to_string(),
            TaskRoute {
                mode: RoutingMode::Bandit,
                student_model: Some("local_student".to_string()),
                student_traffic_percentage: None,
                teacher_probe_percentage: Some(0),
            },
        );
        let snapshot = RoutingSnapshot {
            version: 1,
            default_mode: RoutingMode::TeacherOnly,
            tasks,
        };
        let metadata = RequestMetadata {
            request_id: "req_1".to_string(),
            client_id: Some("crm_backend".to_string()),
            task_id: Some("email_classification_v1".to_string()),
            cost_center: None,
            quality_mode: None,
            pii_level: None,
            no_train: false,
        };

        assert_eq!(
            decide_route(&metadata, MissingTaskBehavior::TeacherFallback, &snapshot),
            RoutingDecision::Student {
                reason: "bandit_student".to_string(),
                task_id: "email_classification_v1".to_string(),
                model_id: "local_student".to_string()
            }
        );
    }

    #[test]
    fn routes_bandit_teacher_probe() {
        let mut tasks = BTreeMap::new();
        tasks.insert(
            "email_classification_v1".to_string(),
            TaskRoute {
                mode: RoutingMode::Bandit,
                student_model: Some("local_student".to_string()),
                student_traffic_percentage: None,
                teacher_probe_percentage: Some(100),
            },
        );
        let snapshot = RoutingSnapshot {
            version: 1,
            default_mode: RoutingMode::TeacherOnly,
            tasks,
        };
        let metadata = RequestMetadata {
            request_id: "req_1".to_string(),
            client_id: Some("crm_backend".to_string()),
            task_id: Some("email_classification_v1".to_string()),
            cost_center: None,
            quality_mode: None,
            pii_level: None,
            no_train: false,
        };

        assert_eq!(
            decide_route(&metadata, MissingTaskBehavior::TeacherFallback, &snapshot),
            RoutingDecision::Teacher {
                reason: "bandit_teacher_probe".to_string(),
                task_id: "email_classification_v1".to_string()
            }
        );
    }

    #[test]
    fn parses_example_snapshot() {
        let snapshot =
            load_routing_snapshot("config/routing_snapshot.json").expect("snapshot should parse");

        assert_eq!(snapshot.version, 1);
        assert_eq!(snapshot.default_mode, RoutingMode::TeacherOnly);
    }
}
