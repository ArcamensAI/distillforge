use crate::config::MissingTaskBehavior;
use http::HeaderMap;
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
        reason: &'static str,
        task_id: String,
    },
    Reject {
        status: u16,
        reason: &'static str,
    },
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
) -> RoutingDecision {
    if metadata.client_id.is_none() {
        return RoutingDecision::Reject {
            status: 400,
            reason: "missing_client_id",
        };
    }

    match metadata.task_id.as_deref() {
        Some(task_id) if !task_id.trim().is_empty() => RoutingDecision::Teacher {
            reason: "teacher_only_v1",
            task_id: task_id.to_string(),
        },
        _ => match missing_task_behavior {
            MissingTaskBehavior::Reject => RoutingDecision::Reject {
                status: 400,
                reason: "missing_task_id",
            },
            MissingTaskBehavior::TeacherFallback => RoutingDecision::Teacher {
                reason: "missing_task_teacher_fallback",
                task_id: "unknown_task".to_string(),
            },
            MissingTaskBehavior::UnknownTask => RoutingDecision::Teacher {
                reason: "missing_task_unknown_task",
                task_id: "unknown_task".to_string(),
            },
        },
    }
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
        headers.insert("X-Task-ID", HeaderValue::from_static("email_classification_v1"));
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
            decide_route(&metadata, MissingTaskBehavior::TeacherFallback),
            RoutingDecision::Teacher {
                reason: "teacher_only_v1",
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
            decide_route(&metadata, MissingTaskBehavior::TeacherFallback),
            RoutingDecision::Reject {
                status: 400,
                reason: "missing_client_id"
            }
        );
    }
}
