use crate::config::RateLimitsConfig;
use crate::routing::RequestMetadata;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct RateLimiter {
    config: RateLimitsConfig,
    window: Duration,
    counters: Mutex<HashMap<String, WindowCounter>>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RateLimitDecision {
    pub allowed: bool,
    pub reason: Option<&'static str>,
    pub limit: Option<u64>,
}

#[derive(Debug)]
struct WindowCounter {
    started_at: Instant,
    count: u64,
}

impl RateLimiter {
    pub fn new(config: RateLimitsConfig) -> Self {
        Self {
            window: Duration::from_millis(config.window_ms.max(1)),
            config,
            counters: Mutex::new(HashMap::new()),
        }
    }

    pub fn check(&self, metadata: &RequestMetadata) -> RateLimitDecision {
        if !self.config.enabled {
            return RateLimitDecision::allowed();
        }

        let Some(client_id) = metadata.client_id.as_deref() else {
            return RateLimitDecision::allowed();
        };

        let now = Instant::now();
        let checks = self.checks_for(metadata, client_id);
        if checks.is_empty() {
            return RateLimitDecision::allowed();
        }

        let Ok(mut counters) = self.counters.lock() else {
            return RateLimitDecision::allowed();
        };

        for check in &checks {
            let count = current_count(&mut counters, &check.key, now, self.window);
            if count >= check.limit {
                return RateLimitDecision {
                    allowed: false,
                    reason: Some(check.reason),
                    limit: Some(check.limit),
                };
            }
        }

        for check in &checks {
            let counter = counters.get_mut(&check.key).expect("counter initialized");
            counter.count += 1;
        }

        RateLimitDecision::allowed()
    }

    fn checks_for(&self, metadata: &RequestMetadata, client_id: &str) -> Vec<LimitCheck> {
        let mut checks = Vec::new();
        let client_limit = self
            .config
            .clients
            .get(client_id)
            .copied()
            .or(self.config.default_requests_per_window);
        if let Some(limit) = client_limit {
            checks.push(LimitCheck {
                key: format!("client:{client_id}"),
                limit,
                reason: "rate_limited_client",
            });
        }

        if let Some(task_id) = metadata.task_id.as_deref() {
            if let Some(limit) = self.config.tasks.get(task_id).copied() {
                checks.push(LimitCheck {
                    key: format!("task:{task_id}"),
                    limit,
                    reason: "rate_limited_task",
                });
            }
        }

        checks
    }
}

impl RateLimitDecision {
    fn allowed() -> Self {
        Self {
            allowed: true,
            reason: None,
            limit: None,
        }
    }
}

struct LimitCheck {
    key: String,
    limit: u64,
    reason: &'static str,
}

fn current_count(
    counters: &mut HashMap<String, WindowCounter>,
    key: &str,
    now: Instant,
    window: Duration,
) -> u64 {
    let counter = counters
        .entry(key.to_string())
        .or_insert_with(|| WindowCounter {
            started_at: now,
            count: 0,
        });
    if now.duration_since(counter.started_at) >= window {
        counter.started_at = now;
        counter.count = 0;
    }
    counter.count
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn metadata(client_id: Option<&str>, task_id: Option<&str>) -> RequestMetadata {
        RequestMetadata {
            request_id: "req_1".to_string(),
            client_id: client_id.map(ToOwned::to_owned),
            task_id: task_id.map(ToOwned::to_owned),
            cost_center: None,
            quality_mode: None,
            pii_level: None,
            no_train: false,
        }
    }

    #[test]
    fn disabled_limiter_allows_requests() {
        let limiter = RateLimiter::new(RateLimitsConfig::default());

        assert!(limiter.check(&metadata(Some("crm"), Some("task"))).allowed);
    }

    #[test]
    fn applies_default_client_limit() {
        let limiter = RateLimiter::new(RateLimitsConfig {
            enabled: true,
            window_ms: 60_000,
            default_requests_per_window: Some(1),
            clients: BTreeMap::new(),
            tasks: BTreeMap::new(),
        });

        assert!(limiter.check(&metadata(Some("crm"), Some("task"))).allowed);
        let decision = limiter.check(&metadata(Some("crm"), Some("task")));

        assert!(!decision.allowed);
        assert_eq!(decision.reason, Some("rate_limited_client"));
        assert_eq!(decision.limit, Some(1));
    }

    #[test]
    fn applies_task_limit() {
        let mut tasks = BTreeMap::new();
        tasks.insert("email".to_string(), 1);
        let limiter = RateLimiter::new(RateLimitsConfig {
            enabled: true,
            window_ms: 60_000,
            default_requests_per_window: None,
            clients: BTreeMap::new(),
            tasks,
        });

        assert!(limiter.check(&metadata(Some("crm"), Some("email"))).allowed);
        let decision = limiter.check(&metadata(Some("other"), Some("email")));

        assert!(!decision.allowed);
        assert_eq!(decision.reason, Some("rate_limited_task"));
    }

    #[test]
    fn ignores_missing_client_until_header_validation() {
        let mut tasks = BTreeMap::new();
        tasks.insert("email".to_string(), 0);
        let limiter = RateLimiter::new(RateLimitsConfig {
            enabled: true,
            window_ms: 60_000,
            default_requests_per_window: Some(0),
            clients: BTreeMap::new(),
            tasks,
        });

        assert!(limiter.check(&metadata(None, Some("email"))).allowed);
    }
}
