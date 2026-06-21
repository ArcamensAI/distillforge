use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Default)]
pub struct ProxyMetrics {
    total_requests: AtomicU64,
    teacher_requests: AtomicU64,
    rejected_requests: AtomicU64,
    upstream_errors: AtomicU64,
}

impl ProxyMetrics {
    pub fn inc_total(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_teacher(&self) {
        self.teacher_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_rejected(&self) {
        self.rejected_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_upstream_error(&self) {
        self.upstream_errors.fetch_add(1, Ordering::Relaxed);
    }

    pub fn render_prometheus(&self) -> String {
        format!(
            concat!(
                "# TYPE distillforge_requests_total counter\n",
                "distillforge_requests_total {}\n",
                "# TYPE distillforge_teacher_requests_total counter\n",
                "distillforge_teacher_requests_total {}\n",
                "# TYPE distillforge_rejected_requests_total counter\n",
                "distillforge_rejected_requests_total {}\n",
                "# TYPE distillforge_upstream_errors_total counter\n",
                "distillforge_upstream_errors_total {}\n"
            ),
            self.total_requests.load(Ordering::Relaxed),
            self.teacher_requests.load(Ordering::Relaxed),
            self.rejected_requests.load(Ordering::Relaxed),
            self.upstream_errors.load(Ordering::Relaxed)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_prometheus_counters() {
        let metrics = ProxyMetrics::default();
        metrics.inc_total();
        metrics.inc_teacher();

        let rendered = metrics.render_prometheus();

        assert!(rendered.contains("distillforge_requests_total 1"));
        assert!(rendered.contains("distillforge_teacher_requests_total 1"));
    }
}
