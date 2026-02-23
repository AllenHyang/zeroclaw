//! Global goal-loop status singleton.
//!
//! The daemon writes status at key transition points; the API reads
//! a snapshot via `snapshot_json()`.  Uses the same `OnceLock<Mutex<_>>`
//! pattern as `crate::health`.

use chrono::Utc;
use parking_lot::Mutex;
use serde::Serialize;
use std::sync::OnceLock;
use std::time::Instant;

// ── Public snapshot (returned by API) ──────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct GoalLoopSnapshot {
    pub mode: String,
    pub next_tick_secs: u64,
    pub loop_interval_idle_secs: u64,
    pub loop_interval_active_secs: u64,
    pub exploration_today: u32,
    pub exploration_today_max: u32,
    pub exploration_cooldown_remaining_secs: u64,
    pub exploration_available: bool,
    pub intent_scan_watermark: Option<String>,
    pub cycle_count: u64,
    pub updated_at: String,
}

// ── Inner mutable state ────────────────────────────────────────

struct Inner {
    mode: String,
    next_tick_at: Option<Instant>,
    loop_interval_idle_secs: u64,
    loop_interval_active_secs: u64,
    exploration_today: u32,
    exploration_today_max: u32,
    last_exploration_at: Option<Instant>,
    explore_cooldown_secs: u64,
    intent_scan_watermark: Option<String>,
    cycle_count: u64,
}

static REGISTRY: OnceLock<Mutex<Inner>> = OnceLock::new();

fn registry() -> &'static Mutex<Inner> {
    REGISTRY.get_or_init(|| {
        Mutex::new(Inner {
            mode: "starting".into(),
            next_tick_at: None,
            loop_interval_idle_secs: 0,
            loop_interval_active_secs: 5,
            exploration_today: 0,
            exploration_today_max: 0,
            last_exploration_at: None,
            explore_cooldown_secs: 0,
            intent_scan_watermark: None,
            cycle_count: 0,
        })
    })
}

// ── Writers (called from daemon) ───────────────────────────────

pub fn set_mode(mode: &str) {
    registry().lock().mode = mode.into();
}

pub fn set_intervals(idle_secs: u64, active_secs: u64) {
    let mut inner = registry().lock();
    inner.loop_interval_idle_secs = idle_secs;
    inner.loop_interval_active_secs = active_secs;
}

pub fn set_exploration_config(today: u32, max: u32, cooldown_secs: u64) {
    let mut inner = registry().lock();
    inner.exploration_today = today;
    inner.exploration_today_max = max;
    inner.explore_cooldown_secs = cooldown_secs;
}

pub fn mark_exploration_run() {
    let mut inner = registry().lock();
    inner.exploration_today += 1;
    inner.last_exploration_at = Some(Instant::now());
}

pub fn set_intent_scan_watermark(ts: &str) {
    registry().lock().intent_scan_watermark = Some(ts.into());
}

pub fn increment_cycle() {
    registry().lock().cycle_count += 1;
}

pub fn set_next_tick(delay: std::time::Duration) {
    registry().lock().next_tick_at = Some(Instant::now() + delay);
}

// ── Reader (called from API / state writer) ────────────────────

pub fn snapshot() -> GoalLoopSnapshot {
    let inner = registry().lock();
    let now = Instant::now();

    let next_tick_secs = inner
        .next_tick_at
        .map(|t| t.saturating_duration_since(now).as_secs())
        .unwrap_or(0);

    let cooldown_remaining_secs = inner
        .last_exploration_at
        .map(|last| {
            let elapsed = last.elapsed().as_secs();
            inner.explore_cooldown_secs.saturating_sub(elapsed)
        })
        .unwrap_or(0);

    let exploration_available =
        inner.exploration_today < inner.exploration_today_max && cooldown_remaining_secs == 0;

    GoalLoopSnapshot {
        mode: inner.mode.clone(),
        next_tick_secs,
        loop_interval_idle_secs: inner.loop_interval_idle_secs,
        loop_interval_active_secs: inner.loop_interval_active_secs,
        exploration_today: inner.exploration_today,
        exploration_today_max: inner.exploration_today_max,
        exploration_cooldown_remaining_secs: cooldown_remaining_secs,
        exploration_available,
        intent_scan_watermark: inner.intent_scan_watermark.clone(),
        cycle_count: inner.cycle_count,
        updated_at: Utc::now().to_rfc3339(),
    }
}

pub fn snapshot_json() -> serde_json::Value {
    serde_json::to_value(snapshot()).unwrap_or_else(|_| {
        serde_json::json!({
            "status": "error",
            "message": "failed to serialize goal loop status"
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_returns_valid_initial_state() {
        let snap = snapshot();
        // Mode starts as "starting" on first access
        assert!(!snap.mode.is_empty());
        assert!(snap.updated_at.contains('T'));
    }

    #[test]
    fn set_mode_updates_snapshot() {
        set_mode("idle");
        let snap = snapshot();
        assert_eq!(snap.mode, "idle");
        // Reset for other tests
        set_mode("starting");
    }

    #[test]
    fn increment_cycle_increases_count() {
        let before = snapshot().cycle_count;
        increment_cycle();
        let after = snapshot().cycle_count;
        assert_eq!(after, before + 1);
    }

    #[test]
    fn set_intervals_updates_snapshot() {
        set_intervals(300, 5);
        let snap = snapshot();
        assert_eq!(snap.loop_interval_idle_secs, 300);
        assert_eq!(snap.loop_interval_active_secs, 5);
    }

    #[test]
    fn exploration_config_and_availability() {
        set_exploration_config(0, 6, 3600);
        let snap = snapshot();
        assert_eq!(snap.exploration_today, 0);
        assert_eq!(snap.exploration_today_max, 6);
        assert!(snap.exploration_available);
    }

    #[test]
    fn snapshot_json_serializes_all_fields() {
        let json = snapshot_json();
        assert!(json.get("mode").is_some());
        assert!(json.get("next_tick_secs").is_some());
        assert!(json.get("cycle_count").is_some());
        assert!(json.get("exploration_available").is_some());
        assert!(json.get("updated_at").is_some());
    }
}
