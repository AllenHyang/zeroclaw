use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Maximum retry attempts per step before marking the goal as blocked.
const MAX_STEP_ATTEMPTS: u32 = 3;

/// Maximum characters retained in `working_memory` between autonomous sessions.
const MAX_WORKING_MEMORY_CHARS: usize = 2000;

/// Default ceiling for `total_iterations` across all autonomous sessions for one goal.
const DEFAULT_MAX_TOTAL_GOAL_ITERATIONS: u32 = 200;

// ── Data Structures ─────────────────────────────────────────────

/// Root state persisted to `{workspace}/state/goals.json`.
/// Format matches the `goal-tracker` skill's file layout.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GoalState {
    #[serde(default)]
    pub goals: Vec<Goal>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    #[serde(deserialize_with = "deserialize_lenient_string")]
    pub id: String,
    #[serde(deserialize_with = "deserialize_lenient_string")]
    pub description: String,
    #[serde(default)]
    pub status: GoalStatus,
    #[serde(default)]
    pub priority: GoalPriority,
    #[serde(default, deserialize_with = "deserialize_lenient_string")]
    pub created_at: String,
    #[serde(default, deserialize_with = "deserialize_lenient_string")]
    pub updated_at: String,
    #[serde(default)]
    pub steps: Vec<Step>,
    /// Accumulated context from previous step results.
    #[serde(default, deserialize_with = "deserialize_lenient_string")]
    pub context: String,
    /// Last error encountered during step execution.
    #[serde(default, deserialize_with = "deserialize_lenient_opt_string")]
    pub last_error: Option<String>,

    // ── Autonomous session fields ───────────────────────────────
    /// Success criteria for autonomous mode (what "done" looks like).
    #[serde(default, deserialize_with = "deserialize_lenient_opt_string")]
    pub success_criteria: Option<String>,
    /// Constraints the agent must respect during autonomous execution.
    #[serde(default, deserialize_with = "deserialize_lenient_opt_string")]
    pub constraints: Option<String>,
    /// Persisted working memory from the last autonomous session.
    #[serde(default, deserialize_with = "deserialize_lenient_opt_string")]
    pub working_memory: Option<String>,
    /// Execution mode: `autonomous` (default) or `stepped`.
    #[serde(default)]
    pub execution_mode: GoalExecutionMode,
    /// Total tool iterations consumed across all autonomous sessions.
    #[serde(default)]
    pub total_iterations: u32,

    // ── Step-0 confirmation fields ──────────────────────────────
    /// Agent-generated understanding + execution plan (set during AwaitingConfirmation).
    #[serde(default, deserialize_with = "deserialize_lenient_opt_string")]
    pub confirmation_plan: Option<String>,
    /// Timestamp (RFC 3339) when the confirmation was requested.
    #[serde(default, deserialize_with = "deserialize_lenient_opt_string")]
    pub confirmation_requested_at: Option<String>,
    /// User feedback when rejecting a confirmation (triggers re-generation).
    #[serde(default, deserialize_with = "deserialize_lenient_opt_string")]
    pub confirmation_feedback: Option<String>,

    // ── Notification delivery tracking ──────────────────────────
    /// Whether the last status-change notification was delivered.
    #[serde(default)]
    pub last_notification_delivered: bool,
    /// Timestamp (RFC 3339) of the last successful delivery.
    #[serde(default, deserialize_with = "deserialize_lenient_opt_string")]
    pub last_notification_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum GoalStatus {
    #[default]
    Pending,
    AwaitingConfirmation,
    InProgress,
    Completed,
    Blocked,
    Cancelled,
}

impl<'de> Deserialize<'de> for GoalStatus {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Ok(match s.as_str() {
            "awaiting_confirmation" => Self::AwaitingConfirmation,
            "in_progress" => Self::InProgress,
            "completed" => Self::Completed,
            "blocked" => Self::Blocked,
            "cancelled" => Self::Cancelled,
            _ => Self::Pending,
        })
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum GoalPriority {
    Low = 0,
    #[default]
    Medium = 1,
    High = 2,
    Critical = 3,
}

impl PartialOrd for GoalPriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GoalPriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (*self as u8).cmp(&(*other as u8))
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum GoalExecutionMode {
    #[default]
    Autonomous,
    Stepped,
}

// Self-healing deserialization: unknown values → Autonomous
impl<'de> Deserialize<'de> for GoalExecutionMode {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Ok(match s.as_str() {
            "stepped" => Self::Stepped,
            _ => Self::Autonomous,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AutonomousSessionStatus {
    Completed,
    InProgress,
    Blocked(String),
}

/// Coerce any JSON value into a `String`.
///
/// LLM-generated goals.json may contain unexpected types for string fields:
/// integers (`1`), booleans (`true`), nulls, or arrays (`["a","b"]`).
/// This visitor converts all of them to a string representation so that
/// deserialization never fails on type mismatch.
fn deserialize_lenient_string<'de, D: serde::Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    struct LenientString;
    impl<'de> serde::de::Visitor<'de> for LenientString {
        type Value = String;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("any JSON value coercible to string")
        }
        fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<String, E> {
            Ok(v.to_owned())
        }
        fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<String, E> {
            Ok(v.to_string())
        }
        fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<String, E> {
            Ok(v.to_string())
        }
        fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<String, E> {
            Ok(v.to_string())
        }
        fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<String, E> {
            Ok(v.to_string())
        }
        fn visit_unit<E: serde::de::Error>(self) -> Result<String, E> {
            Ok(String::new())
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<String, A::Error> {
            let mut parts = Vec::new();
            while let Some(val) = seq.next_element::<serde_json::Value>()? {
                match val {
                    serde_json::Value::String(s) => parts.push(s),
                    other => parts.push(other.to_string()),
                }
            }
            Ok(parts.join(", "))
        }
    }
    d.deserialize_any(LenientString)
}

/// Coerce any JSON value (including null) into `Option<String>`.
fn deserialize_lenient_opt_string<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<String>, D::Error> {
    let s = deserialize_lenient_string(d)?;
    Ok(if s.is_empty() { None } else { Some(s) })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    #[serde(deserialize_with = "deserialize_lenient_string")]
    pub id: String,
    #[serde(deserialize_with = "deserialize_lenient_string")]
    pub description: String,
    #[serde(default)]
    pub status: StepStatus,
    #[serde(default, deserialize_with = "deserialize_lenient_opt_string")]
    pub result: Option<String>,
    #[serde(default)]
    pub attempts: u32,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum StepStatus {
    #[default]
    Pending,
    InProgress,
    Completed,
    Failed,
    Blocked,
}

impl<'de> Deserialize<'de> for StepStatus {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Ok(match s.as_str() {
            "in_progress" => Self::InProgress,
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "blocked" => Self::Blocked,
            _ => Self::Pending,
        })
    }
}

/// Truncate a string to at most `max` **characters** on a char boundary.
fn truncate_str(s: &str, max: usize) -> &str {
    // Use char_indices to find the byte offset of the max-th character.
    // If the string has <= max chars, return it unchanged.
    match s.char_indices().nth(max) {
        Some((byte_idx, _)) => &s[..byte_idx],
        None => s,
    }
}

// ── GoalEngine ──────────────────────────────────────────────────

pub struct GoalEngine {
    state_path: PathBuf,
}

impl GoalEngine {
    pub fn new(workspace_dir: &Path) -> Self {
        Self {
            state_path: workspace_dir.join("state").join("goals.json"),
        }
    }

    /// Load goal state from disk. Returns empty state if file doesn't exist.
    pub async fn load_state(&self) -> Result<GoalState> {
        if !self.state_path.exists() {
            return Ok(GoalState::default());
        }
        let bytes = tokio::fs::read(&self.state_path).await?;
        if bytes.is_empty() {
            return Ok(GoalState::default());
        }
        let state: GoalState = serde_json::from_slice(&bytes)?;
        Ok(state)
    }

    /// Atomic save: write to .tmp then rename.
    pub async fn save_state(&self, state: &GoalState) -> Result<()> {
        if let Some(parent) = self.state_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let tmp = self.state_path.with_extension("json.tmp");
        let data = serde_json::to_vec_pretty(state)?;
        tokio::fs::write(&tmp, data).await?;
        tokio::fs::rename(&tmp, &self.state_path).await?;
        Ok(())
    }

    /// Load, normalize, and re-save goal state.
    ///
    /// Handles schema drift caused by LLM writing goals.json directly via
    /// `file_write` (bypassing `save_state`). Normalizations applied:
    /// 1. Fill empty `id` with a new UUID.
    /// 2. Fill empty `created_at` / `updated_at` with current timestamp.
    /// 3. Deduplicate by `id` (keep last occurrence).
    /// 4. Re-save through `save_state` (drops unknown fields, fills serde
    ///    defaults, normalizes enum strings).
    pub async fn load_and_normalize(&self) -> Result<GoalState> {
        let mut state = self.load_state().await?;
        let now = chrono::Utc::now().to_rfc3339();
        let mut changed = false;

        for goal in &mut state.goals {
            if goal.id.is_empty() {
                goal.id = uuid::Uuid::new_v4().to_string();
                changed = true;
            }
            if goal.created_at.is_empty() {
                goal.created_at = now.clone();
                changed = true;
            }
            if goal.updated_at.is_empty() {
                goal.updated_at = now.clone();
                changed = true;
            }
        }

        // Deduplicate by id — keep last occurrence
        let mut seen = HashSet::new();
        let original_len = state.goals.len();
        // Iterate from the back so the *last* occurrence survives
        state.goals = state
            .goals
            .into_iter()
            .rev()
            .filter(|g| seen.insert(g.id.clone()))
            .collect::<Vec<_>>();
        state.goals.reverse(); // restore original order
        if state.goals.len() != original_len {
            changed = true;
        }

        // ── Hallucination repair ────────────────────────────────────
        // LLM agents can write inconsistent data to goals.json.
        // Fix contradictions so the goal loop operates on clean state.
        for goal in &mut state.goals {
            // A goal that was never executed cannot be blocked.
            if goal.status == GoalStatus::Blocked && goal.total_iterations == 0 {
                let has_real_attempts = goal.steps.iter().any(|s| s.attempts > 0);
                if !has_real_attempts {
                    tracing::info!(
                        goal_id = %goal.id,
                        "Normalize: resetting hallucinated blocked → pending (0 iterations, 0 attempts)"
                    );
                    goal.status = GoalStatus::Pending;
                    goal.last_error = None;
                    changed = true;
                }
            }

            // Steps with 0 attempts cannot have results — clear fabricated ones.
            // Skip completed/cancelled goals: their step results are historical records.
            if goal.status != GoalStatus::Completed && goal.status != GoalStatus::Cancelled {
                for step in &mut goal.steps {
                    if step.attempts == 0 && step.result.is_some() {
                        tracing::info!(
                            goal_id = %goal.id,
                            step_id = %step.id,
                            "Normalize: clearing hallucinated step result (0 attempts)"
                        );
                        step.result = None;
                        changed = true;
                    }
                }
            }
        }

        if changed {
            self.save_state(&state).await?;
        }

        Ok(state)
    }

    /// Select the next actionable (goal_index, step_index) pair.
    ///
    /// Strategy: highest-priority in-progress **Stepped** goal, first pending step
    /// that hasn't exceeded `MAX_STEP_ATTEMPTS`. Autonomous goals are skipped —
    /// they are handled by `select_next_autonomous_goal`.
    pub fn select_next_actionable(state: &GoalState) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize, GoalPriority)> = None;

        for (gi, goal) in state.goals.iter().enumerate() {
            if goal.status != GoalStatus::InProgress {
                continue;
            }
            // Skip autonomous goals — they use a different execution path
            if goal.execution_mode == GoalExecutionMode::Autonomous {
                continue;
            }
            if let Some(si) = goal
                .steps
                .iter()
                .position(|s| s.status == StepStatus::Pending && s.attempts < MAX_STEP_ATTEMPTS)
            {
                match best {
                    Some((_, _, ref bp)) if goal.priority <= *bp => {}
                    _ => best = Some((gi, si, goal.priority)),
                }
            }
        }

        best.map(|(gi, si, _)| (gi, si))
    }

    /// Build a focused prompt for the agent to execute one step.
    pub fn build_step_prompt(goal: &Goal, step: &Step) -> String {
        let mut prompt = String::new();

        let _ = writeln!(
            prompt,
            "[Goal Loop] Executing step for goal: {}\n",
            goal.description
        );

        // Completed steps summary
        let completed: Vec<&Step> = goal
            .steps
            .iter()
            .filter(|s| s.status == StepStatus::Completed)
            .collect();
        if !completed.is_empty() {
            prompt.push_str("Completed steps:\n");
            for s in &completed {
                let _ = writeln!(
                    prompt,
                    "- [done] {}: {}",
                    s.description,
                    s.result.as_deref().unwrap_or("(no result)")
                );
            }
            prompt.push('\n');
        }

        // Accumulated context
        if !goal.context.is_empty() {
            let _ = write!(prompt, "Context so far:\n{}\n\n", goal.context);
        }

        // Current step
        let _ = write!(
            prompt,
            "Current step: {}\n\
             Please execute this step. Provide a clear summary of what you did and the outcome.\n",
            step.description
        );

        // Retry warning
        if step.attempts > 0 {
            let _ = write!(
                prompt,
                "\nWARNING: This step has failed {} time(s) before. \
                 Last error: {}\n\
                 Try a different approach.\n",
                step.attempts,
                goal.last_error.as_deref().unwrap_or("unknown")
            );
        }

        prompt
    }

    /// Simple heuristic: output containing error indicators → failure.
    pub fn interpret_result(output: &str) -> bool {
        let lower = output.to_ascii_lowercase();
        let failure_indicators = [
            "failed to",
            "error:",
            "unable to",
            "cannot ",
            "could not",
            "fatal:",
            "panic:",
        ];
        !failure_indicators.iter().any(|ind| lower.contains(ind))
    }

    pub fn max_step_attempts() -> u32 {
        MAX_STEP_ATTEMPTS
    }

    pub fn default_max_total_goal_iterations() -> u32 {
        DEFAULT_MAX_TOTAL_GOAL_ITERATIONS
    }

    // ── Autonomous session support ─────────────────────────────

    /// Select the highest-priority InProgress goal with `execution_mode == Autonomous`.
    pub fn select_next_autonomous_goal(state: &GoalState) -> Option<usize> {
        let mut best: Option<(usize, GoalPriority)> = None;

        for (gi, goal) in state.goals.iter().enumerate() {
            if goal.status != GoalStatus::InProgress {
                continue;
            }
            if goal.execution_mode != GoalExecutionMode::Autonomous {
                continue;
            }
            match best {
                Some((_, ref bp)) if goal.priority <= *bp => {}
                _ => best = Some((gi, goal.priority)),
            }
        }

        best.map(|(gi, _)| gi)
    }

    /// Build a prompt for autonomous goal execution.
    ///
    /// The prompt gives the agent the goal's intent, success criteria, constraints,
    /// working memory from the prior session, completed context, and steps as
    /// optional reference. The agent must end with a `[GOAL_STATUS: ...]` tag.
    pub fn build_autonomous_prompt(goal: &Goal, state: &GoalState) -> String {
        let mut prompt = String::new();

        let _ = writeln!(
            prompt,
            "[Autonomous Goal Session] Goal: {}\n",
            goal.description
        );

        // Success criteria
        if let Some(ref criteria) = goal.success_criteria {
            let _ = writeln!(prompt, "== Success Criteria ==\n{criteria}\n");
        }

        // Constraints
        if let Some(ref constraints) = goal.constraints {
            let _ = writeln!(prompt, "== Constraints ==\n{constraints}\n");
        }

        // Working memory from last session
        if let Some(ref wm) = goal.working_memory {
            if !wm.is_empty() {
                let _ = writeln!(prompt, "== Working Memory (from last session) ==\n{wm}\n");
            }
        }

        // Accumulated context from completed steps
        if !goal.context.is_empty() {
            let _ = writeln!(prompt, "== Completed Context ==\n{}\n", goal.context);
        }

        // Steps as optional reference
        if !goal.steps.is_empty() {
            prompt.push_str("== Steps (reference only — you may deviate) ==\n");
            for s in &goal.steps {
                let tag = match s.status {
                    StepStatus::Completed => "done",
                    StepStatus::Failed | StepStatus::Blocked => "blocked",
                    StepStatus::InProgress => "active",
                    _ => "pending",
                };
                let result = s.result.as_deref().unwrap_or("");
                if result.is_empty() {
                    let _ = writeln!(prompt, "- [{tag}] {}", s.description);
                } else {
                    let _ = writeln!(prompt, "- [{tag}] {}: {result}", s.description);
                }
            }
            prompt.push('\n');
        }

        // Last error
        if let Some(ref err) = goal.last_error {
            let _ = writeln!(prompt, "== Last Error ==\n{err}\n");
        }

        // Goal panorama (other goals for context)
        Self::append_goals_by_status(&mut prompt, state);

        // Session instructions
        prompt.push_str(
            "== Session Instructions ==\n\
             You are running autonomously — do NOT ask the user for input.\n\n\
             First, read SOUL.md to ground yourself in your identity and operating \
             principles. Your behavior should follow what SOUL.md says.\n\n\
             The goal description above is your task, like a one-liner from your boss.\n\
             Figure out the intent yourself. Break it down, execute, and verify.\n\
             Don't wait for detailed specs — think like a capable employee who \
             receives a brief instruction and delivers a complete result.\n\n\
             If success criteria or constraints are provided, respect them.\n\
             If not, use your own judgment on what \"done\" looks like.\n\n\
             When you complete a goal and there is a clear next step, write a \
             follow-up goal directly into state/goals.json (status: pending, \
             priority: low). Don't leave loose ends for exploration to find later.\n\n\
             When you finish or need to pause, end your output with EXACTLY one of:\n\
             [GOAL_STATUS: completed]\n\
             [GOAL_STATUS: in_progress]\n\
             [GOAL_STATUS: blocked REASON]\n\n\
             After the status tag, optionally write a brief working memory note \
             (max ~500 chars) for your next session. This is your ONLY way to \
             pass context to yourself across sessions.\n\n\
             Config hot-reload: most config changes (allowlists, limits, feature flags, \
             goal_loop params, temperature) take effect automatically — the goal loop \
             re-reads config.toml every cycle. No restart needed.\n\n\
             Graceful restart: if you changed channel/gateway/provider config and need a \
             restart, write a marker file: `echo $(date) > ~/.zeroclaw/restart_requested`. \
             The daemon detects it within 3 seconds, exits cleanly, and launchd restarts it. \
             Your session completes normally. Do NOT use `launchctl unload` — that kills \
             your own session mid-flight.\n",
        );

        prompt
    }

    /// Parse the agent output to determine autonomous session status.
    ///
    /// Priority: (1) explicit `[GOAL_STATUS: ...]` tag in last 500 chars,
    /// (2) fallback to existing `interpret_result()` heuristic (conservatively
    /// returns InProgress rather than Completed).
    pub fn interpret_autonomous_result(output: &str) -> AutonomousSessionStatus {
        // Search in the tail of the output for the status tag (char-boundary-safe)
        let search_region = if output.len() > 500 {
            let mut start = output.len() - 500;
            while !output.is_char_boundary(start) && start < output.len() {
                start += 1;
            }
            &output[start..]
        } else {
            output
        };

        // Look for [GOAL_STATUS: ...] — last occurrence wins
        if let Some(pos) = search_region.rfind("[GOAL_STATUS:") {
            let after = &search_region[pos + "[GOAL_STATUS:".len()..];
            if let Some(end) = after.find(']') {
                let tag_content = after[..end].trim().to_ascii_lowercase();
                if tag_content == "completed" {
                    return AutonomousSessionStatus::Completed;
                } else if tag_content == "in_progress" {
                    return AutonomousSessionStatus::InProgress;
                } else if let Some(reason) = tag_content.strip_prefix("blocked") {
                    let reason = reason.trim().to_string();
                    return AutonomousSessionStatus::Blocked(if reason.is_empty() {
                        "no reason given".into()
                    } else {
                        reason
                    });
                }
            }
        }

        // Fallback: use existing heuristic but conservatively return InProgress
        // (never auto-complete from heuristic alone)
        if !Self::interpret_result(output) {
            AutonomousSessionStatus::Blocked("heuristic detected failure indicators".into())
        } else {
            AutonomousSessionStatus::InProgress
        }
    }

    /// Extract working memory from the agent output.
    ///
    /// If a `[GOAL_STATUS: ...]` tag exists, take text after the closing `]`.
    /// Otherwise, truncate the entire output.
    pub fn extract_working_memory(output: &str, max_chars: usize) -> String {
        // Find the last [GOAL_STATUS: ...] tag
        if let Some(pos) = output.rfind("[GOAL_STATUS:") {
            if let Some(end_bracket) = output[pos..].find(']') {
                let after = &output[pos + end_bracket + 1..];
                let trimmed = after.trim();
                if !trimmed.is_empty() {
                    return truncate_str(trimmed, max_chars).to_string();
                }
            }
        }

        // No tag or nothing after tag: truncate the whole output
        truncate_str(output.trim(), max_chars).to_string()
    }

    /// Build a prompt to scan recent chat messages for un-captured task intents.
    ///
    /// The LLM compares recent user messages against existing goals and creates
    /// new goals for any task intents that have no corresponding goal.
    pub fn build_intent_scan_prompt(
        recent_messages: &[crate::memory::MemoryEntry],
        state: &GoalState,
    ) -> String {
        let mut prompt = String::new();

        prompt.push_str(
            "[Intent Scan] Scanning recent chat messages for tasks that lack a corresponding goal.\n\n",
        );

        // Recent user messages
        prompt.push_str("== Recent User Messages ==\n");
        if recent_messages.is_empty() {
            prompt.push_str("(none)\n");
        } else {
            for msg in recent_messages {
                let _ = writeln!(prompt, "- [{}] {}", msg.timestamp, msg.content);
            }
        }
        prompt.push('\n');

        // Current goals
        prompt.push_str("== Current Goals ==\n");
        if state.goals.is_empty() {
            prompt.push_str("(none)\n");
        } else {
            for g in &state.goals {
                let status = match g.status {
                    GoalStatus::Pending => "pending",
                    GoalStatus::AwaitingConfirmation => "awaiting_confirmation",
                    GoalStatus::InProgress => "in_progress",
                    GoalStatus::Completed => "completed",
                    GoalStatus::Blocked => "blocked",
                    GoalStatus::Cancelled => "cancelled",
                };
                let _ = writeln!(prompt, "- [{}] ({}) {}", g.id, status, g.description);
            }
        }
        prompt.push('\n');

        // Instructions
        prompt.push_str(
            "== Instructions ==\n\
             Compare the user messages above against the current goals.\n\
             Identify messages where the user expressed a TASK INTENT — something they \
             want done — that does NOT have a corresponding goal (any status).\n\n\
             Rules:\n\
             - Casual chat, greetings, questions, acknowledgements are NOT task intents.\n\
             - Only explicit requests to DO something count (e.g., \"help me write X\", \
             \"deploy Y\", \"fix Z\", \"research A\").\n\
             - If a message's intent is already covered by an existing goal (even if \
             worded differently), do NOT create a duplicate.\n\
             - If you find uncaptured task intents, write them to state/goals.json with:\n\
             \x20  status: \"in_progress\", execution_mode: \"autonomous\", \
             priority: your judgment (low/medium/high).\n\
             \x20  Include a clear description and 2-4 concrete steps.\n\
             - If all intents are already captured, or there are no task intents, \
             do nothing and say so briefly.\n\n\
             Constraints:\n\
             - Max 3 tool calls.\n\
             - Do NOT ask the user for clarification — infer intent from context.\n",
        );

        prompt
    }

    /// Build a lightweight prompt for step-0 understanding confirmation.
    ///
    /// The agent outputs its understanding of the goal + a 3-5 step execution plan.
    /// If the user previously rejected and provided feedback, that feedback is
    /// included so the agent can correct its understanding.
    pub fn build_understanding_prompt(goal: &Goal, state: &GoalState) -> String {
        let mut prompt = String::new();

        let _ = writeln!(
            prompt,
            "[Understanding Confirmation] Goal: {}\n",
            goal.description
        );

        // Success criteria
        if let Some(ref criteria) = goal.success_criteria {
            let _ = writeln!(prompt, "== Success Criteria ==\n{criteria}\n");
        }

        // Constraints
        if let Some(ref constraints) = goal.constraints {
            let _ = writeln!(prompt, "== Constraints ==\n{constraints}\n");
        }

        // User feedback from previous rejection
        if let Some(ref feedback) = goal.confirmation_feedback {
            let _ = writeln!(
                prompt,
                "== User Feedback (your previous understanding was REJECTED) ==\n\
                 The user rejected your previous plan with this feedback:\n\
                 \"{feedback}\"\n\n\
                 You MUST incorporate this feedback into your revised understanding.\n"
            );
        }

        // Goal panorama
        Self::append_goals_by_status(&mut prompt, state);

        // Instructions
        prompt.push_str(
            "== Instructions ==\n\
             Before executing this goal, you must first confirm your understanding.\n\
             Output a structured plan in the following format:\n\n\
             **My Understanding:**\n\
             (1-2 sentences explaining what you think the user wants)\n\n\
             **Execution Plan:**\n\
             1. (step 1)\n\
             2. (step 2)\n\
             3. (step 3)\n\
             (3-5 concrete steps)\n\n\
             **Expected Output:**\n\
             (what the user will get when this goal is done)\n\n\
             **Assumptions & Risks:**\n\
             (anything you're unsure about or potential blockers)\n\n\
             == Constraints ==\n\
             - Max 3 tool calls (for reading context only).\n\
             - Do NOT start executing the goal.\n\
             - Do NOT modify any files or state.\n\
             - Focus on producing a clear, accurate understanding.\n",
        );

        prompt
    }

    /// Build an exploration prompt for when no goals are active.
    ///
    /// Output-oriented: the agent's primary task is to discover and create
    /// an actionable goal. Journal writing is a fallback, not the default.
    pub fn build_exploration_prompt(state: &GoalState) -> String {
        let mut prompt = String::new();

        prompt.push_str(
            "[Idle Exploration] No active goals. Your task is to discover and create \
             the next actionable goal.\n\n",
        );

        Self::append_goals_by_status(&mut prompt, state);

        prompt.push_str(
            "Instructions:\n\n\
             == Phase 1: Targeted Reconnaissance (max 3 tool calls) ==\n\
             1. memory_recall query \"exploration journal\" — review past exploration entries.\n\
             2. memory_recall query \"consolidation\" — retrieve recent nightly consolidation\n\
             \x20  summaries (category: core) for errors, discoveries, unfinished threads.\n\
             3. Read SOUL.md to re-ground yourself in the user's mission and priorities.\n\n\
             == Phase 2: Deep Exploration (max 6 tool calls) ==\n\
             Choose ONE direction from the four below and explore it deeply.\n\
             Do not skim — read files, run commands, check data, verify assumptions.\n\n\
             Direction A — Completed Goal Follow-up:\n\
             \x20  Pick a recently completed goal. Check whether its outcome is still valid,\n\
             \x20  whether it opened new opportunities, or whether a natural next step exists.\n\n\
             Direction B — Implicit User Needs:\n\
             \x20  Review recent conversations, errors, or config for pain points the user\n\
             \x20  hasn't explicitly asked about but would benefit from being addressed.\n\n\
             Direction C — Capability Gaps:\n\
             \x20  Identify something the system should be able to do but currently cannot.\n\
             \x20  Check if the infrastructure exists to implement it with a concrete goal.\n\n\
             Direction D — External Changes:\n\
             \x20  Check for environmental changes (new files, updated configs, service states)\n\
             \x20  that create new work opportunities or invalidate existing assumptions.\n\n\
             == Phase 3: Output (max 3 tool calls) ==\n\
             Your DEFAULT output is to create a goal. Only skip goal creation if you\n\
             genuinely found nothing actionable after thorough exploration.\n\n\
             To create a goal:\n\
             \x20  a. VERIFY preconditions first (data exists, service reachable, not duplicate)\n\
             \x20  b. Write to state/goals.json with:\n\
             \x20     - status: \"in_progress\" (so it executes immediately)\n\
             \x20     - execution_mode: \"autonomous\" (agent drives to completion)\n\
             \x20     - priority: \"low\" (auto-approved, no human wait)\n\
             \x20     - 3-5 concrete, verifiable steps\n\
             \x20     - Clear success_criteria (must be objectively checkable)\n\
             \x20  c. Notify the user with a brief rationale\n\n\
             Goal quality rules:\n\
             \x20  - The goal MUST be completable and verifiable (not \"research X\" or \"learn about Y\")\n\
             \x20  - The goal MUST NOT require user intervention or approval to proceed\n\
             \x20  - The goal MUST NOT duplicate any in_progress or completed goal\n\
             \x20  - Pending goals are stalled and do NOT block you from creating new goals\n\
             \x20    in different directions\n\
             \x20  - Each step must have a concrete action and expected outcome\n\n\
             If you genuinely found nothing actionable:\n\
             \x20  - Write a brief journal entry using memory_store\n\
             \x20    (key: \"exploration-journal-YYYY-MM-DD-N\", category: \"exploration\")\n\
             \x20  - Explain specifically WHY nothing was actionable (this is the exception,\n\
             \x20    not the norm)\n\n\
             == Constraints ==\n\
             - Max 12 tool calls total.\n\
             - Budget: 3 reconnaissance + 6 exploration + 3 output.\n\
             - One deep investigation beats five shallow scans.\n\
             - Do NOT duplicate in_progress or completed goals.\n\
             - Pending goals do NOT count as duplicates — create new goals freely.\n\
             - Do NOT create goals that need the user to provide information.\n",
        );

        prompt
    }

    /// Append a summary of goals grouped by status (completed, blocked, pending).
    fn append_goals_by_status(buf: &mut String, state: &GoalState) {
        buf.push_str("== Completed goals ==\n");
        let completed: Vec<&Goal> = state
            .goals
            .iter()
            .filter(|g| g.status == GoalStatus::Completed)
            .collect();
        if completed.is_empty() {
            buf.push_str("(none)\n");
        } else {
            for g in &completed {
                let _ = writeln!(buf, "- {}", g.description);
            }
        }
        buf.push('\n');

        buf.push_str("== Blocked goals ==\n");
        let blocked: Vec<&Goal> = state
            .goals
            .iter()
            .filter(|g| g.status == GoalStatus::Blocked)
            .collect();
        if blocked.is_empty() {
            buf.push_str("(none)\n");
        } else {
            for g in &blocked {
                let err = g.last_error.as_deref().unwrap_or("(no error recorded)");
                let _ = writeln!(buf, "- {} [error: {}]", g.description, err);
            }
        }
        buf.push('\n');

        buf.push_str("== Awaiting confirmation (plan generated, waiting for user approval) ==\n");
        let awaiting: Vec<&Goal> = state
            .goals
            .iter()
            .filter(|g| g.status == GoalStatus::AwaitingConfirmation)
            .collect();
        if awaiting.is_empty() {
            buf.push_str("(none)\n");
        } else {
            for g in &awaiting {
                let _ = writeln!(buf, "- {} [WAITING: confirmation pending]", g.description);
            }
        }
        buf.push('\n');

        buf.push_str("== Pending goals (awaiting approval — NOT being executed) ==\n");
        let pending: Vec<&Goal> = state
            .goals
            .iter()
            .filter(|g| g.status == GoalStatus::Pending)
            .collect();
        if pending.is_empty() {
            buf.push_str("(none)\n");
        } else {
            for g in &pending {
                let _ = writeln!(
                    buf,
                    "- {} [STALLED: waiting for human approval]",
                    g.description
                );
            }
            buf.push_str(
                "NOTE: Pending goals are NOT running and may never be approved. \
                 They do NOT count as coverage — you may create goals in \
                 different directions.\n",
            );
        }
        buf.push('\n');
    }

    /// Find in-progress **Stepped** goals that have no actionable steps remaining.
    ///
    /// A goal is "stalled" when it is `InProgress` but every step is either
    /// completed, blocked, or has exhausted its retry attempts. These goals
    /// need a reflection session to decide: add new steps, mark completed,
    /// mark blocked, or escalate to the user.
    ///
    /// Autonomous goals are excluded — they manage their own lifecycle.
    pub fn find_stalled_goals(state: &GoalState) -> Vec<usize> {
        state
            .goals
            .iter()
            .enumerate()
            .filter(|(_, g)| g.status == GoalStatus::InProgress)
            .filter(|(_, g)| g.execution_mode != GoalExecutionMode::Autonomous)
            .filter(|(_, g)| {
                !g.steps.is_empty()
                    && !g
                        .steps
                        .iter()
                        .any(|s| s.status == StepStatus::Pending && s.attempts < MAX_STEP_ATTEMPTS)
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// Build a reflection prompt for a stalled goal.
    ///
    /// The agent is asked to review the goal's overall progress and decide
    /// what to do next: add new steps, mark the goal completed, or escalate.
    pub fn build_reflection_prompt(goal: &Goal) -> String {
        let mut prompt = String::new();

        let _ = writeln!(prompt, "[Goal Reflection] Goal: {}\n", goal.description);

        prompt.push_str("All steps have been attempted. Here is the current state:\n\n");

        for s in &goal.steps {
            let status_tag = match s.status {
                StepStatus::Completed => "done",
                StepStatus::Failed | StepStatus::Blocked => "blocked",
                _ if s.attempts >= MAX_STEP_ATTEMPTS => "exhausted",
                _ => "pending",
            };
            let result = s.result.as_deref().unwrap_or("(no result)");
            let _ = writeln!(prompt, "- [{status_tag}] {}: {result}", s.description);
        }

        if !goal.context.is_empty() {
            let _ = write!(prompt, "\nAccumulated context:\n{}\n", goal.context);
        }

        if let Some(ref err) = goal.last_error {
            let _ = write!(prompt, "\nLast error: {err}\n");
        }

        prompt.push_str(
            "\nReflect on this goal and take ONE of the following actions:\n\
             1. If the goal is effectively achieved, update state/goals.json to mark it `completed`.\n\
             2. If some steps failed but you can try a different approach, add NEW steps to \
                state/goals.json with fresh descriptions (don't reuse failed step IDs).\n\
             3. If the goal is truly blocked and needs human input, mark it `blocked` in \
                state/goals.json and explain what you need from the user.\n\
             4. Use memory_store to record what you learned from the failures.\n\n\
             Be decisive. Do not leave the goal in its current state.",
        );

        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Build a Stepped goal for existing step-based tests.
    fn make_goal(
        id: &str,
        desc: &str,
        status: GoalStatus,
        priority: GoalPriority,
        steps: Vec<Step>,
        context: &str,
        last_error: Option<String>,
    ) -> Goal {
        Goal {
            id: id.into(),
            description: desc.into(),
            status,
            priority,
            created_at: String::new(),
            updated_at: String::new(),
            steps,
            context: context.into(),
            last_error,
            success_criteria: None,
            constraints: None,
            working_memory: None,
            execution_mode: GoalExecutionMode::Stepped,
            total_iterations: 0,
            confirmation_plan: None,
            confirmation_requested_at: None,
            confirmation_feedback: None,
            last_notification_delivered: false,
            last_notification_at: None,
        }
    }

    /// Build an Autonomous goal for autonomous session tests.
    #[allow(clippy::too_many_arguments)]
    fn make_autonomous_goal(
        id: &str,
        desc: &str,
        status: GoalStatus,
        priority: GoalPriority,
        steps: Vec<Step>,
        context: &str,
        last_error: Option<String>,
        success_criteria: Option<&str>,
        constraints: Option<&str>,
        working_memory: Option<&str>,
        total_iterations: u32,
    ) -> Goal {
        Goal {
            id: id.into(),
            description: desc.into(),
            status,
            priority,
            created_at: String::new(),
            updated_at: String::new(),
            steps,
            context: context.into(),
            last_error,
            success_criteria: success_criteria.map(String::from),
            constraints: constraints.map(String::from),
            working_memory: working_memory.map(String::from),
            execution_mode: GoalExecutionMode::Autonomous,
            total_iterations,
            confirmation_plan: None,
            confirmation_requested_at: None,
            confirmation_feedback: None,
            last_notification_delivered: false,
            last_notification_at: None,
        }
    }

    fn sample_goal_state() -> GoalState {
        GoalState {
            goals: vec![
                make_goal(
                    "g1",
                    "Build automation platform",
                    GoalStatus::InProgress,
                    GoalPriority::High,
                    vec![
                        Step {
                            id: "s1".into(),
                            description: "Research tools".into(),
                            status: StepStatus::Completed,
                            result: Some("Found 3 tools".into()),
                            attempts: 1,
                        },
                        Step {
                            id: "s2".into(),
                            description: "Setup environment".into(),
                            status: StepStatus::Pending,
                            result: None,
                            attempts: 0,
                        },
                        Step {
                            id: "s3".into(),
                            description: "Write code".into(),
                            status: StepStatus::Pending,
                            result: None,
                            attempts: 0,
                        },
                    ],
                    "Using Python + Selenium",
                    None,
                ),
                make_goal(
                    "g2",
                    "Learn Rust",
                    GoalStatus::InProgress,
                    GoalPriority::Medium,
                    vec![Step {
                        id: "s1".into(),
                        description: "Read the book".into(),
                        status: StepStatus::Pending,
                        result: None,
                        attempts: 0,
                    }],
                    "",
                    None,
                ),
            ],
        }
    }

    #[test]
    fn goal_loop_config_serde_roundtrip() {
        let toml_str = r#"
enabled = true
interval_minutes = 15
step_timeout_secs = 180
max_steps_per_cycle = 5
channel = "lark"
target = "oc_test"
"#;
        let config: crate::config::GoalLoopConfig = toml::from_str(toml_str).unwrap();
        assert!(config.enabled);
        assert_eq!(config.interval_minutes, 15);
        assert_eq!(config.step_timeout_secs, 180);
        assert_eq!(config.max_steps_per_cycle, 5);
        assert_eq!(config.channel.as_deref(), Some("lark"));
        assert_eq!(config.target.as_deref(), Some("oc_test"));
    }

    #[test]
    fn goal_loop_config_defaults() {
        let config = crate::config::GoalLoopConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.interval_minutes, 10);
        assert_eq!(config.step_timeout_secs, 120);
        assert_eq!(config.max_steps_per_cycle, 3);
        assert!(config.channel.is_none());
        assert!(config.target.is_none());
    }

    #[test]
    fn goal_state_serde_roundtrip() {
        let state = sample_goal_state();
        let json = serde_json::to_string_pretty(&state).unwrap();
        let parsed: GoalState = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.goals.len(), 2);
        assert_eq!(parsed.goals[0].steps.len(), 3);
        assert_eq!(parsed.goals[0].steps[0].status, StepStatus::Completed);
    }

    #[test]
    fn select_next_actionable_picks_highest_priority() {
        let state = sample_goal_state();
        let result = GoalEngine::select_next_actionable(&state);
        // g1 (High) step s2 should be selected over g2 (Medium)
        assert_eq!(result, Some((0, 1)));
    }

    #[test]
    fn select_next_actionable_skips_exhausted_steps() {
        let mut state = sample_goal_state();
        // Exhaust s2 attempts
        state.goals[0].steps[1].attempts = MAX_STEP_ATTEMPTS;
        let result = GoalEngine::select_next_actionable(&state);
        // Should skip s2, pick s3
        assert_eq!(result, Some((0, 2)));
    }

    #[test]
    fn select_next_actionable_skips_non_in_progress_goals() {
        let mut state = sample_goal_state();
        state.goals[0].status = GoalStatus::Completed;
        let result = GoalEngine::select_next_actionable(&state);
        // g1 completed, should pick g2 s1
        assert_eq!(result, Some((1, 0)));
    }

    #[test]
    fn select_next_actionable_returns_none_when_nothing_actionable() {
        let state = GoalState::default();
        assert!(GoalEngine::select_next_actionable(&state).is_none());
    }

    #[test]
    fn build_step_prompt_includes_goal_and_step() {
        let state = sample_goal_state();
        let prompt = GoalEngine::build_step_prompt(&state.goals[0], &state.goals[0].steps[1]);
        assert!(prompt.contains("Build automation platform"));
        assert!(prompt.contains("Setup environment"));
        assert!(prompt.contains("Research tools"));
        assert!(prompt.contains("Using Python + Selenium"));
        assert!(!prompt.contains("WARNING")); // no retries yet
    }

    #[test]
    fn build_step_prompt_includes_retry_warning() {
        let mut state = sample_goal_state();
        state.goals[0].steps[1].attempts = 2;
        state.goals[0].last_error = Some("connection refused".into());
        let prompt = GoalEngine::build_step_prompt(&state.goals[0], &state.goals[0].steps[1]);
        assert!(prompt.contains("WARNING"));
        assert!(prompt.contains("2 time(s)"));
        assert!(prompt.contains("connection refused"));
    }

    #[test]
    fn interpret_result_success() {
        assert!(GoalEngine::interpret_result(
            "Successfully set up the environment"
        ));
        assert!(GoalEngine::interpret_result("Done. All tasks completed."));
    }

    #[test]
    fn interpret_result_failure() {
        assert!(!GoalEngine::interpret_result("Failed to install package"));
        assert!(!GoalEngine::interpret_result(
            "Error: connection timeout occurred"
        ));
        assert!(!GoalEngine::interpret_result("Unable to find the resource"));
        assert!(!GoalEngine::interpret_result("cannot open file"));
        assert!(!GoalEngine::interpret_result("Fatal: repository not found"));
    }

    #[tokio::test]
    async fn load_save_state_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let engine = GoalEngine::new(tmp.path());

        // Initially empty
        let empty = engine.load_state().await.unwrap();
        assert!(empty.goals.is_empty());

        // Save and reload
        let state = sample_goal_state();
        engine.save_state(&state).await.unwrap();
        let loaded = engine.load_state().await.unwrap();
        assert_eq!(loaded.goals.len(), 2);
        assert_eq!(loaded.goals[0].id, "g1");
        assert_eq!(loaded.goals[1].priority, GoalPriority::Medium);
    }

    #[test]
    fn priority_ordering() {
        assert!(GoalPriority::Critical > GoalPriority::High);
        assert!(GoalPriority::High > GoalPriority::Medium);
        assert!(GoalPriority::Medium > GoalPriority::Low);
    }

    #[test]
    fn goal_status_default_is_pending() {
        assert_eq!(GoalStatus::default(), GoalStatus::Pending);
    }

    #[test]
    fn step_status_default_is_pending() {
        assert_eq!(StepStatus::default(), StepStatus::Pending);
    }

    #[test]
    fn find_stalled_goals_detects_exhausted_steps() {
        let state = GoalState {
            goals: vec![make_goal(
                "g1",
                "Stalled goal",
                GoalStatus::InProgress,
                GoalPriority::High,
                vec![
                    Step {
                        id: "s1".into(),
                        description: "Done step".into(),
                        status: StepStatus::Completed,
                        result: Some("ok".into()),
                        attempts: 1,
                    },
                    Step {
                        id: "s2".into(),
                        description: "Exhausted step".into(),
                        status: StepStatus::Pending,
                        result: None,
                        attempts: 3, // >= MAX_STEP_ATTEMPTS
                    },
                ],
                "",
                Some("step failed 3 times".into()),
            )],
        };

        let stalled = GoalEngine::find_stalled_goals(&state);
        assert_eq!(stalled, vec![0]);
    }

    #[test]
    fn find_stalled_goals_ignores_actionable_goals() {
        let state = sample_goal_state(); // has pending steps with attempts=0
        let stalled = GoalEngine::find_stalled_goals(&state);
        assert!(stalled.is_empty());
    }

    #[test]
    fn find_stalled_goals_ignores_completed_goals() {
        let state = GoalState {
            goals: vec![make_goal(
                "g1",
                "Done",
                GoalStatus::Completed,
                GoalPriority::Medium,
                vec![Step {
                    id: "s1".into(),
                    description: "Only step".into(),
                    status: StepStatus::Completed,
                    result: Some("ok".into()),
                    attempts: 1,
                }],
                "",
                None,
            )],
        };

        let stalled = GoalEngine::find_stalled_goals(&state);
        assert!(stalled.is_empty());
    }

    #[test]
    fn build_reflection_prompt_includes_step_summary() {
        let goal = make_goal(
            "g1",
            "Test reflection",
            GoalStatus::InProgress,
            GoalPriority::High,
            vec![
                Step {
                    id: "s1".into(),
                    description: "Completed step".into(),
                    status: StepStatus::Completed,
                    result: Some("worked".into()),
                    attempts: 1,
                },
                Step {
                    id: "s2".into(),
                    description: "Failed step".into(),
                    status: StepStatus::Pending,
                    result: None,
                    attempts: 3,
                },
            ],
            "some context",
            Some("policy_denied".into()),
        );

        let prompt = GoalEngine::build_reflection_prompt(&goal);
        assert!(prompt.contains("[Goal Reflection]"));
        assert!(prompt.contains("Test reflection"));
        assert!(prompt.contains("[done] Completed step"));
        assert!(prompt.contains("[exhausted] Failed step"));
        assert!(prompt.contains("some context"));
        assert!(prompt.contains("policy_denied"));
        assert!(prompt.contains("memory_store"));
    }

    // ── Self-healing deserialization tests ───────────────────────

    #[test]
    fn goal_status_deserializes_all_valid_variants() {
        let cases = vec![
            ("\"pending\"", GoalStatus::Pending),
            (
                "\"awaiting_confirmation\"",
                GoalStatus::AwaitingConfirmation,
            ),
            ("\"in_progress\"", GoalStatus::InProgress),
            ("\"completed\"", GoalStatus::Completed),
            ("\"blocked\"", GoalStatus::Blocked),
            ("\"cancelled\"", GoalStatus::Cancelled),
        ];
        for (json_str, expected) in cases {
            let parsed: GoalStatus =
                serde_json::from_str(json_str).unwrap_or_else(|e| panic!("{json_str}: {e}"));
            assert_eq!(parsed, expected, "GoalStatus mismatch for {json_str}");
        }
    }

    #[test]
    fn goal_status_self_healing_unknown_variants() {
        for variant in &[
            "\"unknown\"",
            "\"invalid\"",
            "\"PENDING\"",
            "\"IN_PROGRESS\"",
            "\"\"",
        ] {
            let parsed: GoalStatus =
                serde_json::from_str(variant).unwrap_or_else(|e| panic!("{variant}: {e}"));
            assert_eq!(parsed, GoalStatus::Pending);
        }
    }

    #[test]
    fn step_status_deserializes_all_valid_variants() {
        let cases = vec![
            ("\"pending\"", StepStatus::Pending),
            ("\"in_progress\"", StepStatus::InProgress),
            ("\"completed\"", StepStatus::Completed),
            ("\"failed\"", StepStatus::Failed),
            ("\"blocked\"", StepStatus::Blocked),
        ];
        for (json_str, expected) in cases {
            let parsed: StepStatus =
                serde_json::from_str(json_str).unwrap_or_else(|e| panic!("{json_str}: {e}"));
            assert_eq!(parsed, expected, "StepStatus mismatch for {json_str}");
        }
    }

    #[test]
    fn step_status_self_healing_unknown_variants() {
        for variant in &["\"unknown\"", "\"done\"", "\"FAILED\"", "\"\""] {
            let parsed: StepStatus =
                serde_json::from_str(variant).unwrap_or_else(|e| panic!("{variant}: {e}"));
            assert_eq!(parsed, StepStatus::Pending);
        }
    }

    #[test]
    fn goal_status_self_healing_in_full_goal_json() {
        let json = r#"{"id":"g1","description":"test","status":"totally_bogus","steps":[]}"#;
        let goal: Goal = serde_json::from_str(json).unwrap();
        assert_eq!(goal.status, GoalStatus::Pending);
    }

    // ── find_stalled_goals edge cases ───────────────────────────

    #[test]
    fn find_stalled_goals_empty_steps_not_stalled() {
        let state = GoalState {
            goals: vec![make_goal(
                "g1",
                "No steps",
                GoalStatus::InProgress,
                GoalPriority::High,
                vec![],
                "",
                None,
            )],
        };
        assert!(GoalEngine::find_stalled_goals(&state).is_empty());
    }

    #[test]
    fn find_stalled_goals_multiple_stalled() {
        let stalled_goal = |id: &str| {
            make_goal(
                id,
                &format!("Stalled {id}"),
                GoalStatus::InProgress,
                GoalPriority::Medium,
                vec![Step {
                    id: "s1".into(),
                    description: "Exhausted".into(),
                    status: StepStatus::Pending,
                    result: None,
                    attempts: MAX_STEP_ATTEMPTS,
                }],
                "",
                None,
            )
        };
        let state = GoalState {
            goals: vec![stalled_goal("g1"), stalled_goal("g2"), stalled_goal("g3")],
        };
        assert_eq!(GoalEngine::find_stalled_goals(&state), vec![0, 1, 2]);
    }

    #[test]
    fn find_stalled_goals_all_steps_completed_is_stalled() {
        let state = GoalState {
            goals: vec![make_goal(
                "g1",
                "All done but still in-progress",
                GoalStatus::InProgress,
                GoalPriority::High,
                vec![
                    Step {
                        id: "s1".into(),
                        description: "Done".into(),
                        status: StepStatus::Completed,
                        result: Some("ok".into()),
                        attempts: 1,
                    },
                    Step {
                        id: "s2".into(),
                        description: "Also done".into(),
                        status: StepStatus::Completed,
                        result: Some("ok".into()),
                        attempts: 1,
                    },
                ],
                "",
                None,
            )],
        };
        assert_eq!(GoalEngine::find_stalled_goals(&state), vec![0]);
    }

    #[test]
    fn find_stalled_goals_mix_completed_and_blocked_steps() {
        let state = GoalState {
            goals: vec![make_goal(
                "g1",
                "Mixed",
                GoalStatus::InProgress,
                GoalPriority::High,
                vec![
                    Step {
                        id: "s1".into(),
                        description: "Done".into(),
                        status: StepStatus::Completed,
                        result: Some("ok".into()),
                        attempts: 1,
                    },
                    Step {
                        id: "s2".into(),
                        description: "Blocked".into(),
                        status: StepStatus::Blocked,
                        result: None,
                        attempts: 0,
                    },
                ],
                "",
                None,
            )],
        };
        assert_eq!(GoalEngine::find_stalled_goals(&state), vec![0]);
    }

    // ── build_reflection_prompt edge cases ───────────────────────

    #[test]
    fn build_reflection_prompt_empty_context_omits_section() {
        let goal = make_goal(
            "g1",
            "Empty context",
            GoalStatus::InProgress,
            GoalPriority::High,
            vec![Step {
                id: "s1".into(),
                description: "Step".into(),
                status: StepStatus::Completed,
                result: Some("ok".into()),
                attempts: 1,
            }],
            "",
            None,
        );
        let prompt = GoalEngine::build_reflection_prompt(&goal);
        assert!(!prompt.contains("Accumulated context"));
    }

    #[test]
    fn build_reflection_prompt_no_last_error_omits_section() {
        let goal = make_goal(
            "g1",
            "No error",
            GoalStatus::InProgress,
            GoalPriority::High,
            vec![Step {
                id: "s1".into(),
                description: "Step".into(),
                status: StepStatus::Completed,
                result: Some("ok".into()),
                attempts: 1,
            }],
            "some ctx",
            None,
        );
        let prompt = GoalEngine::build_reflection_prompt(&goal);
        assert!(!prompt.contains("Last error"));
    }

    #[test]
    fn build_reflection_prompt_all_done_tags() {
        let goal = make_goal(
            "g1",
            "All done",
            GoalStatus::InProgress,
            GoalPriority::High,
            vec![
                Step {
                    id: "s1".into(),
                    description: "First".into(),
                    status: StepStatus::Completed,
                    result: Some("ok".into()),
                    attempts: 1,
                },
                Step {
                    id: "s2".into(),
                    description: "Second".into(),
                    status: StepStatus::Completed,
                    result: Some("ok".into()),
                    attempts: 1,
                },
            ],
            "",
            None,
        );
        let prompt = GoalEngine::build_reflection_prompt(&goal);
        assert!(prompt.contains("[done] First"));
        assert!(prompt.contains("[done] Second"));
        assert!(!prompt.contains("[exhausted]"));
        assert!(!prompt.contains("[blocked]"));
    }

    // ── GoalPriority comparison and serde ────────────────────────

    #[test]
    fn priority_all_comparisons() {
        assert!(GoalPriority::Critical > GoalPriority::High);
        assert!(GoalPriority::High > GoalPriority::Medium);
        assert!(GoalPriority::Medium > GoalPriority::Low);
        assert!(GoalPriority::Low < GoalPriority::Critical);
    }

    #[test]
    fn priority_serde_roundtrip_all_variants() {
        for priority in &[
            GoalPriority::Low,
            GoalPriority::Medium,
            GoalPriority::High,
            GoalPriority::Critical,
        ] {
            let json = serde_json::to_string(priority).unwrap();
            let parsed: GoalPriority = serde_json::from_str(&json).unwrap();
            assert_eq!(*priority, parsed);
        }
    }

    // ── build_exploration_prompt tests ───────────────────────────

    #[test]
    fn exploration_prompt_empty_state() {
        let state = GoalState::default();
        let prompt = GoalEngine::build_exploration_prompt(&state);
        assert!(prompt.contains("[Idle Exploration]"));
        assert!(prompt.contains("== Completed goals ==\n(none)"));
        assert!(prompt.contains("== Blocked goals ==\n(none)"));
        assert!(prompt.contains("== Awaiting confirmation"));
        assert!(prompt.contains("== Pending goals (awaiting approval"));
        // Phase 1: Targeted Reconnaissance
        assert!(prompt.contains("exploration journal"));
        assert!(prompt.contains("SOUL.md"));
        assert!(prompt.contains("consolidation"));
        // Phase 2: Deep Exploration — four directions
        assert!(prompt.contains("Direction A"));
        assert!(prompt.contains("Direction B"));
        assert!(prompt.contains("Direction C"));
        assert!(prompt.contains("Direction D"));
        // Phase 3: Output — goal creation is default
        assert!(prompt.contains("in_progress"));
        assert!(prompt.contains("autonomous"));
        assert!(prompt.contains("success_criteria"));
        assert!(prompt.contains("VERIFY preconditions"));
        // Journal is fallback only
        assert!(prompt.contains("genuinely found nothing actionable"));
        assert!(prompt.contains("memory_store"));
    }

    #[test]
    fn exploration_prompt_output_oriented() {
        let state = GoalState::default();
        let prompt = GoalEngine::build_exploration_prompt(&state);
        // Goal creation is the DEFAULT, not optional
        assert!(prompt.contains("Your task is to discover and create"));
        assert!(prompt.contains("Your DEFAULT output is to create a goal"));
        // Goal quality rules
        assert!(prompt.contains("MUST be completable and verifiable"));
        assert!(prompt.contains("MUST NOT require user intervention"));
        assert!(prompt.contains("MUST NOT duplicate"));
        // Budget allocation
        assert!(prompt.contains("max 3 tool calls"));
        assert!(prompt.contains("max 6 tool calls"));
        assert!(prompt.contains("max 3 tool calls"));
        assert!(prompt.contains("Max 12 tool calls total"));
    }

    #[test]
    fn exploration_prompt_mixed_state() {
        let state = GoalState {
            goals: vec![
                make_goal(
                    "g1",
                    "Completed task",
                    GoalStatus::Completed,
                    GoalPriority::High,
                    vec![],
                    "",
                    None,
                ),
                make_goal(
                    "g2",
                    "Blocked task",
                    GoalStatus::Blocked,
                    GoalPriority::Medium,
                    vec![],
                    "",
                    Some("network timeout".into()),
                ),
                make_goal(
                    "g3",
                    "Pending task",
                    GoalStatus::Pending,
                    GoalPriority::Low,
                    vec![],
                    "",
                    None,
                ),
            ],
        };
        let prompt = GoalEngine::build_exploration_prompt(&state);
        assert!(prompt.contains("- Completed task"));
        assert!(prompt.contains("- Blocked task [error: network timeout]"));
        assert!(prompt.contains("- Pending task"));
    }

    #[test]
    fn exploration_prompt_blocked_goals_include_last_error() {
        let state = GoalState {
            goals: vec![
                make_goal(
                    "g1",
                    "Blocked with error",
                    GoalStatus::Blocked,
                    GoalPriority::High,
                    vec![],
                    "",
                    Some("API rate limit".into()),
                ),
                make_goal(
                    "g2",
                    "Blocked no error",
                    GoalStatus::Blocked,
                    GoalPriority::Low,
                    vec![],
                    "",
                    None,
                ),
            ],
        };
        let prompt = GoalEngine::build_exploration_prompt(&state);
        assert!(prompt.contains("[error: API rate limit]"));
        assert!(prompt.contains("[error: (no error recorded)]"));
    }

    // ── GoalLoopConfig idle exploration serde tests ──────────────

    #[test]
    fn goal_loop_config_explore_fields_serde() {
        let toml_str = r#"
enabled = true
interval_minutes = 15
step_timeout_secs = 180
max_steps_per_cycle = 5
explore_when_idle = true
explore_cooldown_minutes = 30
max_explorations_per_day = 4
"#;
        let config: crate::config::GoalLoopConfig = toml::from_str(toml_str).unwrap();
        assert!(config.explore_when_idle);
        assert_eq!(config.explore_cooldown_minutes, 30);
        assert_eq!(config.max_explorations_per_day, 4);
    }

    #[test]
    fn goal_loop_config_explore_defaults() {
        let config = crate::config::GoalLoopConfig::default();
        assert!(!config.explore_when_idle);
        assert_eq!(config.explore_cooldown_minutes, 60);
        assert_eq!(config.max_explorations_per_day, 6);
        assert!(!config.auto_approve_low_priority);
        assert_eq!(config.default_execution_mode, "autonomous");
        assert_eq!(config.autonomous_timeout_secs, 600);
        assert_eq!(config.max_total_goal_iterations, 200);
    }

    #[test]
    fn goal_loop_config_backward_compat_no_explore_fields() {
        let toml_str = r#"
enabled = true
interval_minutes = 10
step_timeout_secs = 120
max_steps_per_cycle = 3
"#;
        let config: crate::config::GoalLoopConfig = toml::from_str(toml_str).unwrap();
        assert!(config.enabled);
        assert!(!config.explore_when_idle);
        assert_eq!(config.explore_cooldown_minutes, 60);
        assert_eq!(config.max_explorations_per_day, 6);
        assert!(!config.auto_approve_low_priority);
        // New autonomous fields have defaults when not present
        assert_eq!(config.default_execution_mode, "autonomous");
        assert_eq!(config.autonomous_timeout_secs, 600);
        assert_eq!(config.max_total_goal_iterations, 200);
    }

    #[test]
    fn goal_loop_config_auto_approve_serde() {
        let toml_str = r#"
enabled = true
interval_minutes = 10
step_timeout_secs = 120
max_steps_per_cycle = 3
auto_approve_low_priority = true
"#;
        let config: crate::config::GoalLoopConfig = toml::from_str(toml_str).unwrap();
        assert!(config.auto_approve_low_priority);
    }

    #[test]
    fn exploration_prompt_mentions_auto_approve() {
        let state = GoalState::default();
        let prompt = GoalEngine::build_exploration_prompt(&state);
        assert!(
            prompt.contains("auto-approved"),
            "exploration prompt should inform agent about auto-approval"
        );
    }

    #[test]
    fn exploration_prompt_four_directions() {
        let state = GoalState::default();
        let prompt = GoalEngine::build_exploration_prompt(&state);
        assert!(
            prompt.contains("Direction A") && prompt.contains("Completed Goal Follow-up"),
            "exploration prompt should include Direction A"
        );
        assert!(
            prompt.contains("Direction B") && prompt.contains("Implicit User Needs"),
            "exploration prompt should include Direction B"
        );
        assert!(
            prompt.contains("Direction C") && prompt.contains("Capability Gaps"),
            "exploration prompt should include Direction C"
        );
        assert!(
            prompt.contains("Direction D") && prompt.contains("External Changes"),
            "exploration prompt should include Direction D"
        );
    }

    // ── GoalExecutionMode serde tests ──────────────────────────

    #[test]
    fn execution_mode_serde_roundtrip() {
        let auto = GoalExecutionMode::Autonomous;
        let json = serde_json::to_string(&auto).unwrap();
        assert_eq!(json, "\"autonomous\"");
        let parsed: GoalExecutionMode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, GoalExecutionMode::Autonomous);

        let stepped = GoalExecutionMode::Stepped;
        let json = serde_json::to_string(&stepped).unwrap();
        assert_eq!(json, "\"stepped\"");
        let parsed: GoalExecutionMode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, GoalExecutionMode::Stepped);
    }

    #[test]
    fn execution_mode_self_healing_unknown() {
        for variant in &["\"unknown\"", "\"legacy\"", "\"AUTONOMOUS\"", "\"\""] {
            let parsed: GoalExecutionMode =
                serde_json::from_str(variant).unwrap_or_else(|e| panic!("{variant}: {e}"));
            assert_eq!(parsed, GoalExecutionMode::Autonomous);
        }
    }

    #[test]
    fn execution_mode_default_is_autonomous() {
        assert_eq!(GoalExecutionMode::default(), GoalExecutionMode::Autonomous);
    }

    #[test]
    fn goal_backward_compat_no_autonomous_fields() {
        let json = r#"{
            "id": "g1",
            "description": "Legacy goal",
            "status": "in_progress",
            "steps": []
        }"#;
        let goal: Goal = serde_json::from_str(json).unwrap();
        assert_eq!(goal.execution_mode, GoalExecutionMode::Autonomous);
        assert_eq!(goal.total_iterations, 0);
        assert!(goal.success_criteria.is_none());
        assert!(goal.constraints.is_none());
        assert!(goal.working_memory.is_none());
    }

    #[test]
    fn goal_with_autonomous_fields_roundtrip() {
        let goal = make_autonomous_goal(
            "g1",
            "Autonomous test",
            GoalStatus::InProgress,
            GoalPriority::High,
            vec![],
            "",
            None,
            Some("all tests pass"),
            Some("max 10 tool calls"),
            Some("last session: compiled OK"),
            42,
        );
        let json = serde_json::to_string_pretty(&goal).unwrap();
        let parsed: Goal = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.execution_mode, GoalExecutionMode::Autonomous);
        assert_eq!(parsed.total_iterations, 42);
        assert_eq!(parsed.success_criteria.as_deref(), Some("all tests pass"));
        assert_eq!(parsed.constraints.as_deref(), Some("max 10 tool calls"));
        assert_eq!(
            parsed.working_memory.as_deref(),
            Some("last session: compiled OK")
        );
    }

    // ── select_next_autonomous_goal tests ───────────────────────

    #[test]
    fn select_autonomous_goal_picks_highest_priority() {
        let state = GoalState {
            goals: vec![
                make_autonomous_goal(
                    "g1",
                    "Low auto",
                    GoalStatus::InProgress,
                    GoalPriority::Low,
                    vec![],
                    "",
                    None,
                    None,
                    None,
                    None,
                    0,
                ),
                make_autonomous_goal(
                    "g2",
                    "High auto",
                    GoalStatus::InProgress,
                    GoalPriority::High,
                    vec![],
                    "",
                    None,
                    None,
                    None,
                    None,
                    0,
                ),
            ],
        };
        assert_eq!(GoalEngine::select_next_autonomous_goal(&state), Some(1));
    }

    #[test]
    fn select_autonomous_goal_skips_stepped() {
        let state = GoalState {
            goals: vec![make_goal(
                "g1",
                "Stepped goal",
                GoalStatus::InProgress,
                GoalPriority::High,
                vec![],
                "",
                None,
            )],
        };
        assert!(GoalEngine::select_next_autonomous_goal(&state).is_none());
    }

    #[test]
    fn select_autonomous_goal_skips_non_in_progress() {
        let state = GoalState {
            goals: vec![
                make_autonomous_goal(
                    "g1",
                    "Completed",
                    GoalStatus::Completed,
                    GoalPriority::High,
                    vec![],
                    "",
                    None,
                    None,
                    None,
                    None,
                    0,
                ),
                make_autonomous_goal(
                    "g2",
                    "Blocked",
                    GoalStatus::Blocked,
                    GoalPriority::High,
                    vec![],
                    "",
                    None,
                    None,
                    None,
                    None,
                    0,
                ),
                make_autonomous_goal(
                    "g3",
                    "Pending",
                    GoalStatus::Pending,
                    GoalPriority::High,
                    vec![],
                    "",
                    None,
                    None,
                    None,
                    None,
                    0,
                ),
            ],
        };
        assert!(GoalEngine::select_next_autonomous_goal(&state).is_none());
    }

    #[test]
    fn select_autonomous_goal_empty_state() {
        let state = GoalState::default();
        assert!(GoalEngine::select_next_autonomous_goal(&state).is_none());
    }

    // ── select_next_actionable skips autonomous goals ───────────

    #[test]
    fn select_next_actionable_skips_autonomous_goals() {
        let state = GoalState {
            goals: vec![make_autonomous_goal(
                "g1",
                "Auto goal with steps",
                GoalStatus::InProgress,
                GoalPriority::High,
                vec![Step {
                    id: "s1".into(),
                    description: "A step".into(),
                    status: StepStatus::Pending,
                    result: None,
                    attempts: 0,
                }],
                "",
                None,
                None,
                None,
                None,
                0,
            )],
        };
        // select_next_actionable only picks Stepped goals
        assert!(GoalEngine::select_next_actionable(&state).is_none());
        // select_next_autonomous_goal does find it
        assert_eq!(GoalEngine::select_next_autonomous_goal(&state), Some(0));
    }

    // ── find_stalled_goals skips autonomous goals ────────────────

    #[test]
    fn find_stalled_goals_skips_autonomous_goals() {
        let state = GoalState {
            goals: vec![make_autonomous_goal(
                "g1",
                "Auto goal all steps done",
                GoalStatus::InProgress,
                GoalPriority::High,
                vec![Step {
                    id: "s1".into(),
                    description: "Done".into(),
                    status: StepStatus::Completed,
                    result: Some("ok".into()),
                    attempts: 1,
                }],
                "",
                None,
                None,
                None,
                None,
                0,
            )],
        };
        // Autonomous goals are not considered stalled
        assert!(GoalEngine::find_stalled_goals(&state).is_empty());
    }

    // ── build_autonomous_prompt tests ────────────────────────────

    #[test]
    fn autonomous_prompt_basic() {
        let goal = make_autonomous_goal(
            "g1",
            "Deploy service",
            GoalStatus::InProgress,
            GoalPriority::High,
            vec![],
            "",
            None,
            None,
            None,
            None,
            0,
        );
        let state = GoalState {
            goals: vec![goal.clone()],
        };
        let prompt = GoalEngine::build_autonomous_prompt(&goal, &state);
        assert!(prompt.contains("[Autonomous Goal Session]"));
        assert!(prompt.contains("Deploy service"));
        assert!(prompt.contains("[GOAL_STATUS:"));
        assert!(prompt.contains("do NOT ask the user"));
    }

    #[test]
    fn autonomous_prompt_with_criteria_and_constraints() {
        let goal = make_autonomous_goal(
            "g1",
            "Build feature",
            GoalStatus::InProgress,
            GoalPriority::High,
            vec![],
            "",
            None,
            Some("all tests pass and CI green"),
            Some("do not modify config.toml"),
            None,
            0,
        );
        let state = GoalState {
            goals: vec![goal.clone()],
        };
        let prompt = GoalEngine::build_autonomous_prompt(&goal, &state);
        assert!(prompt.contains("== Success Criteria =="));
        assert!(prompt.contains("all tests pass and CI green"));
        assert!(prompt.contains("== Constraints =="));
        assert!(prompt.contains("do not modify config.toml"));
    }

    #[test]
    fn autonomous_prompt_with_working_memory() {
        let goal = make_autonomous_goal(
            "g1",
            "Continue work",
            GoalStatus::InProgress,
            GoalPriority::High,
            vec![],
            "",
            None,
            None,
            None,
            Some("last time: fixed 3 of 5 tests"),
            10,
        );
        let state = GoalState {
            goals: vec![goal.clone()],
        };
        let prompt = GoalEngine::build_autonomous_prompt(&goal, &state);
        assert!(prompt.contains("== Working Memory (from last session) =="));
        assert!(prompt.contains("last time: fixed 3 of 5 tests"));
    }

    #[test]
    fn autonomous_prompt_with_steps_as_reference() {
        let goal = make_autonomous_goal(
            "g1",
            "Multi-step",
            GoalStatus::InProgress,
            GoalPriority::High,
            vec![
                Step {
                    id: "s1".into(),
                    description: "Research".into(),
                    status: StepStatus::Completed,
                    result: Some("found 3 options".into()),
                    attempts: 1,
                },
                Step {
                    id: "s2".into(),
                    description: "Implement".into(),
                    status: StepStatus::Pending,
                    result: None,
                    attempts: 0,
                },
            ],
            "research done",
            None,
            None,
            None,
            None,
            5,
        );
        let state = GoalState {
            goals: vec![goal.clone()],
        };
        let prompt = GoalEngine::build_autonomous_prompt(&goal, &state);
        assert!(prompt.contains("== Steps (reference only"));
        assert!(prompt.contains("[done] Research: found 3 options"));
        assert!(prompt.contains("[pending] Implement"));
        assert!(prompt.contains("== Completed Context =="));
    }

    #[test]
    fn autonomous_prompt_with_last_error() {
        let goal = make_autonomous_goal(
            "g1",
            "Retry",
            GoalStatus::InProgress,
            GoalPriority::High,
            vec![],
            "",
            Some("connection refused".into()),
            None,
            None,
            None,
            0,
        );
        let state = GoalState {
            goals: vec![goal.clone()],
        };
        let prompt = GoalEngine::build_autonomous_prompt(&goal, &state);
        assert!(prompt.contains("== Last Error =="));
        assert!(prompt.contains("connection refused"));
    }

    #[test]
    fn autonomous_prompt_empty_working_memory_omitted() {
        let goal = make_autonomous_goal(
            "g1",
            "No WM",
            GoalStatus::InProgress,
            GoalPriority::High,
            vec![],
            "",
            None,
            None,
            None,
            Some(""),
            0,
        );
        let state = GoalState {
            goals: vec![goal.clone()],
        };
        let prompt = GoalEngine::build_autonomous_prompt(&goal, &state);
        assert!(!prompt.contains("Working Memory"));
    }

    // ── interpret_autonomous_result tests ─────────────────────────

    #[test]
    fn interpret_autonomous_completed() {
        let output = "I did the work.\n[GOAL_STATUS: completed]";
        assert_eq!(
            GoalEngine::interpret_autonomous_result(output),
            AutonomousSessionStatus::Completed,
        );
    }

    #[test]
    fn interpret_autonomous_in_progress() {
        let output = "Made progress, more to do.\n[GOAL_STATUS: in_progress]";
        assert_eq!(
            GoalEngine::interpret_autonomous_result(output),
            AutonomousSessionStatus::InProgress,
        );
    }

    #[test]
    fn interpret_autonomous_blocked() {
        let output = "Need API key.\n[GOAL_STATUS: blocked missing API credentials]";
        assert_eq!(
            GoalEngine::interpret_autonomous_result(output),
            AutonomousSessionStatus::Blocked("missing api credentials".into()),
        );
    }

    #[test]
    fn interpret_autonomous_blocked_no_reason() {
        let output = "Stuck.\n[GOAL_STATUS: blocked]";
        assert_eq!(
            GoalEngine::interpret_autonomous_result(output),
            AutonomousSessionStatus::Blocked("no reason given".into()),
        );
    }

    #[test]
    fn interpret_autonomous_last_tag_wins() {
        let output = "[GOAL_STATUS: in_progress]\nActually done.\n[GOAL_STATUS: completed]";
        assert_eq!(
            GoalEngine::interpret_autonomous_result(output),
            AutonomousSessionStatus::Completed,
        );
    }

    #[test]
    fn interpret_autonomous_fallback_success() {
        let output = "I did everything correctly and deployed.";
        assert_eq!(
            GoalEngine::interpret_autonomous_result(output),
            AutonomousSessionStatus::InProgress,
        );
    }

    #[test]
    fn interpret_autonomous_fallback_failure() {
        let output = "Failed to connect to database.";
        assert_eq!(
            GoalEngine::interpret_autonomous_result(output),
            AutonomousSessionStatus::Blocked("heuristic detected failure indicators".into()),
        );
    }

    #[test]
    fn interpret_autonomous_tag_in_long_output() {
        let mut output = "x".repeat(1000);
        output.push_str("\n[GOAL_STATUS: completed]");
        assert_eq!(
            GoalEngine::interpret_autonomous_result(&output),
            AutonomousSessionStatus::Completed,
        );
    }

    // ── extract_working_memory tests ─────────────────────────────

    #[test]
    fn extract_wm_after_tag() {
        let output = "Work done.\n[GOAL_STATUS: completed]\nRemember: file.rs line 42 needs fix";
        let wm = GoalEngine::extract_working_memory(output, 2000);
        assert_eq!(wm, "Remember: file.rs line 42 needs fix");
    }

    #[test]
    fn extract_wm_no_tag() {
        let output = "Just some output with no tag";
        let wm = GoalEngine::extract_working_memory(output, 2000);
        assert_eq!(wm, "Just some output with no tag");
    }

    #[test]
    fn extract_wm_truncates() {
        let output = "No tag. ".repeat(500);
        let wm = GoalEngine::extract_working_memory(&output, 50);
        assert!(wm.len() <= 50);
    }

    #[test]
    fn extract_wm_empty_after_tag() {
        let output = "Done.\n[GOAL_STATUS: completed]";
        let wm = GoalEngine::extract_working_memory(output, 2000);
        // Nothing after tag, so falls through to truncating whole output
        assert_eq!(wm, output.trim());
    }

    #[test]
    fn extract_wm_tag_with_only_whitespace_after() {
        let output = "Done.\n[GOAL_STATUS: completed]   \n  ";
        let wm = GoalEngine::extract_working_memory(output, 2000);
        // Only whitespace after tag, falls through
        assert_eq!(wm, output.trim());
    }

    // ── GoalLoopConfig autonomous fields serde ───────────────────

    #[test]
    fn goal_loop_config_autonomous_fields_serde() {
        let toml_str = r#"
enabled = true
interval_minutes = 10
step_timeout_secs = 120
max_steps_per_cycle = 3
default_execution_mode = "stepped"
autonomous_timeout_secs = 300
max_total_goal_iterations = 100
"#;
        let config: crate::config::GoalLoopConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.default_execution_mode, "stepped");
        assert_eq!(config.autonomous_timeout_secs, 300);
        assert_eq!(config.max_total_goal_iterations, 100);
    }

    #[test]
    fn goal_loop_config_autonomous_fields_defaults() {
        let toml_str = r#"
enabled = true
interval_minutes = 10
step_timeout_secs = 120
max_steps_per_cycle = 3
"#;
        let config: crate::config::GoalLoopConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.default_execution_mode, "autonomous");
        assert_eq!(config.autonomous_timeout_secs, 600);
        assert_eq!(config.max_total_goal_iterations, 200);
    }

    // ── truncate_str tests ───────────────────────────────────────

    #[test]
    fn truncate_str_ascii() {
        assert_eq!(super::truncate_str("hello world", 5), "hello");
        assert_eq!(super::truncate_str("hello", 10), "hello");
        assert_eq!(super::truncate_str("", 5), "");
    }

    #[test]
    fn truncate_str_cjk() {
        // Each CJK char is 3 bytes — truncate by char count, not byte count
        let s = "你好世界测试消息"; // 8 chars, 24 bytes
        assert_eq!(super::truncate_str(s, 4), "你好世界");
        assert_eq!(super::truncate_str(s, 2), "你好");
        assert_eq!(super::truncate_str(s, 8), s); // exact boundary
        assert_eq!(super::truncate_str(s, 100), s); // longer than string
    }

    #[test]
    fn truncate_str_mixed_multibyte() {
        // Mix of ASCII (1 byte), accented (2 bytes), CJK (3 bytes), emoji (4 bytes)
        let s = "aé你🦀"; // 4 chars, 10 bytes
        assert_eq!(super::truncate_str(s, 2), "aé");
        assert_eq!(super::truncate_str(s, 3), "aé你");
        assert_eq!(super::truncate_str(s, 4), s);
    }

    #[test]
    fn truncate_str_emoji() {
        let s = "😀😀😀😀"; // 4 chars, 16 bytes
        assert_eq!(super::truncate_str(s, 2), "😀😀");
        assert_eq!(super::truncate_str(s, 0), "");
    }

    #[test]
    fn truncate_str_goal_status_scenario() {
        // Simulate the real scenario that caused the panic: Chinese text with
        // [GOAL_STATUS: blocked] tag, truncated at char boundary
        let s = "状态已更新。\n\n---\n\n[GOAL_STATUS: blocked REASON]\n\n\
                 **阻塞原因：安全策略限制**\n- 所有带路径参数的命令都被拦截";
        let truncated = super::truncate_str(s, 60);
        // Must not panic and must be valid UTF-8
        assert!(truncated.len() <= s.len());
        assert!(truncated.is_char_boundary(truncated.len()));
    }

    // ── build_intent_scan_prompt tests ──────────────────────────

    fn make_memory_entry(key: &str, content: &str, timestamp: &str) -> crate::memory::MemoryEntry {
        crate::memory::MemoryEntry {
            id: key.to_string(),
            key: key.to_string(),
            content: content.to_string(),
            category: crate::memory::MemoryCategory::Conversation,
            timestamp: timestamp.to_string(),
            session_id: None,
            score: None,
        }
    }

    #[test]
    fn intent_scan_prompt_with_messages_and_goals() {
        let messages = vec![
            make_memory_entry(
                "feishu_1",
                "Please deploy the service",
                "2026-02-23T10:00:00Z",
            ),
            make_memory_entry("feishu_2", "Fix the login bug", "2026-02-23T10:05:00Z"),
        ];
        let state = GoalState {
            goals: vec![make_goal(
                "g1",
                "Fix login bug",
                GoalStatus::InProgress,
                GoalPriority::High,
                vec![],
                "",
                None,
            )],
        };
        let prompt = GoalEngine::build_intent_scan_prompt(&messages, &state);

        assert!(prompt.contains("[Intent Scan]"));
        assert!(prompt.contains("Please deploy the service"));
        assert!(prompt.contains("Fix the login bug"));
        assert!(prompt.contains("Fix login bug"));
        assert!(prompt.contains("(in_progress)"));
        assert!(prompt.contains("Max 3 tool calls"));
    }

    #[test]
    fn intent_scan_prompt_no_messages() {
        let state = GoalState::default();
        let prompt = GoalEngine::build_intent_scan_prompt(&[], &state);

        assert!(prompt.contains("[Intent Scan]"));
        assert!(prompt.contains("== Recent User Messages ==\n(none)"));
        assert!(prompt.contains("== Current Goals ==\n(none)"));
    }

    #[test]
    fn intent_scan_prompt_no_goals() {
        let messages = vec![make_memory_entry(
            "feishu_1",
            "Set up monitoring",
            "2026-02-23T10:00:00Z",
        )];
        let state = GoalState::default();
        let prompt = GoalEngine::build_intent_scan_prompt(&messages, &state);

        assert!(prompt.contains("Set up monitoring"));
        assert!(prompt.contains("== Current Goals ==\n(none)"));
    }

    #[test]
    fn intent_scan_prompt_all_statuses() {
        let state = GoalState {
            goals: vec![
                make_goal(
                    "g1",
                    "Pending",
                    GoalStatus::Pending,
                    GoalPriority::Low,
                    vec![],
                    "",
                    None,
                ),
                make_goal(
                    "g2",
                    "Active",
                    GoalStatus::InProgress,
                    GoalPriority::Medium,
                    vec![],
                    "",
                    None,
                ),
                make_goal(
                    "g3",
                    "Done",
                    GoalStatus::Completed,
                    GoalPriority::High,
                    vec![],
                    "",
                    None,
                ),
                make_goal(
                    "g4",
                    "Stuck",
                    GoalStatus::Blocked,
                    GoalPriority::High,
                    vec![],
                    "",
                    None,
                ),
                make_goal(
                    "g5",
                    "Dropped",
                    GoalStatus::Cancelled,
                    GoalPriority::Low,
                    vec![],
                    "",
                    None,
                ),
                make_goal(
                    "g6",
                    "Awaiting",
                    GoalStatus::AwaitingConfirmation,
                    GoalPriority::Medium,
                    vec![],
                    "",
                    None,
                ),
            ],
        };
        let prompt = GoalEngine::build_intent_scan_prompt(&[], &state);

        assert!(prompt.contains("(pending)"));
        assert!(prompt.contains("(awaiting_confirmation)"));
        assert!(prompt.contains("(in_progress)"));
        assert!(prompt.contains("(completed)"));
        assert!(prompt.contains("(blocked)"));
        assert!(prompt.contains("(cancelled)"));
    }

    // ── AwaitingConfirmation serde tests ─────────────────────────

    #[test]
    fn goal_status_awaiting_confirmation_serde_roundtrip() {
        let status = GoalStatus::AwaitingConfirmation;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"awaiting_confirmation\"");
        let parsed: GoalStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, GoalStatus::AwaitingConfirmation);
    }

    #[test]
    fn goal_with_confirmation_fields_roundtrip() {
        let mut goal = make_goal(
            "g1",
            "Test confirmation",
            GoalStatus::AwaitingConfirmation,
            GoalPriority::High,
            vec![Step {
                id: "s1".into(),
                description: "Do thing".into(),
                status: StepStatus::Pending,
                result: None,
                attempts: 0,
            }],
            "",
            None,
        );
        goal.confirmation_plan = Some("I understand you want X. Plan: 1,2,3".into());
        goal.confirmation_requested_at = Some("2026-02-23T10:00:00Z".into());
        goal.confirmation_feedback = None;

        let json = serde_json::to_string_pretty(&goal).unwrap();
        let parsed: Goal = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.status, GoalStatus::AwaitingConfirmation);
        assert_eq!(
            parsed.confirmation_plan.as_deref(),
            Some("I understand you want X. Plan: 1,2,3")
        );
        assert_eq!(
            parsed.confirmation_requested_at.as_deref(),
            Some("2026-02-23T10:00:00Z")
        );
        assert!(parsed.confirmation_feedback.is_none());
    }

    #[test]
    fn goal_backward_compat_no_confirmation_fields() {
        let json = r#"{
            "id": "g1",
            "description": "Legacy goal",
            "status": "in_progress",
            "steps": []
        }"#;
        let goal: Goal = serde_json::from_str(json).unwrap();
        assert!(goal.confirmation_plan.is_none());
        assert!(goal.confirmation_requested_at.is_none());
        assert!(goal.confirmation_feedback.is_none());
    }

    // ── build_understanding_prompt tests ─────────────────────────

    #[test]
    fn understanding_prompt_basic() {
        let goal = make_goal(
            "g1",
            "Build a secure skill downloader",
            GoalStatus::AwaitingConfirmation,
            GoalPriority::High,
            vec![],
            "",
            None,
        );
        let state = GoalState {
            goals: vec![goal.clone()],
        };
        let prompt = GoalEngine::build_understanding_prompt(&goal, &state);
        assert!(prompt.contains("[Understanding Confirmation]"));
        assert!(prompt.contains("Build a secure skill downloader"));
        assert!(prompt.contains("My Understanding"));
        assert!(prompt.contains("Execution Plan"));
        assert!(prompt.contains("Expected Output"));
        assert!(prompt.contains("Assumptions & Risks"));
        assert!(prompt.contains("Max 3 tool calls"));
        assert!(prompt.contains("Do NOT start executing"));
    }

    #[test]
    fn understanding_prompt_with_feedback() {
        let mut goal = make_goal(
            "g1",
            "Build a skill downloader",
            GoalStatus::AwaitingConfirmation,
            GoalPriority::High,
            vec![],
            "",
            None,
        );
        goal.confirmation_feedback =
            Some("I want a NEW tool, not an audit of existing code".into());
        let state = GoalState {
            goals: vec![goal.clone()],
        };
        let prompt = GoalEngine::build_understanding_prompt(&goal, &state);
        assert!(prompt.contains("User Feedback"));
        assert!(prompt.contains("REJECTED"));
        assert!(prompt.contains("I want a NEW tool, not an audit"));
    }

    #[test]
    fn understanding_prompt_with_criteria_and_constraints() {
        let mut goal = make_goal(
            "g1",
            "Build feature",
            GoalStatus::AwaitingConfirmation,
            GoalPriority::High,
            vec![],
            "",
            None,
        );
        goal.success_criteria = Some("all tests pass".into());
        goal.constraints = Some("no new dependencies".into());
        let state = GoalState::default();
        let prompt = GoalEngine::build_understanding_prompt(&goal, &state);
        assert!(prompt.contains("== Success Criteria =="));
        assert!(prompt.contains("all tests pass"));
        assert!(prompt.contains("== Constraints =="));
        assert!(prompt.contains("no new dependencies"));
    }

    // ── append_goals_by_status includes awaiting_confirmation ────

    #[test]
    fn exploration_prompt_shows_awaiting_confirmation_goals() {
        let state = GoalState {
            goals: vec![make_goal(
                "g1",
                "Awaiting goal",
                GoalStatus::AwaitingConfirmation,
                GoalPriority::High,
                vec![],
                "",
                None,
            )],
        };
        let prompt = GoalEngine::build_exploration_prompt(&state);
        assert!(prompt.contains("Awaiting goal [WAITING: confirmation pending]"));
    }

    // ── load_and_normalize tests ─────────────────────────────────

    #[tokio::test]
    async fn normalize_fills_empty_id() {
        let tmp = TempDir::new().unwrap();
        let engine = GoalEngine::new(tmp.path());

        // Write a goal with empty id directly (simulating LLM file_write)
        let raw = r#"{"goals":[{"id":"","description":"test","status":"pending","steps":[]}]}"#;
        let state_dir = tmp.path().join("state");
        std::fs::create_dir_all(&state_dir).unwrap();
        std::fs::write(state_dir.join("goals.json"), raw).unwrap();

        let state = engine.load_and_normalize().await.unwrap();
        assert_eq!(state.goals.len(), 1);
        assert!(!state.goals[0].id.is_empty(), "empty id should be filled");
        // Verify it's a valid UUID
        assert!(uuid::Uuid::parse_str(&state.goals[0].id).is_ok());
    }

    #[tokio::test]
    async fn normalize_fills_empty_timestamps() {
        let tmp = TempDir::new().unwrap();
        let engine = GoalEngine::new(tmp.path());

        let raw = r#"{"goals":[{"id":"g1","description":"test","status":"pending","created_at":"","updated_at":"","steps":[]}]}"#;
        let state_dir = tmp.path().join("state");
        std::fs::create_dir_all(&state_dir).unwrap();
        std::fs::write(state_dir.join("goals.json"), raw).unwrap();

        let state = engine.load_and_normalize().await.unwrap();
        assert!(!state.goals[0].created_at.is_empty());
        assert!(!state.goals[0].updated_at.is_empty());
    }

    #[tokio::test]
    async fn normalize_deduplicates_by_id() {
        let tmp = TempDir::new().unwrap();
        let engine = GoalEngine::new(tmp.path());

        let raw = r#"{"goals":[
            {"id":"g1","description":"first","status":"pending","steps":[]},
            {"id":"g2","description":"unique","status":"pending","steps":[]},
            {"id":"g1","description":"duplicate-last","status":"in_progress","steps":[]}
        ]}"#;
        let state_dir = tmp.path().join("state");
        std::fs::create_dir_all(&state_dir).unwrap();
        std::fs::write(state_dir.join("goals.json"), raw).unwrap();

        let state = engine.load_and_normalize().await.unwrap();
        assert_eq!(state.goals.len(), 2);
        // g2 (unique) retains its position; g1's last occurrence wins
        assert_eq!(state.goals[0].id, "g2");
        assert_eq!(state.goals[0].description, "unique");
        assert_eq!(state.goals[1].id, "g1");
        assert_eq!(state.goals[1].description, "duplicate-last");
    }

    #[tokio::test]
    async fn normalize_drops_unknown_fields() {
        let tmp = TempDir::new().unwrap();
        let engine = GoalEngine::new(tmp.path());

        // Unknown field "llm_junk" should be dropped on re-save
        let raw = r#"{"goals":[{"id":"g1","description":"test","status":"pending","steps":[],"llm_junk":"should_be_dropped"}]}"#;
        let state_dir = tmp.path().join("state");
        std::fs::create_dir_all(&state_dir).unwrap();
        std::fs::write(state_dir.join("goals.json"), raw).unwrap();

        let _ = engine.load_and_normalize().await.unwrap();

        // Re-read raw file and verify unknown field is gone
        let saved = std::fs::read_to_string(state_dir.join("goals.json")).unwrap();
        assert!(
            !saved.contains("llm_junk"),
            "unknown field should be dropped after normalize"
        );
    }

    #[tokio::test]
    async fn normalize_heals_invalid_enum() {
        let tmp = TempDir::new().unwrap();
        let engine = GoalEngine::new(tmp.path());

        // "BOGUS_STATUS" should self-heal to "pending" via custom Deserialize
        let raw =
            r#"{"goals":[{"id":"g1","description":"test","status":"BOGUS_STATUS","steps":[]}]}"#;
        let state_dir = tmp.path().join("state");
        std::fs::create_dir_all(&state_dir).unwrap();
        std::fs::write(state_dir.join("goals.json"), raw).unwrap();

        let state = engine.load_and_normalize().await.unwrap();
        assert_eq!(state.goals[0].status, GoalStatus::Pending);

        // After re-save, status should be canonical "pending"
        let saved = std::fs::read_to_string(state_dir.join("goals.json")).unwrap();
        assert!(saved.contains("\"pending\""));
    }

    #[tokio::test]
    async fn normalize_noop_on_clean_state() {
        let tmp = TempDir::new().unwrap();
        let engine = GoalEngine::new(tmp.path());

        // Save a clean state first
        let state = sample_goal_state();
        engine.save_state(&state).await.unwrap();

        // Normalize should not error and should return same data
        let normalized = engine.load_and_normalize().await.unwrap();
        assert_eq!(normalized.goals.len(), state.goals.len());
    }

    #[tokio::test]
    async fn normalize_empty_state() {
        let tmp = TempDir::new().unwrap();
        let engine = GoalEngine::new(tmp.path());

        let state = engine.load_and_normalize().await.unwrap();
        assert!(state.goals.is_empty());
    }

    #[test]
    fn deserialize_goal_without_notification_fields() {
        // Old goals.json will not have the notification tracking fields.
        // Ensure serde(default) lets them deserialize without error.
        let json = r#"{
            "id": "g-old",
            "description": "Legacy goal",
            "status": "in_progress",
            "priority": "medium",
            "created_at": "",
            "updated_at": "",
            "steps": [],
            "context": ""
        }"#;
        let goal: Goal = serde_json::from_str(json).unwrap();
        assert!(!goal.last_notification_delivered);
        assert!(goal.last_notification_at.is_none());
    }

    #[test]
    fn step_id_accepts_integer() {
        let json = r#"{"id": 42, "description": "test step", "status": "pending"}"#;
        let step: Step = serde_json::from_str(json).unwrap();
        assert_eq!(step.id, "42");
    }

    #[test]
    fn step_id_accepts_array() {
        let json = r#"{"id": ["a", "b"], "description": "test step"}"#;
        let step: Step = serde_json::from_str(json).unwrap();
        assert_eq!(step.id, "a, b");
    }

    #[test]
    fn step_id_accepts_bool() {
        let json = r#"{"id": true, "description": "test step"}"#;
        let step: Step = serde_json::from_str(json).unwrap();
        assert_eq!(step.id, "true");
    }

    #[test]
    fn goal_string_fields_accept_arrays() {
        // LLM sometimes writes arrays for string fields
        let json = r#"{
            "id": "g1",
            "description": ["do thing A", "then thing B"],
            "status": "pending",
            "steps": []
        }"#;
        let goal: Goal = serde_json::from_str(json).unwrap();
        assert_eq!(goal.description, "do thing A, then thing B");
    }

    #[test]
    fn goal_opt_string_null_stays_none() {
        let json = r#"{
            "id": "g1",
            "description": "test",
            "success_criteria": null,
            "constraints": null,
            "steps": []
        }"#;
        let goal: Goal = serde_json::from_str(json).unwrap();
        assert!(goal.success_criteria.is_none());
        assert!(goal.constraints.is_none());
    }

    #[test]
    fn goal_opt_string_accepts_array() {
        let json = r#"{
            "id": "g1",
            "description": "test",
            "success_criteria": ["all tests pass", "no regressions"],
            "steps": []
        }"#;
        let goal: Goal = serde_json::from_str(json).unwrap();
        assert_eq!(
            goal.success_criteria.as_deref(),
            Some("all tests pass, no regressions")
        );
    }

    #[test]
    fn notification_fields_roundtrip() {
        let mut goal = make_goal(
            "g-rt",
            "Roundtrip test",
            GoalStatus::InProgress,
            GoalPriority::Medium,
            vec![],
            "",
            None,
        );
        goal.last_notification_delivered = true;
        goal.last_notification_at = Some("2026-02-24T12:00:00Z".into());

        let json = serde_json::to_string(&goal).unwrap();
        let restored: Goal = serde_json::from_str(&json).unwrap();
        assert!(restored.last_notification_delivered);
        assert_eq!(
            restored.last_notification_at.as_deref(),
            Some("2026-02-24T12:00:00Z"),
        );
    }

    #[tokio::test]
    async fn normalize_resets_hallucinated_blocked_goal() {
        let tmp = TempDir::new().unwrap();
        let engine = GoalEngine::new(tmp.path());

        // Simulate LLM writing a blocked goal with 0 iterations and 0 attempts
        let state = GoalState {
            goals: vec![make_goal(
                "g-hallucinated",
                "Build a calculator",
                GoalStatus::Blocked,
                GoalPriority::High,
                vec![Step {
                    id: "1".into(),
                    description: "Use coding-agent".into(),
                    status: StepStatus::Pending,
                    result: Some("Codex API exhausted".into()), // hallucinated
                    attempts: 0,
                }],
                "fabricated context",
                Some("fabricated error".into()),
            )],
        };
        engine.save_state(&state).await.unwrap();

        let normalized = engine.load_and_normalize().await.unwrap();
        let g = &normalized.goals[0];
        assert_eq!(g.status, GoalStatus::Pending, "should reset to pending");
        assert!(g.last_error.is_none(), "should clear hallucinated error");
        assert!(
            g.steps[0].result.is_none(),
            "should clear hallucinated step result"
        );
    }

    #[tokio::test]
    async fn normalize_keeps_legitimately_blocked_goal() {
        let tmp = TempDir::new().unwrap();
        let engine = GoalEngine::new(tmp.path());

        // Goal that was actually executed and blocked
        let state = GoalState {
            goals: vec![{
                let g = make_autonomous_goal(
                    "g-real-blocked",
                    "Real goal",
                    GoalStatus::Blocked,
                    GoalPriority::High,
                    vec![],
                    "",
                    Some("real error after execution".into()),
                    None,
                    None,
                    None,
                    50, // total_iterations > 0
                );
                g
            }],
        };
        engine.save_state(&state).await.unwrap();

        let normalized = engine.load_and_normalize().await.unwrap();
        let g = &normalized.goals[0];
        assert_eq!(g.status, GoalStatus::Blocked, "should stay blocked");
        assert!(g.last_error.is_some(), "should keep real error");
    }

    #[tokio::test]
    async fn normalize_keeps_blocked_goal_with_real_step_attempts() {
        let tmp = TempDir::new().unwrap();
        let engine = GoalEngine::new(tmp.path());

        // Stepped goal that actually attempted steps
        let state = GoalState {
            goals: vec![make_goal(
                "g-step-blocked",
                "Stepped goal",
                GoalStatus::Blocked,
                GoalPriority::High,
                vec![Step {
                    id: "1".into(),
                    description: "Failed step".into(),
                    status: StepStatus::Pending,
                    result: Some("actual failure output".into()),
                    attempts: 3, // real attempts
                }],
                "",
                Some("Step failed 3 times".into()),
            )],
        };
        engine.save_state(&state).await.unwrap();

        let normalized = engine.load_and_normalize().await.unwrap();
        let g = &normalized.goals[0];
        assert_eq!(g.status, GoalStatus::Blocked, "should stay blocked");
        assert!(g.steps[0].result.is_some(), "should keep real result");
    }

    #[tokio::test]
    async fn normalize_preserves_completed_goal_step_results() {
        let tmp = TempDir::new().unwrap();
        let engine = GoalEngine::new(tmp.path());

        // Completed goal with step results but 0 attempts (LLM-managed steps)
        let state = GoalState {
            goals: vec![make_goal(
                "g-done",
                "Completed goal",
                GoalStatus::Completed,
                GoalPriority::Medium,
                vec![Step {
                    id: "1".into(),
                    description: "Search stuff".into(),
                    status: StepStatus::Completed,
                    result: Some("Found useful data".into()),
                    attempts: 0, // LLM never incremented attempts
                }],
                "",
                None,
            )],
        };
        engine.save_state(&state).await.unwrap();

        let normalized = engine.load_and_normalize().await.unwrap();
        let g = &normalized.goals[0];
        assert_eq!(g.status, GoalStatus::Completed);
        assert_eq!(
            g.steps[0].result.as_deref(),
            Some("Found useful data"),
            "should preserve completed goal step results"
        );
    }
}
