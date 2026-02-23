use anyhow::Result;
use serde::{Deserialize, Serialize};
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
    pub id: String,
    pub description: String,
    #[serde(default)]
    pub status: GoalStatus,
    #[serde(default)]
    pub priority: GoalPriority,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub updated_at: String,
    #[serde(default)]
    pub steps: Vec<Step>,
    /// Accumulated context from previous step results.
    #[serde(default)]
    pub context: String,
    /// Last error encountered during step execution.
    #[serde(default)]
    pub last_error: Option<String>,

    // ── Autonomous session fields ───────────────────────────────
    /// Success criteria for autonomous mode (what "done" looks like).
    #[serde(default)]
    pub success_criteria: Option<String>,
    /// Constraints the agent must respect during autonomous execution.
    #[serde(default)]
    pub constraints: Option<String>,
    /// Persisted working memory from the last autonomous session.
    #[serde(default)]
    pub working_memory: Option<String>,
    /// Execution mode: `autonomous` (default) or `stepped`.
    #[serde(default)]
    pub execution_mode: GoalExecutionMode,
    /// Total tool iterations consumed across all autonomous sessions.
    #[serde(default)]
    pub total_iterations: u32,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum GoalStatus {
    #[default]
    Pending,
    InProgress,
    Completed,
    Blocked,
    Cancelled,
}

impl<'de> Deserialize<'de> for GoalStatus {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Ok(match s.as_str() {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    pub id: String,
    pub description: String,
    #[serde(default)]
    pub status: StepStatus,
    #[serde(default)]
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

/// Truncate a string to at most `max` characters on a char boundary.
fn truncate_str(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    // Find the largest char boundary <= max
    let mut end = max;
    while !s.is_char_boundary(end) && end > 0 {
        end -= 1;
    }
    &s[..end]
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
             pass context to yourself across sessions.\n",
        );

        prompt
    }

    /// Parse the agent output to determine autonomous session status.
    ///
    /// Priority: (1) explicit `[GOAL_STATUS: ...]` tag in last 500 chars,
    /// (2) fallback to existing `interpret_result()` heuristic (conservatively
    /// returns InProgress rather than Completed).
    pub fn interpret_autonomous_result(output: &str) -> AutonomousSessionStatus {
        // Search in the tail of the output for the status tag
        let search_region = if output.len() > 500 {
            &output[output.len() - 500..]
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

    /// Build an exploration prompt for when no goals are active.
    ///
    /// The agent is asked to review context, check for unfinished threads,
    /// and optionally propose new goals with verified preconditions.
    pub fn build_exploration_prompt(state: &GoalState) -> String {
        let mut prompt = String::new();

        prompt.push_str(
            "[Idle Exploration] No active goals. Review context and consider proposing new work.\n\n",
        );

        Self::append_goals_by_status(&mut prompt, state);

        prompt.push_str(
            "Instructions:\n\n\
             == Phase 1: Recall and Reflect ==\n\
             1. Use memory_recall with query \"exploration journal\" to retrieve your past\n\
             \x20  exploration entries (category: exploration). Review what you explored\n\
             \x20  before, what you found, and what directions you suggested for next time.\n\
             2. Use memory_recall with query \"exploration scores\" to retrieve the latest\n\
             \x20  exploration scorecard (category: exploration, key: exploration-scores-*).\n\
             \x20  This tells you which directions scored well (continue) and which scored\n\
             \x20  poorly (deprioritize). DO NOT repeat directions listed in\n\
             \x20  `directions_to_deprioritize` unless you have a strong new reason.\n\
             3. Use memory_recall with query \"consolidation\" to retrieve recent nightly\n\
             \x20  consolidation summaries (category: core). These distill the full day's\n\
             \x20  activity — cron results, errors, and discoveries you may have missed.\n\
             4. Use memory_recall to review recent daily activity entries.\n\
             5. Read SOUL.md to re-ground yourself in the user's mission areas.\n\
             6. REFLECT on your exploration history, scores, AND consolidation learnings:\n\
             \x20  - Am I stuck in a rut, repeating the same topics?\n\
             \x20  - Which past explorations scored well? Why? How can I build on them?\n\
             \x20  - Which scored poorly? Am I about to repeat a deprioritized direction?\n\
             \x20  - Has the user's situation changed (new conversations, new priorities)?\n\
             \x20  - What blind spots might I have? What am I NOT looking at?\n\n\
             == Phase 2: Explore with Intent ==\n\
             7. Based on your reflection, choose ONE direction to explore. Prefer:\n\
             \x20  - Directions listed in `directions_to_continue` from the scorecard\n\
             \x20  - Directions you suggested in your last journal entry\n\
             \x20  - Consolidation learnings that suggest unfinished threads\n\
             \x20  - Blind spots or areas you haven't covered recently\n\
             \x20  - Follow-ups to completed goals that might have new developments\n\
             \x20  AVOID: directions in `directions_to_deprioritize`, recent repeats.\n\
             8. If you identify a valuable new goal:\n\
             \x20  a. VERIFY the precondition first (e.g., check if data exists, if a\n\
             \x20     service is reachable, if the user hasn't already addressed it)\n\
             \x20  b. Only if preconditions hold: write it to state/goals.json with\n\
             \x20     status \"pending\", priority \"low\", and 3-5 concrete steps\n\
             \x20  c. Goals with priority \"low\" will be auto-approved and executed\n\
             \x20     by the goal loop without waiting for human approval\n\
             \x20  d. Notify the user with a brief rationale\n\n\
             == Phase 3: Journal ==\n\
             9. ALWAYS end by writing an exploration journal entry using memory_store.\n\
             \x20  Use key \"exploration-journal-YYYY-MM-DD-N\" (N = sequence number today).\n\
             \x20  Use category \"exploration\". Include:\n\
             \x20  - What you explored and why\n\
             \x20  - Key findings (or \"nothing notable\")\n\
             \x20  - Self-critique: what went well, what was wasted effort\n\
             \x20  - 2-3 specific directions for next exploration (not vague, be concrete)\n\
             \x20  - Any shifts in your mental model of the user's priorities\n\n\
             == Constraints ==\n\
             - Max 12 tool calls total.\n\
             - If nothing valuable surfaces, say so briefly. Do not force a goal.\n\
             - Quality over quantity. One deep insight beats five shallow scans.\n",
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

        buf.push_str("== Pending goals (awaiting approval) ==\n");
        let pending: Vec<&Goal> = state
            .goals
            .iter()
            .filter(|g| g.status == GoalStatus::Pending)
            .collect();
        if pending.is_empty() {
            buf.push_str("(none)\n");
        } else {
            for g in &pending {
                let _ = writeln!(buf, "- {}", g.description);
            }
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
        assert!(prompt.contains("== Pending goals (awaiting approval) ==\n(none)"));
        // Phase 1: Recall and Reflect
        assert!(prompt.contains("exploration journal"));
        assert!(prompt.contains("REFLECT on your exploration history"));
        assert!(prompt.contains("blind spots"));
        // Phase 2: Explore with Intent
        assert!(prompt.contains("VERIFY the precondition"));
        // Phase 3: Journal
        assert!(prompt.contains("memory_store"));
        assert!(prompt.contains("exploration-journal-"));
        assert!(prompt.contains("Self-critique"));
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
    fn exploration_prompt_mentions_scorecard() {
        let state = GoalState::default();
        let prompt = GoalEngine::build_exploration_prompt(&state);
        assert!(
            prompt.contains("exploration scores"),
            "exploration prompt should read scorecard"
        );
        assert!(
            prompt.contains("directions_to_deprioritize"),
            "exploration prompt should mention deprioritize list"
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
    fn truncate_str_multibyte() {
        // Each CJK char is 3 bytes
        let s = "你好世界"; // 12 bytes
        let t = super::truncate_str(s, 7);
        // Should truncate to at most 7 bytes = 2 full chars (6 bytes)
        assert_eq!(t, "你好");
        assert!(t.len() <= 7);
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
            ],
        };
        let prompt = GoalEngine::build_intent_scan_prompt(&[], &state);

        assert!(prompt.contains("(pending)"));
        assert!(prompt.contains("(in_progress)"));
        assert!(prompt.contains("(completed)"));
        assert!(prompt.contains("(blocked)"));
        assert!(prompt.contains("(cancelled)"));
    }
}
