# Self-Evolution Design: Logic and Philosophy

This document describes ZeroClaw's self-evolution architecture — how the agent
runtime turns a stateless LLM into a continuously learning, self-reflecting
autonomous system.

> **Core thesis:** Intelligence comes from the model; reliability comes from the
> runtime. ZeroClaw is not building a "smarter AI" — it is building a **memory
> and reflection infrastructure** that makes an ordinary LLM reliable and
> continuously improving because it remembers, reflects, and has discipline.

## Five Engineering Capabilities

| Capability | Plain language | Engineering essence | Implementation |
|---|---|---|---|
| Knows who it is | DNA | Mandatory system prompt injection | `prompt.rs` IdentitySection |
| Learns from mistakes | Reflection system | Short-term error correction + long-term memory consolidation | FailureTracker + consolidation job |
| Finds its own work | Autonomous exploration | goal loop + idle exploration | `daemon/mod.rs` + `engine.rs` |
| Fully traceable | Audit logging | Tamper-resistant operation records | AuditLogger + cron_runs |
| Aware of the world | External perception | MCP / external data source integration | Planned |

## Philosophy 1: Cognitive Continuity over Single-Shot Intelligence

Traditional agents are stateless — context window ends, memory is gone. ZeroClaw
bets that **cross-session cognitive accumulation matters more than single-turn
reasoning quality**.

### DNA Injection

Every time the agent wakes up, `IdentitySection` (`prompt.rs:102-111`) reads
eight workspace files in fixed priority order:

```
AGENTS.md → SOUL.md → TOOLS.md → IDENTITY.md
→ USER.md → HEARTBEAT.md → BOOTSTRAP.md → MEMORY.md
```

The model has no tool that can modify its own system prompt. Identity is
hard-injected and tamper-proof.

### Memory Driveshaft

Cron jobs carry their previous execution result into the next wakeup. In
`scheduler.rs:151-164`, `build_agent_prompt()` appends:

```
[cron:job_id name] <original prompt>

[Previous run result]
<last_output from SQLite, max 16KB>
```

The agent does not start from scratch — it resumes from where it left off.

### Three-Phase Exploration Loop

Idle exploration (`engine.rs:267-318`) is not random wandering. It follows a
structured protocol:

1. **Recall and Reflect** — Read past exploration journals AND consolidation
   summaries. Ask: Am I repeating myself? Are there blind spots? Has the user's
   situation changed?

2. **Explore with Intent** — Based on reflection, choose ONE direction. Verify
   preconditions before proposing new goals. Prefer: suggested directions from
   last journal, consolidation learnings, blind spots, unblocking stuck goals.

3. **Write Journal** — Always write to brain.db (category: `exploration`, key:
   `exploration-journal-YYYY-MM-DD-N`). Include: findings, self-critique,
   2–3 concrete next directions, shifts in mental model.

Journals persist in brain.db and transfer across sessions via `memory_recall`.
Each exploration reads the previous rounds, forming **incremental cognition** —
not starting from zero every time.

## Philosophy 2: Autonomous but Traceable

Not "set it free" autonomy — **every step has an audit trail**.

### Full Tool Call Audit

Every tool call passes through `execute_one_tool` (`loop_.rs:1919-1944`), which
writes a structured audit event via `AuditLogger::log_tool_call()`: tool name,
success/failure, duration, error label. Arguments and output are deliberately
omitted to prevent secret leakage.

### Exploration Output Persistence

Idle exploration results are written to the `cron_runs` table (job_id:
`__idle_exploration`) in `daemon/mod.rs`. Even if the LLM does not follow the
prompt instruction to write a journal entry, the output is not lost. This is a
Rust-level hard guarantee, not a prompt-level hope.

### Multi-Layer Safety Constraints

- `AutonomyLevel` has exactly three values: `readonly` / `supervised` / `full`.
  No ambiguous intermediate states.
- Exploration is rate-limited: 6/day max, 60-minute cooldown, network
  pre-check (TCP connect with 5s timeout).
- `SecurityPolicy` is deny-by-default. `forbidden_paths` blocks at the Rust
  level.
- `FailureTracker`: same tool + same error repeated twice triggers a forced
  reflection prompt, compelling the model to change strategy.

## Philosophy 3: Minimal Closed Loop over Feature Sprawl

Not "build a perfect architecture first" — **get the smallest flywheel spinning,
then add thickness**.

### The Reflection Flywheel

```
          ┌──────────────────────────────────────────────┐
          │                                              │
          ▼                                              │
    cron wakeup                                          │
    (carries last_output)                                │
          │                                              │
          ▼                                              │
    execute task ──→ full tool audit                     │
          │         (AuditLogger)                         │
          ▼                                              │
    idle exploration                                     │
    ├─ read exploration journals                         │
    ├─ read exploration scorecard (scores + directions)  │
    ├─ read consolidation summaries  ◄── cross-pollinated│
    ├─ write journal (brain.db)                          │
    ├─ propose goal (pending, low priority) ──┐          │
    └─ write cron_runs (__idle_exploration)    │ hard backup
          │                                   │          │
          ▼                                   ▼          │
    goal loop auto-approve                               │
    (pending + low priority → in_progress)               │
          │                                              │
          ▼                                              │
    03:00 nightly consolidation                          │
    ├─ read cron_runs (incl. __idle_exploration)         │
    ├─ read exploration journals                         │
    ├─ score each exploration (1-5 scale)                │
    ├─ write scorecard (directions to continue/deprioritize)
    ├─ write consolidation summary (category: core)      │
    └─ append MEMORY.md ────────────────────────────────→┘
```

Key design decisions:

- Exploration journals reuse `memory_store`/`memory_recall` + brain.db. **No new
  storage infrastructure was created.**
- Nightly consolidation is a cron agent job + a prompt. **Not a separate
  subsystem.**
- Goal loop, cron, and idle exploration share a single `crate::agent::run()`
  entry point.
- The consolidation job is idempotently auto-registered on daemon startup.
  **No manual setup required.**

## Philosophy 4: Rust Guarantees over Prompt Expectations

This is the fundamental difference between ZeroClaw and pure prompt-engineering
approaches.

| Behavior | Prompt expectation | Rust guarantee |
|---|---|---|
| Remember last run | "Please recall previous conversation" | `last_output` hard-injected into prompt |
| Exploration results not lost | "Please write to memory_store" | `record_run` writes to cron_runs table |
| Don't repeat same mistake | "Please try a different approach" | FailureTracker threshold triggers forced reflection |
| Run consolidation every night | Manual cron job creation | `ensure_consolidation_job` auto-registers on startup |
| Tool calls are recorded | "Please log your actions" | `log_tool_call` in execute_one_tool, mandatory |
| Don't over-explore | "Please be mindful of frequency" | Daily cap + cooldown + network pre-check |

**Principle:** Every critical step the flywheel depends on is guaranteed by Rust
code, not by LLM probabilistic execution. Prompts guide direction and quality;
Rust guarantees structure and non-skippability.

## Philosophy 5: Quality-Gated Autonomy

Full autonomy without quality control produces noise. ZeroClaw uses a two-layer
quality gate to keep the flywheel productive.

### Exploration Quality Scoring

The nightly consolidation job scores each exploration journal entry on a 1–5
scale:

| Score | Meaning |
|---|---|
| 5 | Led to a concrete, actionable goal or verified new capability |
| 4 | Produced a valuable insight that changes understanding or unblocks work |
| 3 | Useful information gathered, but no immediate action resulted |
| 2 | Mostly noise — repeated known information or explored a dead end |
| 1 | Complete waste — no findings, wrong direction, or repeated a recent topic |

The scores are stored as a scorecard (`exploration-scores-YYYY-MM-DD`,
category: `exploration`) containing:
- Per-entry scores with one-line reasons
- `directions_to_continue`: high-scoring directions to pursue
- `directions_to_deprioritize`: low-scoring directions to avoid

The exploration prompt reads this scorecard and uses it to steer direction
selection: prefer `directions_to_continue`, avoid `directions_to_deprioritize`.
This creates a **natural selection loop** — productive directions are reinforced,
unproductive ones are pruned.

### Goal Auto-Approval

Exploration proposes new goals with `status: "pending"` and `priority: "low"`.
When `auto_approve_low_priority` is enabled in config (`[goal_loop]`), the goal
loop automatically promotes these to `in_progress` without human approval.

This closes the full autonomy loop:

```
explore → propose goal → auto-approve → execute steps → complete → explore again
```

Safety constraints:
- Only `low` priority goals are auto-approved. Medium/high/critical still
  require human approval.
- Goals must have at least one step defined (empty goals are skipped).
- The user is notified via the configured channel when goals are auto-approved.
- The scoring system deprioritizes low-quality exploration directions, which
  reduces the rate of low-value goals being proposed in the first place.

Config: `[goal_loop] auto_approve_low_priority = true` (default: `false`).

## Summary

ZeroClaw's self-evolution does not make AI smarter — it makes AI **remember,
reflect, and follow discipline**. It turns a forgetful genius into a reliable
operator, one who summarizes experience every night and wakes up knowing a little
more than yesterday.
