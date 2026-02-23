use crate::config::Config;
use anyhow::Result;
use chrono::Utc;
use std::collections::HashMap;
use std::fs::File;
use std::future::Future;
use std::io::{Read as _, Write as _};
use std::path::PathBuf;
use tokio::task::JoinHandle;
use tokio::time::{Duration, Instant};

const STATUS_FLUSH_SECONDS: u64 = 5;

pub async fn run(config: Config, host: String, port: u16) -> Result<()> {
    // Acquire PID lock to enforce single-instance daemon.
    // The lock is held for the lifetime of `run()` and released on exit/crash.
    let _pid_lock = acquire_pid_lock(&config)?;

    let initial_backoff = config.reliability.channel_initial_backoff_secs.max(1);
    let max_backoff = config
        .reliability
        .channel_max_backoff_secs
        .max(initial_backoff);

    crate::providers::reliable::init_llm_concurrency(config.reliability.max_concurrent_llm_calls);

    crate::health::mark_component_ok("daemon");

    if config.heartbeat.enabled {
        let _ =
            crate::heartbeat::engine::HeartbeatEngine::ensure_heartbeat_file(&config.workspace_dir)
                .await;
    }

    let mut handles: Vec<JoinHandle<()>> = vec![spawn_state_writer(config.clone())];

    {
        let gateway_cfg = config.clone();
        let gateway_host = host.clone();
        handles.push(spawn_component_supervisor(
            "gateway",
            initial_backoff,
            max_backoff,
            move || {
                let cfg = gateway_cfg.clone();
                let host = gateway_host.clone();
                async move { crate::gateway::run_gateway(&host, port, cfg).await }
            },
        ));
    }

    {
        if has_supervised_channels(&config) {
            let channels_cfg = config.clone();
            handles.push(spawn_component_supervisor(
                "channels",
                initial_backoff,
                max_backoff,
                move || {
                    let cfg = channels_cfg.clone();
                    async move { crate::channels::start_channels(cfg).await }
                },
            ));
        } else {
            crate::health::mark_component_ok("channels");
            tracing::info!("No real-time channels configured; channel supervisor disabled");
        }
    }

    if config.heartbeat.enabled {
        let heartbeat_cfg = config.clone();
        handles.push(spawn_component_supervisor(
            "heartbeat",
            initial_backoff,
            max_backoff,
            move || {
                let cfg = heartbeat_cfg.clone();
                async move { Box::pin(run_heartbeat_worker(cfg)).await }
            },
        ));
    }

    if config.goal_loop.enabled {
        let goal_cfg = config.clone();
        handles.push(spawn_component_supervisor(
            "goal-loop",
            initial_backoff,
            max_backoff,
            move || {
                let cfg = goal_cfg.clone();
                Box::pin(async move { Box::pin(run_goal_loop_worker(cfg)).await })
            },
        ));
    } else {
        crate::health::mark_component_ok("goal-loop");
        tracing::info!("Goal loop disabled; goal-loop supervisor not started");
    }

    if config.cron.enabled {
        // Ensure nightly consolidation job exists (idempotent).
        ensure_consolidation_job(&config);

        let scheduler_cfg = config.clone();
        handles.push(spawn_component_supervisor(
            "scheduler",
            initial_backoff,
            max_backoff,
            move || {
                let cfg = scheduler_cfg.clone();
                async move { crate::cron::scheduler::run(cfg).await }
            },
        ));
    } else {
        crate::health::mark_component_ok("scheduler");
        tracing::info!("Cron disabled; scheduler supervisor not started");
    }

    println!("🧠 ZeroClaw daemon started");
    println!("   Gateway:  http://{host}:{port}");
    println!("   Components: gateway, channels, heartbeat, goal-loop, scheduler");
    println!("   Ctrl+C to stop");

    tokio::signal::ctrl_c().await?;
    crate::health::mark_component_error("daemon", "shutdown requested");

    for handle in &handles {
        handle.abort();
    }
    for handle in handles {
        let _ = handle.await;
    }

    Ok(())
}

pub fn state_file_path(config: &Config) -> PathBuf {
    config
        .config_path
        .parent()
        .map_or_else(|| PathBuf::from("."), PathBuf::from)
        .join("daemon_state.json")
}

fn pid_lock_path(config: &Config) -> PathBuf {
    config
        .config_path
        .parent()
        .map_or_else(|| PathBuf::from("."), PathBuf::from)
        .join("daemon.pid")
}

/// Acquire an exclusive advisory lock on the PID file. Returns the open [`File`]
/// handle — the OS releases the lock automatically when the handle is dropped
/// (on normal exit or crash), so no stale-lock cleanup is needed.
fn acquire_pid_lock(config: &Config) -> Result<File> {
    use std::os::unix::io::AsRawFd;

    let path = pid_lock_path(config);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&path)?;

    let fd = file.as_raw_fd();
    let ret = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
    if ret != 0 {
        let err = std::io::Error::last_os_error();
        if err.kind() == std::io::ErrorKind::WouldBlock {
            let mut existing_pid = String::new();
            let _ = file.read_to_string(&mut existing_pid);
            let existing_pid = existing_pid.trim();
            anyhow::bail!(
                "Another ZeroClaw daemon is already running (PID {existing_pid}). \
                 Lock file: {}",
                path.display()
            );
        }
        return Err(err.into());
    }

    // Lock acquired — write our PID
    file.set_len(0)?;
    write!(file, "{}", std::process::id())?;
    file.sync_all()?;

    Ok(file)
}

fn spawn_state_writer(config: Config) -> JoinHandle<()> {
    tokio::spawn(async move {
        let path = state_file_path(&config);
        if let Some(parent) = path.parent() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }

        let mut interval = tokio::time::interval(Duration::from_secs(STATUS_FLUSH_SECONDS));
        loop {
            interval.tick().await;
            let mut json = crate::health::snapshot_json();
            if let Some(obj) = json.as_object_mut() {
                obj.insert(
                    "written_at".into(),
                    serde_json::json!(Utc::now().to_rfc3339()),
                );
            }
            let data = serde_json::to_vec_pretty(&json).unwrap_or_else(|_| b"{}".to_vec());
            let _ = tokio::fs::write(&path, data).await;
        }
    })
}

fn spawn_component_supervisor<F, Fut>(
    name: &'static str,
    initial_backoff_secs: u64,
    max_backoff_secs: u64,
    mut run_component: F,
) -> JoinHandle<()>
where
    F: FnMut() -> Fut + Send + 'static,
    Fut: Future<Output = Result<()>> + Send + 'static,
{
    tokio::spawn(async move {
        let mut backoff = initial_backoff_secs.max(1);
        let max_backoff = max_backoff_secs.max(backoff);

        loop {
            crate::health::mark_component_ok(name);
            match run_component().await {
                Ok(()) => {
                    crate::health::mark_component_error(name, "component exited unexpectedly");
                    tracing::warn!("Daemon component '{name}' exited unexpectedly");
                    // Clean exit — reset backoff since the component ran successfully
                    backoff = initial_backoff_secs.max(1);
                }
                Err(e) => {
                    crate::health::mark_component_error(name, e.to_string());
                    tracing::error!("Daemon component '{name}' failed: {e}");
                }
            }

            crate::health::bump_component_restart(name);
            tokio::time::sleep(Duration::from_secs(backoff)).await;
            // Double backoff AFTER sleeping so first error uses initial_backoff
            backoff = backoff.saturating_mul(2).min(max_backoff);
        }
    })
}

/// Maximum consecutive failures before a heartbeat task is auto-disabled
/// for the remainder of this daemon lifetime.
const HEARTBEAT_MAX_CONSECUTIVE_FAILURES: u32 = 3;

async fn run_heartbeat_worker(config: Config) -> Result<()> {
    let observer: std::sync::Arc<dyn crate::observability::Observer> =
        std::sync::Arc::from(crate::observability::create_observer(&config.observability));
    let engine = crate::heartbeat::engine::HeartbeatEngine::new(
        config.heartbeat.clone(),
        config.workspace_dir.clone(),
        observer,
    );

    let initial_backoff_secs = config.reliability.channel_initial_backoff_secs.max(1);
    let max_backoff_secs = config
        .reliability
        .channel_max_backoff_secs
        .max(initial_backoff_secs);

    // Per-task failure tracking: task_description -> (consecutive_failures, last_failure_at)
    let mut failure_map: HashMap<String, (u32, Instant)> = HashMap::new();

    let interval_mins = config.heartbeat.interval_minutes.max(5);
    let mut interval = tokio::time::interval(Duration::from_secs(u64::from(interval_mins) * 60));

    loop {
        interval.tick().await;

        let tasks = engine.collect_tasks().await?;
        if tasks.is_empty() {
            continue;
        }

        for task in tasks {
            // Check if task is permanently disabled (hit max failures)
            if let Some(&(failures, _)) = failure_map.get(&task) {
                if failures >= HEARTBEAT_MAX_CONSECUTIVE_FAILURES {
                    tracing::debug!(
                        "Heartbeat task disabled after {failures} consecutive failures, \
                         skipping: {task}"
                    );
                    continue;
                }

                // Check exponential backoff cooldown
                let backoff = initial_backoff_secs
                    .saturating_mul(1u64.checked_shl(failures).unwrap_or(u64::MAX))
                    .min(max_backoff_secs);
                let (_, last_failure_at) = failure_map[&task];
                if last_failure_at.elapsed() < Duration::from_secs(backoff) {
                    tracing::debug!("Heartbeat task in cooldown ({backoff}s), skipping: {task}");
                    continue;
                }
            }

            crate::health::set_component_activity("heartbeat", Some("running"));
            let prompt = format!("[Heartbeat Task] {task}");
            let temp = config.default_temperature;
            match crate::agent::run(
                config.clone(),
                Some(prompt),
                None,
                None,
                temp,
                vec![],
                false,
                false,
                Some("heartbeat"),
            )
            .await
            {
                Ok(output) => {
                    // Success: reset failure tracking for this task
                    failure_map.remove(&task);
                    crate::health::set_component_activity("heartbeat", None);
                    crate::health::mark_component_ok("heartbeat");
                    // Deliver to configured channel (best-effort)
                    if let (Some(ch_name), Some(target)) =
                        (&config.heartbeat.channel, &config.heartbeat.target)
                    {
                        if let Err(e) = deliver_heartbeat(&config, ch_name, target, &output).await {
                            tracing::warn!("Heartbeat delivery to {ch_name} failed: {e}");
                        }
                    }
                }
                Err(e) => {
                    crate::health::set_component_activity("heartbeat", None);
                    let (failures, _) = failure_map
                        .entry(task.clone())
                        .or_insert((0, Instant::now()));
                    *failures += 1;
                    let f = *failures;
                    // Update last_failure_at
                    failure_map.get_mut(&task).unwrap().1 = Instant::now();

                    if f >= HEARTBEAT_MAX_CONSECUTIVE_FAILURES {
                        tracing::error!(
                            "Heartbeat task disabled after {f} consecutive failures. \
                             Check HEARTBEAT.md configuration: {task} — error: {e}"
                        );
                    } else {
                        tracing::warn!(
                            "Heartbeat task failed ({f}/{HEARTBEAT_MAX_CONSECUTIVE_FAILURES}): {e}"
                        );
                    }
                    crate::health::mark_component_error("heartbeat", e.to_string());
                }
            }
        }
    }
}

async fn deliver_heartbeat(
    config: &Config,
    channel_name: &str,
    target: &str,
    output: &str,
) -> Result<()> {
    use crate::channels::{Channel, SendMessage};

    match channel_name.to_ascii_lowercase().as_str() {
        "lark" | "feishu" => {
            #[cfg(feature = "channel-lark")]
            {
                if let Some(fs) = config.channels_config.feishu.as_ref() {
                    let channel = crate::channels::LarkChannel::from_feishu_config(fs);
                    channel
                        .send_card(target, "🦀 ZeroClaw 心跳报告", output)
                        .await?;
                } else if let Some(lk) = config.channels_config.lark.as_ref() {
                    let channel = crate::channels::LarkChannel::from_config(lk);
                    channel
                        .send_card(target, "🦀 ZeroClaw 心跳报告", output)
                        .await?;
                } else {
                    anyhow::bail!("lark/feishu channel not configured");
                }
            }
            #[cfg(not(feature = "channel-lark"))]
            anyhow::bail!("lark channel requires the `channel-lark` build feature");
        }
        "telegram" => {
            let tg = config
                .channels_config
                .telegram
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("telegram channel not configured"))?;
            let channel = crate::channels::TelegramChannel::new(
                tg.bot_token.clone(),
                tg.allowed_users.clone(),
                tg.mention_only,
            );
            channel.send(&SendMessage::new(output, target)).await?;
        }
        "discord" => {
            let dc = config
                .channels_config
                .discord
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("discord channel not configured"))?;
            let channel = crate::channels::DiscordChannel::new(
                dc.bot_token.clone(),
                dc.guild_id.clone(),
                dc.allowed_users.clone(),
                dc.listen_to_bots,
                dc.mention_only,
            );
            channel.send(&SendMessage::new(output, target)).await?;
        }
        "slack" => {
            let sl = config
                .channels_config
                .slack
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("slack channel not configured"))?;
            let channel = crate::channels::SlackChannel::new(
                sl.bot_token.clone(),
                sl.channel_id.clone(),
                sl.allowed_users.clone(),
            );
            channel.send(&SendMessage::new(output, target)).await?;
        }
        "mattermost" => {
            let mm = config
                .channels_config
                .mattermost
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("mattermost channel not configured"))?;
            let channel = crate::channels::MattermostChannel::new(
                mm.url.clone(),
                mm.bot_token.clone(),
                mm.channel_id.clone(),
                mm.allowed_users.clone(),
                mm.thread_replies.unwrap_or(true),
                mm.mention_only.unwrap_or(false),
            );
            channel.send(&SendMessage::new(output, target)).await?;
        }
        other => anyhow::bail!("unsupported heartbeat delivery channel: {other}"),
    }

    Ok(())
}

// ── Goal Loop Worker ─────────────────────────────────────────────

/// Returns `true` if the state has goals that the loop can act on
/// (autonomous goals, stepped goals with actionable steps, or stalled goals).
fn has_actionable_goals(state: &crate::goals::engine::GoalState) -> bool {
    use crate::goals::engine::GoalEngine;
    GoalEngine::select_next_autonomous_goal(state).is_some()
        || GoalEngine::select_next_actionable(state).is_some()
        || !GoalEngine::find_stalled_goals(state).is_empty()
}

/// Lightweight network reachability check (TCP connect with 5s timeout).
/// Returns `true` if the provider endpoint is reachable.
async fn check_network_reachable(config: &Config) -> bool {
    let host = if let Some(ref provider) = config.default_provider {
        // Extract host from provider string like "anthropic-custom:https://open.bigmodel.cn/api/..."
        if let Some(url_part) = provider.split("https://").nth(1) {
            url_part.split('/').next().unwrap_or("").to_string()
        } else if let Some(url_part) = provider.split("http://").nth(1) {
            url_part.split('/').next().unwrap_or("").to_string()
        } else {
            // Known provider names → use their default hosts
            match provider.as_str() {
                "anthropic" => "api.anthropic.com".to_string(),
                "openai" => "api.openai.com".to_string(),
                p if p.starts_with("google") || p.starts_with("gemini") => {
                    "generativelanguage.googleapis.com".to_string()
                }
                _ => return true, // Unknown provider, skip check
            }
        }
    } else {
        return true; // No provider configured, skip check
    };

    if host.is_empty() {
        return true;
    }

    let addr = if host.contains(':') {
        host
    } else {
        format!("{host}:443")
    };

    tokio::time::timeout(
        Duration::from_secs(5),
        tokio::net::TcpStream::connect(&addr),
    )
    .await
    .is_ok_and(|r| r.is_ok())
}

async fn run_goal_loop_worker(config: Config) -> Result<()> {
    use crate::goals::engine::{
        GoalEngine, GoalExecutionMode, GoalPriority, GoalStatus, StepStatus,
    };

    let engine = GoalEngine::new(&config.workspace_dir);
    let interval_mins = config.goal_loop.interval_minutes.max(1);
    let max_steps = config.goal_loop.max_steps_per_cycle.max(1);
    let step_timeout = Duration::from_secs(config.goal_loop.step_timeout_secs.max(10));
    let autonomous_timeout = Duration::from_secs(config.goal_loop.autonomous_timeout_secs.max(30));
    let max_total_iterations = config.goal_loop.max_total_goal_iterations;

    let idle_interval = Duration::from_secs(u64::from(interval_mins) * 60);
    let active_interval = Duration::from_secs(5);
    let mut next_delay = Duration::ZERO; // first tick is immediate

    // Intent scan state
    let mem = crate::memory::SqliteMemory::new(&config.workspace_dir).ok(); // None if DB fails — intent scan will be skipped
    let mut last_intent_scan_ts: Option<String> = None;

    // Idle exploration state
    let mut last_exploration: Option<tokio::time::Instant> = None;
    let mut daily_exploration_count: u32 = 0;
    let mut last_exploration_date: Option<chrono::NaiveDate> = None;

    loop {
        tokio::time::sleep(next_delay).await;

        let mut state = match engine.load_state().await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("Goal loop: failed to load state: {e}");
                continue;
            }
        };

        // ── Auto-approve: promote pending low-priority goals ─────────
        if config.goal_loop.auto_approve_low_priority {
            let mut promoted = false;
            let mode = if config.goal_loop.default_execution_mode == "stepped" {
                GoalExecutionMode::Stepped
            } else {
                GoalExecutionMode::Autonomous
            };
            for goal in &mut state.goals {
                if goal.status == GoalStatus::Pending
                    && goal.priority == GoalPriority::Low
                    && !goal.steps.is_empty()
                {
                    goal.status = GoalStatus::InProgress;
                    goal.execution_mode = mode.clone();
                    goal.updated_at = chrono::Utc::now().to_rfc3339();
                    tracing::info!(
                        goal_id = %goal.id,
                        goal = %goal.description,
                        mode = ?mode,
                        "Goal loop: auto-approved low-priority goal"
                    );
                    promoted = true;
                }
            }
            if promoted {
                if let Err(e) = engine.save_state(&state).await {
                    tracing::warn!("Goal loop: failed to save state after auto-approve: {e}");
                }
                notify_goal_event(&config, "已自动批准待处理的低优先级目标").await;
            }
        }

        // ── Idle: no actionable goals — run intent scan, then explore ──
        if state.goals.is_empty() || !has_actionable_goals(&state) {
            if !config.goal_loop.explore_when_idle {
                continue;
            }

            // Network pre-check: skip if offline
            if !check_network_reachable(&config).await {
                tracing::debug!("Idle: network unreachable, skipping");
                continue;
            }

            // ── Intent scan (every cycle, no cooldown/cap) ──────────────
            if let Some(ref mem) = mem {
                'intent_scan: {
                    use crate::memory::MemoryCategory;

                    let entries = match mem
                        .recent_by_category(&MemoryCategory::Conversation, interval_mins)
                        .await
                    {
                        Ok(e) => e,
                        Err(e) => {
                            tracing::warn!("Intent scan: failed to query recent messages: {e}");
                            break 'intent_scan;
                        }
                    };

                    // Only user messages from Feishu, excluding already-scanned
                    let messages: Vec<_> = entries
                        .into_iter()
                        .filter(|e| e.key.starts_with("feishu_"))
                        .filter(|e| {
                            if let Some(ref ts) = last_intent_scan_ts {
                                e.timestamp.as_str() > ts.as_str()
                            } else {
                                true
                            }
                        })
                        .collect();

                    if messages.is_empty() {
                        tracing::debug!("Intent scan: no new user messages, skipping");
                        break 'intent_scan;
                    }

                    // Update watermark to newest message timestamp
                    if let Some(newest) =
                        messages.iter().max_by(|a, b| a.timestamp.cmp(&b.timestamp))
                    {
                        last_intent_scan_ts = Some(newest.timestamp.clone());
                    }

                    tracing::info!(
                        count = messages.len(),
                        "Goal loop: running intent scan on recent messages"
                    );
                    crate::health::set_component_activity("goal-loop", Some("intent-scan"));

                    let prompt = GoalEngine::build_intent_scan_prompt(&messages, &state);
                    let temp = config.default_temperature;
                    let started_at = chrono::Utc::now();
                    let result = tokio::time::timeout(
                        step_timeout,
                        crate::agent::run(
                            config.clone(),
                            Some(prompt),
                            None,
                            None,
                            temp,
                            vec![],
                            false,
                            false,
                            Some("intent-scan"),
                        ),
                    )
                    .await;
                    let finished_at = chrono::Utc::now();
                    crate::health::set_component_activity("goal-loop", None);

                    let duration_ms = (finished_at - started_at).num_milliseconds();
                    match &result {
                        Ok(Ok(output)) => {
                            tracing::info!("Intent scan completed");
                            crate::health::mark_component_ok("goal-loop");
                            let _ = crate::cron::record_run(
                                &config,
                                "__intent_scan",
                                started_at,
                                finished_at,
                                "ok",
                                Some(output),
                                duration_ms,
                            );
                        }
                        Ok(Err(e)) => {
                            tracing::warn!("Intent scan failed: {e}");
                            let _ = crate::cron::record_run(
                                &config,
                                "__intent_scan",
                                started_at,
                                finished_at,
                                "error",
                                Some(&e.to_string()),
                                duration_ms,
                            );
                        }
                        Err(_) => {
                            tracing::warn!("Intent scan timed out");
                            let _ = crate::cron::record_run(
                                &config,
                                "__intent_scan",
                                started_at,
                                finished_at,
                                "timeout",
                                None,
                                duration_ms,
                            );
                        }
                    }

                    // Reload state — agent may have created new goals
                    if let Ok(new_state) = engine.load_state().await {
                        state = new_state;
                    }
                }
            }

            // If intent scan created actionable goals, skip exploration
            // and fall through to goal execution
            if has_actionable_goals(&state) {
                // fall through — the execution code below will handle it
            } else {
                // ── Idle exploration (subject to cooldown/cap) ──────────
                let today = chrono::Utc::now().date_naive();
                if last_exploration_date != Some(today) {
                    daily_exploration_count = 0;
                    last_exploration_date = Some(today);
                }

                if daily_exploration_count >= config.goal_loop.max_explorations_per_day {
                    tracing::debug!(
                        "Idle exploration: daily cap reached ({daily_exploration_count})"
                    );
                    continue;
                }

                let cooldown = Duration::from_secs(
                    u64::from(config.goal_loop.explore_cooldown_minutes.max(1)) * 60,
                );
                if let Some(last) = last_exploration {
                    if last.elapsed() < cooldown {
                        continue;
                    }
                }

                tracing::info!("Goal loop: no active goals, triggering idle exploration");
                crate::health::set_component_activity("goal-loop", Some("exploring"));
                let prompt = GoalEngine::build_exploration_prompt(&state);
                let temp = config.default_temperature;

                let started_at = chrono::Utc::now();
                let result = tokio::time::timeout(
                    step_timeout,
                    crate::agent::run(
                        config.clone(),
                        Some(prompt),
                        None,
                        None,
                        temp,
                        vec![],
                        false,
                        false,
                        Some("idle-exploration"),
                    ),
                )
                .await;
                let finished_at = chrono::Utc::now();

                last_exploration = Some(tokio::time::Instant::now());
                daily_exploration_count += 1;

                crate::health::set_component_activity("goal-loop", None);
                match result {
                    Ok(Ok(output)) => {
                        tracing::info!("Idle exploration completed");
                        crate::health::mark_component_ok("goal-loop");
                        let duration_ms = (finished_at - started_at).num_milliseconds();
                        let _ = crate::cron::record_run(
                            &config,
                            "__idle_exploration",
                            started_at,
                            finished_at,
                            "ok",
                            Some(&output),
                            duration_ms,
                        );
                        notify_goal_event(
                            &config,
                            &format!(
                                "空闲探索完成\n\n{}",
                                crate::util::truncate_with_ellipsis(
                                    &clean_for_display(&output),
                                    300,
                                ),
                            ),
                        )
                        .await;
                    }
                    Ok(Err(e)) => {
                        let duration_ms = (finished_at - started_at).num_milliseconds();
                        let _ = crate::cron::record_run(
                            &config,
                            "__idle_exploration",
                            started_at,
                            finished_at,
                            "error",
                            Some(&e.to_string()),
                            duration_ms,
                        );
                        tracing::warn!("Idle exploration failed: {e}");
                    }
                    Err(_) => {
                        let duration_ms = (finished_at - started_at).num_milliseconds();
                        let _ = crate::cron::record_run(
                            &config,
                            "__idle_exploration",
                            started_at,
                            finished_at,
                            "timeout",
                            None,
                            duration_ms,
                        );
                        tracing::warn!("Idle exploration timed out");
                    }
                }

                continue;
            }
        }

        // ── Reflection: detect stalled goals before executing steps ──
        let stalled = GoalEngine::find_stalled_goals(&state);
        for gi in stalled {
            tracing::info!(
                goal = %state.goals[gi].description,
                "Goal loop: goal stalled, triggering reflection"
            );
            crate::health::set_component_activity(
                "goal-loop",
                Some(&format!(
                    "reflecting: {}",
                    crate::util::truncate_with_ellipsis(&state.goals[gi].description, 60)
                )),
            );
            let prompt = GoalEngine::build_reflection_prompt(&state.goals[gi]);
            let temp = config.default_temperature;

            let result = tokio::time::timeout(
                step_timeout,
                crate::agent::run(
                    config.clone(),
                    Some(prompt),
                    None,
                    None,
                    temp,
                    vec![],
                    false,
                    true,
                    Some("goal-loop"),
                ),
            )
            .await;

            crate::health::set_component_activity("goal-loop", None);
            match result {
                Ok(Ok(output)) => {
                    tracing::info!(
                        goal = %state.goals[gi].description,
                        "Goal reflection completed"
                    );
                    // Reload state — the agent may have modified goals.json
                    state = match engine.load_state().await {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::warn!(
                                "Goal loop: failed to reload state after reflection: {e}"
                            );
                            break;
                        }
                    };
                    notify_goal_event(
                        &config,
                        &format!(
                            "目标反思: {}\n\n{}",
                            state
                                .goals
                                .get(gi)
                                .map(|g| g.description.as_str())
                                .unwrap_or("?"),
                            crate::util::truncate_with_ellipsis(&clean_for_display(&output), 300,),
                        ),
                    )
                    .await;
                }
                Ok(Err(e)) => {
                    tracing::warn!("Goal reflection failed: {e}");
                }
                Err(_) => {
                    tracing::warn!("Goal reflection timed out");
                }
            }
        }

        // ── Autonomous goal execution ──────────────────────────────
        let mut autonomous_executed: u32 = 0;
        while autonomous_executed < max_steps {
            let Some(gi) = GoalEngine::select_next_autonomous_goal(&state) else {
                break;
            };

            let goal_id = state.goals[gi].id.clone();

            // Budget check: auto-block if total_iterations exceeded
            if state.goals[gi].total_iterations >= max_total_iterations {
                state.goals[gi].status = GoalStatus::Blocked;
                state.goals[gi].last_error = Some(format!(
                    "total iteration budget exhausted ({} >= {max_total_iterations})",
                    state.goals[gi].total_iterations,
                ));
                state.goals[gi].updated_at = chrono::Utc::now().to_rfc3339();
                let _ = engine.save_state(&state).await;
                notify_goal_event(
                    &config,
                    &format!(
                        "目标已暂停: '{}' 迭代次数已用尽",
                        state.goals[gi].description,
                    ),
                )
                .await;
                break;
            }

            tracing::info!(
                goal_id = %goal_id,
                goal = %state.goals[gi].description,
                iteration = state.goals[gi].total_iterations,
                "Goal loop: running autonomous session"
            );

            crate::health::set_component_activity(
                "goal-loop",
                Some(&format!(
                    "autonomous: {}",
                    crate::util::truncate_with_ellipsis(&state.goals[gi].description, 60)
                )),
            );

            let prompt = GoalEngine::build_autonomous_prompt(&state.goals[gi], &state);
            let temp = config.default_temperature;
            let started_at = chrono::Utc::now();

            let result = tokio::time::timeout(
                autonomous_timeout,
                crate::agent::run(
                    config.clone(),
                    Some(prompt),
                    None,
                    None,
                    temp,
                    vec![],
                    false,
                    true, // enable_state_reconciliation
                    Some("goal-loop"),
                ),
            )
            .await;

            let finished_at = chrono::Utc::now();
            let duration_ms = (finished_at - started_at).num_milliseconds();
            crate::health::set_component_activity("goal-loop", None);

            // Record in cron_runs so consolidation can see it
            let job_id = format!("__goal_{goal_id}");

            // Reload state — the agent may have modified goals.json directly
            state = match engine.load_state().await {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(
                        "Goal loop: failed to reload state after autonomous session: {e}"
                    );
                    break;
                }
            };

            // Re-locate goal by id (index may have shifted)
            let Some(gi) = state.goals.iter().position(|g| g.id == goal_id) else {
                tracing::warn!("Goal loop: goal {goal_id} disappeared after autonomous session");
                break;
            };

            // Approximate iteration cost: use agent max_tool_iterations as upper bound
            let iter_cost = config.agent.max_tool_iterations as u32;

            // Check if the agent already changed the goal status via state
            // reconciliation (directly editing goals.json). If so, handle
            // notification immediately regardless of how the session ended
            // (success, error, or timeout).
            let agent_reconciled_status = state.goals[gi].status.clone();
            if agent_reconciled_status == GoalStatus::Completed
                || agent_reconciled_status == GoalStatus::Blocked
            {
                state.goals[gi].total_iterations += iter_cost;
                state.goals[gi].updated_at = chrono::Utc::now().to_rfc3339();
                let _ = engine.save_state(&state).await;

                let (cron_status, cron_output) = match &result {
                    Ok(Ok(output)) => {
                        ("ok", Some(crate::util::truncate_with_ellipsis(output, 500)))
                    }
                    Ok(Err(e)) => ("error", Some(e.to_string())),
                    Err(_) => ("timeout", None),
                };
                let _ = crate::cron::record_run(
                    &config,
                    &job_id,
                    started_at,
                    finished_at,
                    cron_status,
                    cron_output.as_deref(),
                    duration_ms,
                );

                if agent_reconciled_status == GoalStatus::Completed {
                    crate::health::mark_component_ok("goal-loop");
                    let msg = format_goal_completed_message(&state.goals[gi]);
                    notify_goal_event(&config, &msg).await;
                } else {
                    let reason = state.goals[gi]
                        .last_error
                        .as_deref()
                        .unwrap_or("agent 标记为受阻");
                    notify_goal_event(
                        &config,
                        &format!(
                            "目标受阻: '{}'\n原因: {}",
                            state.goals[gi].description,
                            clean_for_display(reason),
                        ),
                    )
                    .await;
                }
                // Goal is done, move on to next goal
                autonomous_executed += 1;
                continue;
            }

            match result {
                Ok(Ok(output)) => {
                    let status = GoalEngine::interpret_autonomous_result(&output);
                    let wm = GoalEngine::extract_working_memory(&output, 2000);
                    state.goals[gi].working_memory = Some(wm);
                    state.goals[gi].total_iterations += iter_cost;
                    state.goals[gi].updated_at = chrono::Utc::now().to_rfc3339();

                    let _ = crate::cron::record_run(
                        &config,
                        &job_id,
                        started_at,
                        finished_at,
                        "ok",
                        Some(&crate::util::truncate_with_ellipsis(&output, 500)),
                        duration_ms,
                    );

                    match status {
                        crate::goals::engine::AutonomousSessionStatus::Completed => {
                            state.goals[gi].status = GoalStatus::Completed;
                            state.goals[gi].last_error = None;
                            let _ = engine.save_state(&state).await;
                            crate::health::mark_component_ok("goal-loop");
                            let msg = format_goal_completed_message(&state.goals[gi]);
                            notify_goal_event(&config, &msg).await;
                        }
                        crate::goals::engine::AutonomousSessionStatus::InProgress => {
                            state.goals[gi].last_error = None;
                            let _ = engine.save_state(&state).await;
                            crate::health::mark_component_ok("goal-loop");
                            tracing::info!(
                                goal_id = %goal_id,
                                "Autonomous session: goal still in progress"
                            );
                        }
                        crate::goals::engine::AutonomousSessionStatus::Blocked(reason) => {
                            state.goals[gi].status = GoalStatus::Blocked;
                            state.goals[gi].last_error = Some(reason.clone());
                            let _ = engine.save_state(&state).await;
                            notify_goal_event(
                                &config,
                                &format!(
                                    "目标受阻: '{}'\n原因: {}",
                                    state.goals[gi].description,
                                    clean_for_display(&reason),
                                ),
                            )
                            .await;
                        }
                    }
                }
                Ok(Err(e)) => {
                    let _ = crate::cron::record_run(
                        &config,
                        &job_id,
                        started_at,
                        finished_at,
                        "error",
                        Some(&e.to_string()),
                        duration_ms,
                    );
                    state.goals[gi].working_memory = Some(format!("[session error] {e}"));
                    state.goals[gi].total_iterations += iter_cost;
                    let _ = engine.save_state(&state).await;
                    tracing::warn!("Autonomous session error for goal {goal_id}: {e}");
                    crate::health::mark_component_error("goal-loop", e.to_string());
                    break;
                }
                Err(_elapsed) => {
                    let _ = crate::cron::record_run(
                        &config,
                        &job_id,
                        started_at,
                        finished_at,
                        "timeout",
                        None,
                        duration_ms,
                    );
                    state.goals[gi].working_memory =
                        Some("[session timed out — will resume next cycle]".into());
                    state.goals[gi].total_iterations += iter_cost;
                    let _ = engine.save_state(&state).await;
                    tracing::warn!("Autonomous session timed out for goal {goal_id}");
                    crate::health::mark_component_error(
                        "goal-loop",
                        "autonomous session timed out",
                    );
                    break;
                }
            }

            autonomous_executed += 1;
        }

        // ── Normal step execution (Stepped goals) ─────────────────────
        for _ in 0..max_steps {
            let Some((gi, si)) = GoalEngine::select_next_actionable(&state) else {
                break;
            };

            // Mark step in_progress
            state.goals[gi].steps[si].status = StepStatus::InProgress;
            let _ = engine.save_state(&state).await;

            crate::health::set_component_activity(
                "goal-loop",
                Some(&format!(
                    "executing: {}",
                    crate::util::truncate_with_ellipsis(&state.goals[gi].steps[si].description, 60)
                )),
            );

            let prompt =
                GoalEngine::build_step_prompt(&state.goals[gi], &state.goals[gi].steps[si]);
            let temp = config.default_temperature;

            let result = tokio::time::timeout(
                step_timeout,
                crate::agent::run(
                    config.clone(),
                    Some(prompt),
                    None,
                    None,
                    temp,
                    vec![],
                    false,
                    true,
                    Some("goal-loop"),
                ),
            )
            .await;

            crate::health::set_component_activity("goal-loop", None);
            match result {
                Ok(Ok(output)) => {
                    let success = GoalEngine::interpret_result(&output);
                    if success {
                        state.goals[gi].steps[si].status = StepStatus::Completed;
                        state.goals[gi].steps[si].result =
                            Some(crate::util::truncate_with_ellipsis(&output, 500));
                        // Append to goal context
                        let step_desc = state.goals[gi].steps[si].description.clone();
                        let summary = crate::util::truncate_with_ellipsis(&output, 200);
                        use std::fmt::Write as _;
                        let _ = write!(state.goals[gi].context, "\n- {step_desc}: {summary}");
                        state.goals[gi].last_error = None;

                        crate::health::mark_component_ok("goal-loop");
                    } else {
                        state.goals[gi].steps[si].status = StepStatus::Pending;
                        state.goals[gi].steps[si].attempts += 1;
                        state.goals[gi].last_error =
                            Some(crate::util::truncate_with_ellipsis(&output, 200));

                        if state.goals[gi].steps[si].attempts >= GoalEngine::max_step_attempts() {
                            state.goals[gi].status = GoalStatus::Blocked;
                            state.goals[gi].last_error = Some(format!(
                                "Step '{}' failed {} times",
                                state.goals[gi].steps[si].description,
                                state.goals[gi].steps[si].attempts,
                            ));
                            notify_goal_event(
                                &config,
                                &format!(
                                    "目标受阻: '{}'\n步骤 '{}' 已失败 {} 次",
                                    state.goals[gi].description,
                                    state.goals[gi].steps[si].description,
                                    state.goals[gi].steps[si].attempts,
                                ),
                            )
                            .await;
                            let _ = engine.save_state(&state).await;
                            break;
                        }
                    }
                }
                Ok(Err(e)) => {
                    state.goals[gi].steps[si].status = StepStatus::Pending;
                    state.goals[gi].steps[si].attempts += 1;
                    state.goals[gi].last_error = Some(e.to_string());
                    tracing::warn!(
                        "Goal loop step error (attempt {}): {e}",
                        state.goals[gi].steps[si].attempts
                    );
                    crate::health::mark_component_error("goal-loop", e.to_string());
                    let _ = engine.save_state(&state).await;
                    break;
                }
                Err(_elapsed) => {
                    state.goals[gi].steps[si].status = StepStatus::Pending;
                    state.goals[gi].steps[si].attempts += 1;
                    state.goals[gi].last_error = Some("step execution timed out".into());
                    tracing::warn!(
                        "Goal loop step timed out (attempt {})",
                        state.goals[gi].steps[si].attempts
                    );
                    crate::health::mark_component_error("goal-loop", "step timed out");
                    let _ = engine.save_state(&state).await;
                    break;
                }
            }

            // Check if all steps done → goal completed
            let all_done = state.goals[gi]
                .steps
                .iter()
                .all(|s| s.status == StepStatus::Completed);
            if all_done {
                state.goals[gi].status = GoalStatus::Completed;
                state.goals[gi].updated_at = chrono::Utc::now().to_rfc3339();
                let msg = format_goal_completed_message(&state.goals[gi]);
                notify_goal_event(&config, &msg).await;
                let _ = engine.save_state(&state).await;
                break;
            }

            let _ = engine.save_state(&state).await;
        }

        // Dynamic interval: short delay if goals are actionable, long delay if idle
        next_delay = if has_actionable_goals(&state) {
            active_interval
        } else {
            idle_interval
        };
    }
}

/// Build a completion notification that includes working_memory as a summary.
fn format_goal_completed_message(goal: &crate::goals::engine::Goal) -> String {
    let mut msg = format!("目标已完成: {}", goal.description);
    if let Some(ref wm) = goal.working_memory {
        let cleaned = clean_for_display(wm);
        if !cleaned.is_empty() {
            let summary = crate::util::truncate_with_ellipsis(&cleaned, 800);
            msg.push_str("\n\n");
            msg.push_str(&summary);
        }
    }
    msg
}

/// Strip `[GOAL_STATUS: ...]` markers and other internal protocol text
/// from agent output before showing to end users.
fn clean_for_display(text: &str) -> String {
    let re = regex::Regex::new(r"(?s)\[GOAL_STATUS:[^\]]*\]").unwrap();
    let cleaned = re.replace_all(text, "");
    let collapse = regex::Regex::new(r"\n{3,}").unwrap();
    collapse.replace_all(&cleaned, "\n\n").trim().to_string()
}

/// Send a goal event notification via the configured channel (best-effort).
async fn notify_goal_event(config: &Config, message: &str) {
    if let (Some(ch_name), Some(target)) = (&config.goal_loop.channel, &config.goal_loop.target) {
        if let Err(e) = deliver_heartbeat(config, ch_name, target, message).await {
            tracing::warn!("Goal event delivery to {ch_name} failed: {e}");
        }
    } else {
        tracing::info!("Goal event (no delivery channel): {message}");
    }
}

fn has_supervised_channels(config: &Config) -> bool {
    config
        .channels_config
        .channels_except_webhook()
        .iter()
        .any(|(_, ok)| *ok)
}

/// Ensure the nightly consolidation job exists in the cron store.
/// Called once on daemon startup when cron is enabled.  If the job
/// already exists this is a no-op.
fn ensure_consolidation_job(config: &Config) {
    use crate::cron::consolidation::CONSOLIDATION_JOB_NAME;

    match crate::cron::list_jobs(config) {
        Ok(jobs) => {
            if jobs
                .iter()
                .any(|j| j.name.as_deref() == Some(CONSOLIDATION_JOB_NAME))
            {
                return;
            }
        }
        Err(e) => {
            tracing::warn!("Failed to check for consolidation job: {e}");
            return;
        }
    }

    match crate::cron::consolidation::create_consolidation_job(config) {
        Ok(job) => {
            tracing::info!(
                job_id = %job.id,
                "Auto-registered nightly consolidation job"
            );
        }
        Err(e) => {
            tracing::warn!("Failed to create nightly consolidation job: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_config(tmp: &TempDir) -> Config {
        let config = Config {
            workspace_dir: tmp.path().join("workspace"),
            config_path: tmp.path().join("config.toml"),
            ..Config::default()
        };
        std::fs::create_dir_all(&config.workspace_dir).unwrap();
        config
    }

    #[test]
    fn state_file_path_uses_config_directory() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp);

        let path = state_file_path(&config);
        assert_eq!(path, tmp.path().join("daemon_state.json"));
    }

    #[tokio::test]
    async fn supervisor_marks_error_and_restart_on_failure() {
        let handle = spawn_component_supervisor("daemon-test-fail", 1, 1, || async {
            anyhow::bail!("boom")
        });

        tokio::time::sleep(Duration::from_millis(50)).await;
        handle.abort();
        let _ = handle.await;

        let snapshot = crate::health::snapshot_json();
        let component = &snapshot["components"]["daemon-test-fail"];
        assert_eq!(component["status"], "error");
        assert!(component["restart_count"].as_u64().unwrap_or(0) >= 1);
        assert!(component["last_error"]
            .as_str()
            .unwrap_or("")
            .contains("boom"));
    }

    #[tokio::test]
    async fn supervisor_marks_unexpected_exit_as_error() {
        let handle = spawn_component_supervisor("daemon-test-exit", 1, 1, || async { Ok(()) });

        tokio::time::sleep(Duration::from_millis(50)).await;
        handle.abort();
        let _ = handle.await;

        let snapshot = crate::health::snapshot_json();
        let component = &snapshot["components"]["daemon-test-exit"];
        assert_eq!(component["status"], "error");
        assert!(component["restart_count"].as_u64().unwrap_or(0) >= 1);
        assert!(component["last_error"]
            .as_str()
            .unwrap_or("")
            .contains("component exited unexpectedly"));
    }

    #[test]
    fn detects_no_supervised_channels() {
        let config = Config::default();
        assert!(!has_supervised_channels(&config));
    }

    #[test]
    fn detects_supervised_channels_present() {
        let mut config = Config::default();
        config.channels_config.telegram = Some(crate::config::TelegramConfig {
            bot_token: "token".into(),
            allowed_users: vec![],
            stream_mode: crate::config::StreamMode::default(),
            draft_update_interval_ms: 1000,
            interrupt_on_new_message: false,
            mention_only: false,
        });
        assert!(has_supervised_channels(&config));
    }

    #[test]
    fn detects_dingtalk_as_supervised_channel() {
        let mut config = Config::default();
        config.channels_config.dingtalk = Some(crate::config::schema::DingTalkConfig {
            client_id: "client_id".into(),
            client_secret: "client_secret".into(),
            allowed_users: vec!["*".into()],
        });
        assert!(has_supervised_channels(&config));
    }

    #[test]
    fn detects_mattermost_as_supervised_channel() {
        let mut config = Config::default();
        config.channels_config.mattermost = Some(crate::config::schema::MattermostConfig {
            url: "https://mattermost.example.com".into(),
            bot_token: "token".into(),
            channel_id: Some("channel-id".into()),
            allowed_users: vec!["*".into()],
            thread_replies: Some(true),
            mention_only: Some(false),
        });
        assert!(has_supervised_channels(&config));
    }

    #[test]
    fn detects_qq_as_supervised_channel() {
        let mut config = Config::default();
        config.channels_config.qq = Some(crate::config::schema::QQConfig {
            app_id: "app-id".into(),
            app_secret: "app-secret".into(),
            allowed_users: vec!["*".into()],
        });
        assert!(has_supervised_channels(&config));
    }

    #[test]
    fn detects_nextcloud_talk_as_supervised_channel() {
        let mut config = Config::default();
        config.channels_config.nextcloud_talk = Some(crate::config::schema::NextcloudTalkConfig {
            base_url: "https://cloud.example.com".into(),
            app_token: "app-token".into(),
            webhook_secret: None,
            allowed_users: vec!["*".into()],
        });
        assert!(has_supervised_channels(&config));
    }

    #[tokio::test]
    async fn deliver_heartbeat_unsupported_channel_returns_error() {
        let config = Config::default();
        let err = deliver_heartbeat(&config, "carrier_pigeon", "target", "hello")
            .await
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("unsupported heartbeat delivery channel"));
    }

    #[tokio::test]
    async fn deliver_heartbeat_lark_not_configured_returns_error() {
        let config = Config::default();
        let err = deliver_heartbeat(&config, "lark", "oc_abc123", "report")
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("lark/feishu channel not configured") || msg.contains("channel-lark"),
            "unexpected error: {msg}"
        );
    }

    #[tokio::test]
    async fn deliver_heartbeat_feishu_not_configured_returns_error() {
        let config = Config::default();
        let err = deliver_heartbeat(&config, "feishu", "oc_abc123", "report")
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("lark/feishu channel not configured") || msg.contains("channel-lark"),
            "unexpected error: {msg}"
        );
    }

    #[tokio::test]
    async fn deliver_heartbeat_telegram_not_configured_returns_error() {
        let config = Config::default();
        let err = deliver_heartbeat(&config, "telegram", "12345", "report")
            .await
            .unwrap_err();
        assert!(err.to_string().contains("telegram channel not configured"));
    }

    #[tokio::test]
    async fn deliver_heartbeat_case_insensitive_channel_name() {
        let config = Config::default();
        // "LARK" should match as "lark" (case insensitive), then fail because not configured
        let err = deliver_heartbeat(&config, "LARK", "oc_abc123", "report")
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("lark/feishu channel not configured") || msg.contains("channel-lark"),
            "unexpected error: {msg}"
        );
    }
}
