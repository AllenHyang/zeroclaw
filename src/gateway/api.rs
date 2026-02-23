//! REST API handlers for the web dashboard.
//!
//! All `/api/*` routes require bearer token authentication (PairingGuard).

use super::AppState;
use axum::{
    extract::{Path, Query, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Json},
};
use serde::Deserialize;

// ── Bearer token auth extractor ─────────────────────────────────

/// Extract and validate bearer token from Authorization header.
fn extract_bearer_token(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|auth| auth.strip_prefix("Bearer "))
}

/// Verify bearer token against PairingGuard. Returns error response if unauthorized.
fn require_auth(
    state: &AppState,
    headers: &HeaderMap,
) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    if !state.pairing.require_pairing() {
        return Ok(());
    }

    let token = extract_bearer_token(headers).unwrap_or("");
    if state.pairing.is_authenticated(token) {
        Ok(())
    } else {
        Err((
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "error": "Unauthorized — pair first via POST /pair, then send Authorization: Bearer <token>"
            })),
        ))
    }
}

// ── Query parameters ─────────────────────────────────────────────

#[derive(Deserialize)]
pub struct MemoryQuery {
    pub query: Option<String>,
    pub category: Option<String>,
}

#[derive(Deserialize)]
pub struct MemoryStoreBody {
    pub key: String,
    pub content: String,
    pub category: Option<String>,
}

#[derive(Deserialize)]
pub struct CronAddBody {
    pub name: Option<String>,
    pub schedule: String,
    pub command: String,
}

#[derive(Deserialize)]
pub struct GoalCreateBody {
    pub description: String,
    #[serde(default)]
    pub priority: Option<String>,
    #[serde(default)]
    pub execution_mode: Option<String>,
    #[serde(default)]
    pub steps: Option<Vec<GoalStepBody>>,
    #[serde(default)]
    pub success_criteria: Option<String>,
    #[serde(default)]
    pub constraints: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub context: Option<String>,
}

#[derive(Deserialize)]
pub struct GoalStepBody {
    pub description: String,
}

#[derive(Deserialize)]
pub struct GoalUpdateBody {
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub priority: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub last_error: Option<String>,
}

// ── Handlers ────────────────────────────────────────────────────

/// GET /api/status — system status overview
pub async fn handle_api_status(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let config = state.config.lock().clone();
    let health = crate::health::snapshot();

    let mut channels = serde_json::Map::new();

    for (channel, present) in config.channels_config.channels() {
        channels.insert(channel.name().to_string(), serde_json::Value::Bool(present));
    }

    let body = serde_json::json!({
        "provider": config.default_provider,
        "model": state.model,
        "temperature": state.temperature,
        "uptime_seconds": health.uptime_seconds,
        "gateway_port": config.gateway.port,
        "locale": "en",
        "memory_backend": state.mem.name(),
        "paired": state.pairing.is_paired(),
        "channels": channels,
        "health": health,
        "goal_loop": crate::goals::status::snapshot_json(),
    });

    Json(body).into_response()
}

/// GET /api/config — current config (api_key masked)
pub async fn handle_api_config_get(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let config = state.config.lock().clone();

    // Serialize to TOML, then mask sensitive fields
    let toml_str = match toml::to_string_pretty(&config) {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("Failed to serialize config: {e}")})),
            )
                .into_response();
        }
    };

    // Mask api_key in the TOML output
    let masked = mask_sensitive_fields(&toml_str);

    Json(serde_json::json!({
        "format": "toml",
        "content": masked,
    }))
    .into_response()
}

/// PUT /api/config — update config from TOML body
pub async fn handle_api_config_put(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: String,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    // Parse the incoming TOML
    let new_config: crate::config::Config = match toml::from_str(&body) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("Invalid TOML: {e}")})),
            )
                .into_response();
        }
    };

    // Save to disk
    if let Err(e) = new_config.save().await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("Failed to save config: {e}")})),
        )
            .into_response();
    }

    // Update in-memory config
    *state.config.lock() = new_config;

    Json(serde_json::json!({"status": "ok"})).into_response()
}

/// GET /api/tools — list registered tool specs
pub async fn handle_api_tools(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let tools: Vec<serde_json::Value> = state
        .tools_registry
        .iter()
        .map(|spec| {
            serde_json::json!({
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            })
        })
        .collect();

    Json(serde_json::json!({"tools": tools})).into_response()
}

/// GET /api/cron — list cron jobs
pub async fn handle_api_cron_list(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let config = state.config.lock().clone();
    match crate::cron::list_jobs(&config) {
        Ok(jobs) => {
            let jobs_json: Vec<serde_json::Value> = jobs
                .iter()
                .map(|job| {
                    serde_json::json!({
                        "id": job.id,
                        "name": job.name,
                        "command": job.command,
                        "next_run": job.next_run.to_rfc3339(),
                        "last_run": job.last_run.map(|t| t.to_rfc3339()),
                        "last_status": job.last_status,
                        "enabled": job.enabled,
                    })
                })
                .collect();
            Json(serde_json::json!({"jobs": jobs_json})).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("Failed to list cron jobs: {e}")})),
        )
            .into_response(),
    }
}

/// POST /api/cron — add a new cron job
pub async fn handle_api_cron_add(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<CronAddBody>,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let config = state.config.lock().clone();
    let schedule = crate::cron::Schedule::Cron {
        expr: body.schedule,
        tz: None,
    };

    match crate::cron::add_shell_job(&config, body.name, schedule, &body.command) {
        Ok(job) => Json(serde_json::json!({
            "status": "ok",
            "job": {
                "id": job.id,
                "name": job.name,
                "command": job.command,
                "enabled": job.enabled,
            }
        }))
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("Failed to add cron job: {e}")})),
        )
            .into_response(),
    }
}

/// DELETE /api/cron/:id — remove a cron job
pub async fn handle_api_cron_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let config = state.config.lock().clone();
    match crate::cron::remove_job(&config, &id) {
        Ok(()) => Json(serde_json::json!({"status": "ok"})).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("Failed to remove cron job: {e}")})),
        )
            .into_response(),
    }
}

/// GET /api/integrations — list all integrations with status
pub async fn handle_api_integrations(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let config = state.config.lock().clone();
    let entries = crate::integrations::registry::all_integrations();

    let integrations: Vec<serde_json::Value> = entries
        .iter()
        .map(|entry| {
            let status = (entry.status_fn)(&config);
            serde_json::json!({
                "name": entry.name,
                "description": entry.description,
                "category": entry.category,
                "status": status,
            })
        })
        .collect();

    Json(serde_json::json!({"integrations": integrations})).into_response()
}

/// POST /api/doctor — run diagnostics
pub async fn handle_api_doctor(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let config = state.config.lock().clone();
    let results = crate::doctor::diagnose(&config);

    let ok_count = results
        .iter()
        .filter(|r| r.severity == crate::doctor::Severity::Ok)
        .count();
    let warn_count = results
        .iter()
        .filter(|r| r.severity == crate::doctor::Severity::Warn)
        .count();
    let error_count = results
        .iter()
        .filter(|r| r.severity == crate::doctor::Severity::Error)
        .count();

    Json(serde_json::json!({
        "results": results,
        "summary": {
            "ok": ok_count,
            "warnings": warn_count,
            "errors": error_count,
        }
    }))
    .into_response()
}

/// GET /api/memory — list or search memory entries
pub async fn handle_api_memory_list(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<MemoryQuery>,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    if let Some(ref query) = params.query {
        // Search mode
        match state.mem.recall(query, 50, None).await {
            Ok(entries) => Json(serde_json::json!({"entries": entries})).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("Memory recall failed: {e}")})),
            )
                .into_response(),
        }
    } else {
        // List mode
        let category = params.category.as_deref().map(|cat| match cat {
            "core" => crate::memory::MemoryCategory::Core,
            "daily" => crate::memory::MemoryCategory::Daily,
            "conversation" => crate::memory::MemoryCategory::Conversation,
            other => crate::memory::MemoryCategory::Custom(other.to_string()),
        });

        match state.mem.list(category.as_ref(), None).await {
            Ok(entries) => Json(serde_json::json!({"entries": entries})).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("Memory list failed: {e}")})),
            )
                .into_response(),
        }
    }
}

/// POST /api/memory — store a memory entry
pub async fn handle_api_memory_store(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<MemoryStoreBody>,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let category = body
        .category
        .as_deref()
        .map(|cat| match cat {
            "core" => crate::memory::MemoryCategory::Core,
            "daily" => crate::memory::MemoryCategory::Daily,
            "conversation" => crate::memory::MemoryCategory::Conversation,
            other => crate::memory::MemoryCategory::Custom(other.to_string()),
        })
        .unwrap_or(crate::memory::MemoryCategory::Core);

    match state
        .mem
        .store(&body.key, &body.content, category, None)
        .await
    {
        Ok(()) => Json(serde_json::json!({"status": "ok"})).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("Memory store failed: {e}")})),
        )
            .into_response(),
    }
}

/// DELETE /api/memory/:key — delete a memory entry
pub async fn handle_api_memory_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(key): Path<String>,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    match state.mem.forget(&key).await {
        Ok(deleted) => {
            Json(serde_json::json!({"status": "ok", "deleted": deleted})).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("Memory forget failed: {e}")})),
        )
            .into_response(),
    }
}

/// GET /api/cost — cost summary
pub async fn handle_api_cost(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    if let Some(ref tracker) = state.cost_tracker {
        match tracker.get_summary() {
            Ok(summary) => Json(serde_json::json!({"cost": summary})).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("Cost summary failed: {e}")})),
            )
                .into_response(),
        }
    } else {
        Json(serde_json::json!({
            "cost": {
                "session_cost_usd": 0.0,
                "daily_cost_usd": 0.0,
                "monthly_cost_usd": 0.0,
                "total_tokens": 0,
                "request_count": 0,
                "by_model": {},
            }
        }))
        .into_response()
    }
}

/// GET /api/cli-tools — discovered CLI tools
pub async fn handle_api_cli_tools(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let tools = crate::tools::cli_discovery::discover_cli_tools(&[], &[]);

    Json(serde_json::json!({"cli_tools": tools})).into_response()
}

/// GET /api/health — component health snapshot
pub async fn handle_api_health(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let snapshot = crate::health::snapshot();
    Json(serde_json::json!({"health": snapshot})).into_response()
}

/// GET /api/goals — current goal-loop state (goals, steps, status)
pub async fn handle_api_goals(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let config = state.config.lock().clone();
    let engine = crate::goals::engine::GoalEngine::new(&config.workspace_dir);
    match engine.load_state().await {
        Ok(goal_state) => Json(serde_json::json!({"goals": goal_state.goals})).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("Failed to load goals: {e}")})),
        )
            .into_response(),
    }
}

/// Request body for POST /api/goals/{id}/confirm
#[derive(Deserialize)]
pub struct GoalConfirmBody {
    pub approved: bool,
    pub feedback: Option<String>,
}

/// POST /api/goals/{id}/confirm — approve or reject a goal's understanding plan
pub async fn handle_api_goal_confirm(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(body): Json<GoalConfirmBody>,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let config = state.config.lock().clone();
    let engine = crate::goals::engine::GoalEngine::new(&config.workspace_dir);
    let mut goal_state = match engine.load_state().await {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("Failed to load goals: {e}")})),
            )
                .into_response();
        }
    };

    let Some(goal) = goal_state.goals.iter_mut().find(|g| g.id == id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("Goal '{id}' not found")})),
        )
            .into_response();
    };

    if goal.status != crate::goals::engine::GoalStatus::AwaitingConfirmation {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": format!(
                    "Goal '{}' is not awaiting confirmation (current status: {:?})",
                    id, goal.status
                )
            })),
        )
            .into_response();
    }

    if body.approved {
        goal.status = crate::goals::engine::GoalStatus::InProgress;
        goal.updated_at = chrono::Utc::now().to_rfc3339();
    } else {
        // Reject: clear plan, store feedback for re-generation
        goal.confirmation_plan = None;
        goal.confirmation_requested_at = None;
        goal.confirmation_feedback = body.feedback.clone();
        goal.updated_at = chrono::Utc::now().to_rfc3339();
    }

    if let Err(e) = engine.save_state(&goal_state).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("Failed to save state: {e}")})),
        )
            .into_response();
    }

    let action = if body.approved {
        "approved"
    } else {
        "rejected"
    };
    Json(serde_json::json!({
        "status": "ok",
        "goal_id": id,
        "action": action,
    }))
    .into_response()
}

/// POST /api/goals — create a new goal
pub async fn handle_api_goals_create(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<GoalCreateBody>,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let description = body.description.trim().to_string();
    if description.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "description must not be empty"})),
        )
            .into_response();
    }

    let config = state.config.lock().clone();
    let engine = crate::goals::engine::GoalEngine::new(&config.workspace_dir);

    let mut goal_state = match engine.load_state().await {
        Ok(gs) => gs,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("Failed to load goals: {e}")})),
            )
                .into_response();
        }
    };

    let id = next_goal_id(&goal_state.goals);
    let now = chrono::Utc::now().to_rfc3339();

    let steps: Vec<crate::goals::engine::Step> = body
        .steps
        .unwrap_or_default()
        .into_iter()
        .enumerate()
        .map(|(i, s)| crate::goals::engine::Step {
            id: format!("{}-s{}", id, i + 1),
            description: s.description,
            status: crate::goals::engine::StepStatus::Pending,
            result: None,
            attempts: 0,
        })
        .collect();

    let goal = crate::goals::engine::Goal {
        id: id.clone(),
        description: description.clone(),
        status: body
            .status
            .as_deref()
            .map(parse_status)
            .unwrap_or(crate::goals::engine::GoalStatus::Pending),
        priority: body
            .priority
            .as_deref()
            .map(parse_priority)
            .unwrap_or_default(),
        created_at: now.clone(),
        updated_at: now,
        steps,
        context: body.context.unwrap_or_default(),
        last_error: None,
        success_criteria: body.success_criteria,
        constraints: body.constraints,
        working_memory: None,
        execution_mode: body
            .execution_mode
            .as_deref()
            .map(parse_execution_mode)
            .unwrap_or_default(),
        total_iterations: 0,
    };

    goal_state.goals.push(goal.clone());

    if let Err(e) = engine.save_state(&goal_state).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("Failed to save goals: {e}")})),
        )
            .into_response();
    }

    let _ = state.event_tx.send(serde_json::json!({
        "type": "goal_created",
        "goal_id": id,
        "description": description,
    }));

    (StatusCode::CREATED, Json(serde_json::json!({"goal": goal}))).into_response()
}

/// PATCH /api/goals/{id} — update an existing goal
pub async fn handle_api_goals_update(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(body): Json<GoalUpdateBody>,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let config = state.config.lock().clone();
    let engine = crate::goals::engine::GoalEngine::new(&config.workspace_dir);

    let mut goal_state = match engine.load_state().await {
        Ok(gs) => gs,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("Failed to load goals: {e}")})),
            )
                .into_response();
        }
    };

    let goal = match goal_state.goals.iter_mut().find(|g| g.id == id) {
        Some(g) => g,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": format!("Goal {id} not found")})),
            )
                .into_response();
        }
    };

    if let Some(ref s) = body.status {
        goal.status = parse_status(s);
    }
    if let Some(ref p) = body.priority {
        goal.priority = parse_priority(p);
    }
    if let Some(ref d) = body.description {
        goal.description = d.clone();
    }
    if let Some(ref e) = body.last_error {
        goal.last_error = Some(e.clone());
    }
    goal.updated_at = chrono::Utc::now().to_rfc3339();

    let updated_goal = goal.clone();

    if let Err(e) = engine.save_state(&goal_state).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("Failed to save goals: {e}")})),
        )
            .into_response();
    }

    let _ = state.event_tx.send(serde_json::json!({
        "type": "goal_updated",
        "goal_id": id,
    }));

    Json(serde_json::json!({"goal": updated_goal})).into_response()
}

// ── Goal helpers ────────────────────────────────────────────────

fn next_goal_id(goals: &[crate::goals::engine::Goal]) -> String {
    let max_num = goals
        .iter()
        .filter_map(|g| {
            g.id.strip_prefix("g-")
                .or_else(|| g.id.strip_prefix('g'))
                .and_then(|n| n.parse::<u32>().ok())
        })
        .max()
        .unwrap_or(0);
    format!("g-{:03}", max_num + 1)
}

fn parse_priority(s: &str) -> crate::goals::engine::GoalPriority {
    match s {
        "low" => crate::goals::engine::GoalPriority::Low,
        "high" => crate::goals::engine::GoalPriority::High,
        "critical" => crate::goals::engine::GoalPriority::Critical,
        _ => crate::goals::engine::GoalPriority::Medium,
    }
}

fn parse_status(s: &str) -> crate::goals::engine::GoalStatus {
    match s {
        "in_progress" => crate::goals::engine::GoalStatus::InProgress,
        "completed" => crate::goals::engine::GoalStatus::Completed,
        "blocked" => crate::goals::engine::GoalStatus::Blocked,
        "cancelled" => crate::goals::engine::GoalStatus::Cancelled,
        _ => crate::goals::engine::GoalStatus::Pending,
    }
}

fn parse_execution_mode(s: &str) -> crate::goals::engine::GoalExecutionMode {
    match s {
        "stepped" => crate::goals::engine::GoalExecutionMode::Stepped,
        _ => crate::goals::engine::GoalExecutionMode::Autonomous,
    }
}

/// GET /api/dashboard — aggregated dashboard view
pub async fn handle_api_dashboard(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let config = state.config.lock().clone();

    // Health + uptime
    let health = crate::health::snapshot();
    let uptime_seconds = health.uptime_seconds;

    // Goals summary
    let engine = crate::goals::engine::GoalEngine::new(&config.workspace_dir);
    let goals_summary = match engine.load_state().await {
        Ok(gs) => {
            use crate::goals::engine::GoalStatus;
            let mut completed = 0u32;
            let mut in_progress = 0u32;
            let mut pending = 0u32;
            let mut awaiting_confirmation = 0u32;
            let mut blocked = 0u32;
            let mut cancelled = 0u32;
            for g in &gs.goals {
                match g.status {
                    GoalStatus::Completed => completed += 1,
                    GoalStatus::InProgress => in_progress += 1,
                    GoalStatus::Pending => pending += 1,
                    GoalStatus::AwaitingConfirmation => awaiting_confirmation += 1,
                    GoalStatus::Blocked => blocked += 1,
                    GoalStatus::Cancelled => cancelled += 1,
                }
            }
            serde_json::json!({
                "total": gs.goals.len(),
                "completed": completed,
                "in_progress": in_progress,
                "awaiting_confirmation": awaiting_confirmation,
                "pending": pending,
                "blocked": blocked,
                "cancelled": cancelled,
            })
        }
        Err(_) => serde_json::json!({"total": 0}),
    };

    // Cost
    let cost = if let Some(ref tracker) = state.cost_tracker {
        match tracker.get_summary() {
            Ok(s) => serde_json::json!({
                "session_cost_usd": s.session_cost_usd,
                "daily_cost_usd": s.daily_cost_usd,
                "monthly_cost_usd": s.monthly_cost_usd,
                "total_tokens": s.total_tokens,
                "request_count": s.request_count,
            }),
            Err(_) => serde_json::json!({}),
        }
    } else {
        serde_json::json!({})
    };

    // Memory count
    let memory_count = state.mem.count().await.unwrap_or(0);

    // Tools + capabilities
    let tools_count = state.tools_registry.len();
    let capabilities = derive_capabilities(&state.tools_registry);

    // Milestones (last 10)
    use crate::memory::MemoryCategory;
    let milestones: Vec<serde_json::Value> = state
        .mem
        .list(Some(&MemoryCategory::Custom("milestone".into())), None)
        .await
        .unwrap_or_default()
        .into_iter()
        .rev()
        .take(10)
        .map(|e| {
            serde_json::json!({
                "key": e.key,
                "content": e.content,
                "timestamp": e.timestamp,
            })
        })
        .collect();

    let body = serde_json::json!({
        "uptime_seconds": uptime_seconds,
        "goals": goals_summary,
        "goal_loop": crate::goals::status::snapshot_json(),
        "cost": cost,
        "memory": { "total_entries": memory_count },
        "tools": { "registered": tools_count },
        "capabilities": capabilities,
        "milestones": milestones,
        "health": health,
    });

    Json(body).into_response()
}

/// Derive high-level capability labels from registered tool names.
fn derive_capabilities(tools: &[crate::tools::traits::ToolSpec]) -> Vec<&'static str> {
    let names: std::collections::HashSet<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    let mut caps = Vec::new();

    if names.contains("shell") {
        caps.push("shell");
    }
    if names.contains("file_read") || names.contains("file_write") || names.contains("file_edit") {
        caps.push("file_system");
    }
    if names.contains("memory_store") || names.contains("memory_recall") {
        caps.push("memory");
    }
    if names.contains("browser") || names.contains("browser_open") {
        caps.push("web_browsing");
    }
    if names.contains("web_search") || names.contains("http_request") {
        caps.push("web");
    }
    if names.contains("git_operations") {
        caps.push("git");
    }
    if names.contains("cron_add") || names.contains("cron_list") {
        caps.push("cron");
    }
    if names.contains("delegate") {
        caps.push("delegation");
    }
    if names.contains("screenshot") || names.contains("image_info") {
        caps.push("vision");
    }
    if names.contains("pdf_read") {
        caps.push("pdf");
    }

    caps
}

// ── Helpers ─────────────────────────────────────────────────────

fn mask_sensitive_fields(toml_str: &str) -> String {
    let mut output = String::with_capacity(toml_str.len());
    for line in toml_str.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("api_key")
            || trimmed.starts_with("bot_token")
            || trimmed.starts_with("access_token")
            || trimmed.starts_with("secret")
            || trimmed.starts_with("app_secret")
            || trimmed.starts_with("signing_secret")
        {
            if let Some(eq_pos) = line.find('=') {
                output.push_str(&line[..eq_pos + 1]);
                output.push_str(" \"***MASKED***\"");
            } else {
                output.push_str(line);
            }
        } else {
            output.push_str(line);
        }
        output.push('\n');
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_goal(id: &str) -> crate::goals::engine::Goal {
        crate::goals::engine::Goal {
            id: id.to_string(),
            description: String::new(),
            status: crate::goals::engine::GoalStatus::Pending,
            priority: crate::goals::engine::GoalPriority::Medium,
            created_at: String::new(),
            updated_at: String::new(),
            steps: vec![],
            context: String::new(),
            last_error: None,
            success_criteria: None,
            constraints: None,
            working_memory: None,
            execution_mode: crate::goals::engine::GoalExecutionMode::Autonomous,
            total_iterations: 0,
        }
    }

    #[test]
    fn next_goal_id_empty() {
        assert_eq!(next_goal_id(&[]), "g-001");
    }

    #[test]
    fn next_goal_id_sequential() {
        let goals = vec![make_goal("g-001"), make_goal("g-002"), make_goal("g-003")];
        assert_eq!(next_goal_id(&goals), "g-004");
    }

    #[test]
    fn next_goal_id_mixed_formats() {
        let goals = vec![make_goal("g1"), make_goal("g-005"), make_goal("g-002")];
        assert_eq!(next_goal_id(&goals), "g-006");
    }
}
