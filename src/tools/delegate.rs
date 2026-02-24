use super::traits::{Tool, ToolResult};
use crate::agent::loop_::run_tool_call_loop;
use crate::config::DelegateAgentConfig;
use crate::observability::traits::{Observer, ObserverEvent, ObserverMetric};
use crate::providers::{self, ChatMessage, Provider};
use crate::security::policy::ToolOperation;
use crate::security::SecurityPolicy;
use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// Default timeout for sub-agent provider calls.
const DELEGATE_TIMEOUT_SECS: u64 = 120;
/// Default timeout for agentic sub-agent runs.
const DELEGATE_AGENTIC_TIMEOUT_SECS: u64 = 300;

/// Tool that delegates a subtask to a named agent with a different
/// provider/model configuration. Enables multi-agent workflows where
/// a primary agent can hand off specialized work (research, coding,
/// summarization) to purpose-built sub-agents.
pub struct DelegateTool {
    agents: Arc<HashMap<String, DelegateAgentConfig>>,
    security: Arc<SecurityPolicy>,
    /// Global credential fallback (from config.api_key)
    fallback_credential: Option<String>,
    /// Provider runtime options inherited from root config.
    provider_runtime_options: providers::ProviderRuntimeOptions,
    /// Depth at which this tool instance lives in the delegation chain.
    depth: u32,
    /// Parent tool registry for agentic sub-agents.
    parent_tools: Arc<Vec<Arc<dyn Tool>>>,
    /// Inherited multimodal handling config for sub-agent loops.
    multimodal_config: crate::config::MultimodalConfig,
}

impl DelegateTool {
    pub fn new(
        agents: HashMap<String, DelegateAgentConfig>,
        fallback_credential: Option<String>,
        security: Arc<SecurityPolicy>,
    ) -> Self {
        Self::new_with_options(
            agents,
            fallback_credential,
            security,
            providers::ProviderRuntimeOptions::default(),
        )
    }

    pub fn new_with_options(
        agents: HashMap<String, DelegateAgentConfig>,
        fallback_credential: Option<String>,
        security: Arc<SecurityPolicy>,
        provider_runtime_options: providers::ProviderRuntimeOptions,
    ) -> Self {
        Self {
            agents: Arc::new(agents),
            security,
            fallback_credential,
            provider_runtime_options,
            depth: 0,
            parent_tools: Arc::new(Vec::new()),
            multimodal_config: crate::config::MultimodalConfig::default(),
        }
    }

    /// Create a DelegateTool for a sub-agent (with incremented depth).
    /// When sub-agents eventually get their own tool registry, construct
    /// their DelegateTool via this method with `depth: parent.depth + 1`.
    pub fn with_depth(
        agents: HashMap<String, DelegateAgentConfig>,
        fallback_credential: Option<String>,
        security: Arc<SecurityPolicy>,
        depth: u32,
    ) -> Self {
        Self::with_depth_and_options(
            agents,
            fallback_credential,
            security,
            depth,
            providers::ProviderRuntimeOptions::default(),
        )
    }

    pub fn with_depth_and_options(
        agents: HashMap<String, DelegateAgentConfig>,
        fallback_credential: Option<String>,
        security: Arc<SecurityPolicy>,
        depth: u32,
        provider_runtime_options: providers::ProviderRuntimeOptions,
    ) -> Self {
        Self {
            agents: Arc::new(agents),
            security,
            fallback_credential,
            provider_runtime_options,
            depth,
            parent_tools: Arc::new(Vec::new()),
            multimodal_config: crate::config::MultimodalConfig::default(),
        }
    }

    /// Attach parent tools used to build sub-agent allowlist registries.
    pub fn with_parent_tools(mut self, parent_tools: Arc<Vec<Arc<dyn Tool>>>) -> Self {
        self.parent_tools = parent_tools;
        self
    }

    /// Attach multimodal configuration for sub-agent tool loops.
    pub fn with_multimodal_config(mut self, config: crate::config::MultimodalConfig) -> Self {
        self.multimodal_config = config;
        self
    }
}

#[async_trait]
impl Tool for DelegateTool {
    fn name(&self) -> &str {
        "delegate"
    }

    fn description(&self) -> &str {
        "Delegate a subtask to a specialized agent. Use when: a task benefits from a different model \
         (e.g. fast summarization, deep reasoning, code generation). The sub-agent runs a single \
         prompt by default; with agentic=true it can iterate with a filtered tool-call loop."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        let agent_names: Vec<&str> = self.agents.keys().map(|s: &String| s.as_str()).collect();
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "agent": {
                    "type": "string",
                    "minLength": 1,
                    "description": format!(
                        "Name of the agent to delegate to. Available: {}",
                        if agent_names.is_empty() {
                            "(none configured)".to_string()
                        } else {
                            agent_names.join(", ")
                        }
                    )
                },
                "prompt": {
                    "type": "string",
                    "minLength": 1,
                    "description": "The task/prompt to send to the sub-agent"
                },
                "context": {
                    "type": "string",
                    "description": "Optional context to prepend (e.g. relevant code, prior findings)"
                }
            },
            "required": ["agent", "prompt"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let agent_name = args
            .get("agent")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .ok_or_else(|| anyhow::anyhow!("Missing 'agent' parameter"))?;

        if agent_name.is_empty() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("'agent' parameter must not be empty".into()),
                error_kind: None,
            });
        }

        let prompt = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .ok_or_else(|| anyhow::anyhow!("Missing 'prompt' parameter"))?;

        if prompt.is_empty() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("'prompt' parameter must not be empty".into()),
                error_kind: None,
            });
        }

        // Look up agent config
        let agent_config = match self.agents.get(agent_name) {
            Some(cfg) => cfg,
            None => {
                let available: Vec<&str> =
                    self.agents.keys().map(|s: &String| s.as_str()).collect();
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!(
                        "Unknown agent '{agent_name}'. Available agents: {}",
                        if available.is_empty() {
                            "(none configured)".to_string()
                        } else {
                            available.join(", ")
                        }
                    )),
                    error_kind: None,
                });
            }
        };

        // Check recursion depth (immutable — set at construction, incremented for sub-agents)
        if self.depth >= agent_config.max_depth {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Delegation depth limit reached ({depth}/{max}). \
                     Cannot delegate further to prevent infinite loops.",
                    depth = self.depth,
                    max = agent_config.max_depth
                )),
                error_kind: None,
            });
        }

        if let Err(error) = self
            .security
            .enforce_tool_operation(ToolOperation::Act, "delegate")
        {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(error),
                error_kind: None,
            });
        }

        // Build the full prompt early — needed for both local and remote paths.
        let full_prompt = {
            let context = args
                .get("context")
                .and_then(|v| v.as_str())
                .map(str::trim)
                .unwrap_or("");
            if context.is_empty() {
                prompt.to_string()
            } else {
                format!("[Context]\n{context}\n\n[Task]\n{prompt}")
            }
        };

        // Remote delegation: forward to a peer daemon via HTTP.
        if let Some(ref remote_ws) = agent_config.remote {
            return self
                .execute_remote(agent_name, remote_ws, &full_prompt)
                .await;
        }

        // Create provider for this agent
        let provider_credential_owned = agent_config
            .api_key
            .clone()
            .or_else(|| self.fallback_credential.clone());
        #[allow(clippy::option_as_ref_deref)]
        let provider_credential = provider_credential_owned.as_ref().map(String::as_str);

        let provider: Box<dyn Provider> = match providers::create_provider_with_options(
            &agent_config.provider,
            provider_credential,
            &self.provider_runtime_options,
        ) {
            Ok(p) => p,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!(
                        "Failed to create provider '{}' for agent '{agent_name}': {e}",
                        agent_config.provider
                    )),
                    error_kind: None,
                });
            }
        };

        let temperature = agent_config.temperature.unwrap_or(0.7);

        // Agentic mode: run full tool-call loop with allowlisted tools.
        if agent_config.agentic {
            return self
                .execute_agentic(
                    agent_name,
                    agent_config,
                    &*provider,
                    &full_prompt,
                    temperature,
                )
                .await;
        }

        // Wrap the provider call in a timeout to prevent indefinite blocking
        let result = tokio::time::timeout(
            Duration::from_secs(DELEGATE_TIMEOUT_SECS),
            provider.chat_with_system(
                agent_config.system_prompt.as_deref(),
                &full_prompt,
                &agent_config.model,
                temperature,
            ),
        )
        .await;

        let result = match result {
            Ok(inner) => inner,
            Err(_elapsed) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!(
                        "Agent '{agent_name}' timed out after {DELEGATE_TIMEOUT_SECS}s"
                    )),
                    error_kind: None,
                });
            }
        };

        match result {
            Ok(response) => {
                let mut rendered = response;
                if rendered.trim().is_empty() {
                    rendered = "[Empty response]".to_string();
                }

                Ok(ToolResult {
                    success: true,
                    output: format!(
                        "[Agent '{agent_name}' ({provider}/{model})]\n{rendered}",
                        provider = agent_config.provider,
                        model = agent_config.model
                    ),
                    error: None,
                    error_kind: None,
                })
            }
            Err(e) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Agent '{agent_name}' failed: {e}",)),
                error_kind: None,
            }),
        }
    }
}

impl DelegateTool {
    async fn execute_agentic(
        &self,
        agent_name: &str,
        agent_config: &DelegateAgentConfig,
        provider: &dyn Provider,
        full_prompt: &str,
        temperature: f64,
    ) -> anyhow::Result<ToolResult> {
        if agent_config.allowed_tools.is_empty() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Agent '{agent_name}' has agentic=true but allowed_tools is empty"
                )),
                error_kind: None,
            });
        }

        let allowed = agent_config
            .allowed_tools
            .iter()
            .map(|name| name.trim())
            .filter(|name| !name.is_empty())
            .collect::<std::collections::HashSet<_>>();

        let sub_tools: Vec<Box<dyn Tool>> = self
            .parent_tools
            .iter()
            .filter(|tool| allowed.contains(tool.name()))
            .filter(|tool| tool.name() != "delegate")
            .map(|tool| Box::new(ToolArcRef::new(tool.clone())) as Box<dyn Tool>)
            .collect();

        if sub_tools.is_empty() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Agent '{agent_name}' has no executable tools after filtering allowlist ({})",
                    agent_config.allowed_tools.join(", ")
                )),
                error_kind: None,
            });
        }

        let mut history = Vec::new();
        if let Some(system_prompt) = agent_config.system_prompt.as_ref() {
            history.push(ChatMessage::system(system_prompt.clone()));
        }
        history.push(ChatMessage::user(full_prompt.to_string()));

        let noop_observer = NoopObserver;

        let result = tokio::time::timeout(
            Duration::from_secs(DELEGATE_AGENTIC_TIMEOUT_SECS),
            run_tool_call_loop(
                provider,
                &mut history,
                &sub_tools,
                &noop_observer,
                &agent_config.provider,
                &agent_config.model,
                temperature,
                true,
                None,
                "delegate",
                &self.multimodal_config,
                agent_config.max_iterations,
                None,
                None,
                None,
                &[],
                None,
                None, // sub-agent tool calls are not double-audited
            ),
        )
        .await;

        match result {
            Ok(Ok(response)) => {
                let rendered = if response.trim().is_empty() {
                    "[Empty response]".to_string()
                } else {
                    response
                };

                Ok(ToolResult {
                    success: true,
                    output: format!(
                        "[Agent '{agent_name}' ({provider}/{model}, agentic)]\n{rendered}",
                        provider = agent_config.provider,
                        model = agent_config.model
                    ),
                    error: None,
                    error_kind: None,
                })
            }
            Ok(Err(e)) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Agent '{agent_name}' failed: {e}")),
                error_kind: None,
            }),
            Err(_) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Agent '{agent_name}' timed out after {DELEGATE_AGENTIC_TIMEOUT_SECS}s"
                )),
                error_kind: None,
            }),
        }
    }

    /// Delegate to a peer daemon via HTTP webhook.
    async fn execute_remote(
        &self,
        agent_name: &str,
        remote_ws: &str,
        full_prompt: &str,
    ) -> anyhow::Result<ToolResult> {
        use crate::workspace::{discover_workspaces, RunStatus};

        // Hard cap on remote delegation depth to prevent infinite HTTP loops
        // between peers that mutually configure remote delegation.
        const MAX_REMOTE_DELEGATION_DEPTH: u32 = 5;
        if self.depth >= MAX_REMOTE_DELEGATION_DEPTH {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Remote delegation depth limit reached ({}/{}). Refusing to delegate further.",
                    self.depth, MAX_REMOTE_DELEGATION_DEPTH
                )),
                error_kind: None,
            });
        }

        let self_config_dir = self
            .provider_runtime_options
            .zeroclaw_dir
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("Cannot resolve config dir for remote delegation"))?;

        let workspaces = discover_workspaces(Some(self_config_dir))
            .map_err(|e| anyhow::anyhow!("Failed to discover workspaces: {e}"))?;

        let target = workspaces
            .iter()
            .find(|w| w.name == remote_ws)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Remote workspace '{remote_ws}' not found. Available: {}",
                    workspaces
                        .iter()
                        .map(|w| w.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?;

        if !matches!(target.running, RunStatus::Running { .. }) {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Remote workspace '{remote_ws}' is not running. Start it with: zeroclaw workspace start {remote_ws}"
                )),
                error_kind: None,
            });
        }

        // Read peer token for authentication
        let token_path = self_config_dir
            .join(".peer_tokens")
            .join(format!("{remote_ws}.token"));
        let token = std::fs::read_to_string(&token_path).map_err(|e| {
            anyhow::anyhow!(
                "No peer token for '{remote_ws}' at {}: {e}. \
                 Re-clone or manually exchange tokens.",
                token_path.display()
            )
        })?;
        let token = token.trim().to_string();

        let is_loopback = matches!(
            target.host.as_str(),
            "127.0.0.1" | "localhost" | "::1"
        );

        // Read TLS config from the target workspace to decide scheme.
        let target_tls_cert = std::fs::read_to_string(target.config_dir.join("config.toml"))
            .ok()
            .and_then(|s| toml::from_str::<toml::Value>(&s).ok())
            .and_then(|v| {
                v.get("gateway")?
                    .get("tls_cert")?
                    .as_str()
                    .map(String::from)
            });

        let scheme = if !is_loopback && target_tls_cert.is_some() {
            "https"
        } else {
            "http"
        };

        let url = format!("{scheme}://{}:{}/webhook", target.host, target.port);

        // Warn if delegating over non-loopback without TLS.
        if !is_loopback && scheme == "http" {
            tracing::warn!(
                "Remote delegation to '{}' uses non-loopback host {} over plaintext HTTP — bearer token is unencrypted",
                remote_ws, target.host
            );
        }

        let client = build_peer_client(
            self_config_dir,
            remote_ws,
            DELEGATE_TIMEOUT_SECS,
        );

        let resp = match client
            .post(&url)
            .header("Authorization", format!("Bearer {token}"))
            .header("Content-Type", "application/json")
            .header("X-Delegation-Depth", (self.depth + 1).to_string())
            .json(&serde_json::json!({ "message": full_prompt }))
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                let msg = if e.is_timeout() {
                    format!("Remote delegation to '{remote_ws}' timed out after {DELEGATE_TIMEOUT_SECS}s")
                } else {
                    format!("Remote delegation to '{remote_ws}' failed: {e}")
                };
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(msg),
                    error_kind: None,
                });
            }
        };

        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();

        if status.is_success() {
            // Try to extract "response" field from JSON, fallback to raw body
            let output = serde_json::from_str::<serde_json::Value>(&body)
                .ok()
                .and_then(|v| v.get("response").and_then(|r| r.as_str()).map(String::from))
                .unwrap_or(body);

            Ok(ToolResult {
                success: true,
                output: format!(
                    "[Agent '{agent_name}' via remote '{remote_ws}' (depth={depth}, uses remote daemon config)]\n{output}",
                    depth = self.depth
                ),
                error: None,
                error_kind: None,
            })
        } else {
            Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Remote '{remote_ws}' returned HTTP {status}: {body}"
                )),
                error_kind: None,
            })
        }
    }
}

/// Build a reqwest client for peer delegation, optionally loading a custom
/// CA certificate if the peer has deposited one during workspace cloning.
///
/// Falls back to the standard runtime-proxy client when no peer CA exists.
fn build_peer_client(
    self_config_dir: &std::path::Path,
    peer_name: &str,
    timeout_secs: u64,
) -> reqwest::Client {
    let peer_ca_path = self_config_dir
        .join("tls")
        .join("peers")
        .join(format!("{peer_name}-ca.pem"));

    if let Ok(pem_bytes) = std::fs::read(&peer_ca_path) {
        if let Ok(cert) = reqwest::Certificate::from_pem(&pem_bytes) {
            if let Ok(client) = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(timeout_secs))
                .connect_timeout(std::time::Duration::from_secs(10))
                .add_root_certificate(cert)
                .build()
            {
                tracing::debug!(
                    "Using custom CA for peer '{}' from {}",
                    peer_name,
                    peer_ca_path.display()
                );
                return client;
            }
        }
        tracing::warn!(
            "Failed to parse peer CA certificate at {}; falling back to default client",
            peer_ca_path.display()
        );
    }

    crate::config::build_runtime_proxy_client_with_timeouts("delegate.remote", timeout_secs, 10)
}

struct ToolArcRef {
    inner: Arc<dyn Tool>,
}

impl ToolArcRef {
    fn new(inner: Arc<dyn Tool>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl Tool for ToolArcRef {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn parameters_schema(&self) -> serde_json::Value {
        self.inner.parameters_schema()
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        self.inner.execute(args).await
    }
}

struct NoopObserver;

impl Observer for NoopObserver {
    fn record_event(&self, _event: &ObserverEvent) {}

    fn record_metric(&self, _metric: &ObserverMetric) {}

    fn name(&self) -> &str {
        "noop"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::{ChatRequest, ChatResponse, ToolCall};
    use crate::security::{AutonomyLevel, SecurityPolicy};
    use anyhow::anyhow;

    fn test_security() -> Arc<SecurityPolicy> {
        Arc::new(SecurityPolicy::default())
    }

    fn sample_agents() -> HashMap<String, DelegateAgentConfig> {
        let mut agents = HashMap::new();
        agents.insert(
            "researcher".to_string(),
            DelegateAgentConfig {
                provider: "ollama".to_string(),
                model: "llama3".to_string(),
                system_prompt: Some("You are a research assistant.".to_string()),
                api_key: None,
                temperature: Some(0.3),
                max_depth: 3,
                agentic: false,
                allowed_tools: Vec::new(),
                max_iterations: 10,
                remote: None,
            },
        );
        agents.insert(
            "coder".to_string(),
            DelegateAgentConfig {
                provider: "openrouter".to_string(),
                model: "anthropic/claude-sonnet-4-20250514".to_string(),
                system_prompt: None,
                api_key: Some("delegate-test-credential".to_string()),
                temperature: None,
                max_depth: 2,
                agentic: false,
                allowed_tools: Vec::new(),
                max_iterations: 10,
                remote: None,
            },
        );
        agents
    }

    #[derive(Default)]
    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo_tool"
        }

        fn description(&self) -> &str {
            "Echoes the `value` argument."
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "value": {"type": "string"}
                },
                "required": ["value"]
            })
        }

        async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
            let value = args
                .get("value")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string();
            Ok(ToolResult {
                success: true,
                output: format!("echo:{value}"),
                error: None,
                error_kind: None,
            })
        }
    }

    struct OneToolThenFinalProvider;

    #[async_trait]
    impl Provider for OneToolThenFinalProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            Ok("unused".to_string())
        }

        async fn chat(
            &self,
            request: ChatRequest<'_>,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<ChatResponse> {
            let has_tool_message = request.messages.iter().any(|m| m.role == "tool");
            if has_tool_message {
                Ok(ChatResponse {
                    text: Some("done".to_string()),
                    tool_calls: Vec::new(),
                    usage: None,
                    reasoning_content: None,
                })
            } else {
                Ok(ChatResponse {
                    text: None,
                    tool_calls: vec![ToolCall {
                        id: "call_1".to_string(),
                        name: "echo_tool".to_string(),
                        arguments: "{\"value\":\"ping\"}".to_string(),
                    }],
                    usage: None,
                    reasoning_content: None,
                })
            }
        }
    }

    struct InfiniteToolCallProvider;

    #[async_trait]
    impl Provider for InfiniteToolCallProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            Ok("unused".to_string())
        }

        async fn chat(
            &self,
            _request: ChatRequest<'_>,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<ChatResponse> {
            Ok(ChatResponse {
                text: None,
                tool_calls: vec![ToolCall {
                    id: "loop".to_string(),
                    name: "echo_tool".to_string(),
                    arguments: "{\"value\":\"x\"}".to_string(),
                }],
                usage: None,
                reasoning_content: None,
            })
        }
    }

    struct FailingProvider;

    #[async_trait]
    impl Provider for FailingProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            Ok("unused".to_string())
        }

        async fn chat(
            &self,
            _request: ChatRequest<'_>,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<ChatResponse> {
            Err(anyhow!("provider boom"))
        }
    }

    fn agentic_config(allowed_tools: Vec<String>, max_iterations: usize) -> DelegateAgentConfig {
        DelegateAgentConfig {
            provider: "openrouter".to_string(),
            model: "model-test".to_string(),
            system_prompt: Some("You are agentic.".to_string()),
            api_key: Some("delegate-test-credential".to_string()),
            temperature: Some(0.2),
            max_depth: 3,
            agentic: true,
            allowed_tools,
            max_iterations,
            remote: None,
        }
    }

    #[test]
    fn name_and_schema() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        assert_eq!(tool.name(), "delegate");
        let schema = tool.parameters_schema();
        assert!(schema["properties"]["agent"].is_object());
        assert!(schema["properties"]["prompt"].is_object());
        assert!(schema["properties"]["context"].is_object());
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("agent")));
        assert!(required.contains(&json!("prompt")));
        assert_eq!(schema["additionalProperties"], json!(false));
        assert_eq!(schema["properties"]["agent"]["minLength"], json!(1));
        assert_eq!(schema["properties"]["prompt"]["minLength"], json!(1));
    }

    #[test]
    fn description_not_empty() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        assert!(!tool.description().is_empty());
    }

    #[test]
    fn schema_lists_agent_names() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        let schema = tool.parameters_schema();
        let desc = schema["properties"]["agent"]["description"]
            .as_str()
            .unwrap();
        assert!(desc.contains("researcher") || desc.contains("coder"));
    }

    #[tokio::test]
    async fn missing_agent_param() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        let result = tool.execute(json!({"prompt": "test"})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn missing_prompt_param() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        let result = tool.execute(json!({"agent": "researcher"})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn unknown_agent_returns_error() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        let result = tool
            .execute(json!({"agent": "nonexistent", "prompt": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("Unknown agent"));
    }

    #[tokio::test]
    async fn depth_limit_enforced() {
        let tool = DelegateTool::with_depth(sample_agents(), None, test_security(), 3);
        let result = tool
            .execute(json!({"agent": "researcher", "prompt": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("depth limit"));
    }

    #[tokio::test]
    async fn depth_limit_per_agent() {
        // coder has max_depth=2, so depth=2 should be blocked
        let tool = DelegateTool::with_depth(sample_agents(), None, test_security(), 2);
        let result = tool
            .execute(json!({"agent": "coder", "prompt": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("depth limit"));
    }

    #[test]
    fn empty_agents_schema() {
        let tool = DelegateTool::new(HashMap::new(), None, test_security());
        let schema = tool.parameters_schema();
        let desc = schema["properties"]["agent"]["description"]
            .as_str()
            .unwrap();
        assert!(desc.contains("none configured"));
    }

    #[tokio::test]
    async fn invalid_provider_returns_error() {
        let mut agents = HashMap::new();
        agents.insert(
            "broken".to_string(),
            DelegateAgentConfig {
                provider: "totally-invalid-provider".to_string(),
                model: "model".to_string(),
                system_prompt: None,
                api_key: None,
                temperature: None,
                max_depth: 3,
                agentic: false,
                allowed_tools: Vec::new(),
                max_iterations: 10,
                remote: None,
            },
        );
        let tool = DelegateTool::new(agents, None, test_security());
        let result = tool
            .execute(json!({"agent": "broken", "prompt": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("Failed to create provider"));
    }

    #[tokio::test]
    async fn blank_agent_rejected() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        let result = tool
            .execute(json!({"agent": "  ", "prompt": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("must not be empty"));
    }

    #[tokio::test]
    async fn blank_prompt_rejected() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        let result = tool
            .execute(json!({"agent": "researcher", "prompt": "  \t  "}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("must not be empty"));
    }

    #[tokio::test]
    async fn whitespace_agent_name_trimmed_and_found() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        // " researcher " with surrounding whitespace — after trim becomes "researcher"
        let result = tool
            .execute(json!({"agent": " researcher ", "prompt": "test"}))
            .await
            .unwrap();
        // Should find "researcher" after trim — will fail at provider level
        // since ollama isn't running, but must NOT get "Unknown agent".
        assert!(
            result.error.is_none()
                || !result
                    .error
                    .as_deref()
                    .unwrap_or("")
                    .contains("Unknown agent")
        );
    }

    #[tokio::test]
    async fn delegation_blocked_in_readonly_mode() {
        let readonly = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::ReadOnly,
            ..SecurityPolicy::default()
        });
        let tool = DelegateTool::new(sample_agents(), None, readonly);
        let result = tool
            .execute(json!({"agent": "researcher", "prompt": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("read-only mode"));
    }

    #[tokio::test]
    async fn delegation_blocked_when_rate_limited() {
        let limited = Arc::new(SecurityPolicy {
            max_actions_per_hour: 0,
            ..SecurityPolicy::default()
        });
        let tool = DelegateTool::new(sample_agents(), None, limited);
        let result = tool
            .execute(json!({"agent": "researcher", "prompt": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("Rate limit exceeded"));
    }

    #[tokio::test]
    async fn delegate_context_is_prepended_to_prompt() {
        let mut agents = HashMap::new();
        agents.insert(
            "tester".to_string(),
            DelegateAgentConfig {
                provider: "invalid-for-test".to_string(),
                model: "test-model".to_string(),
                system_prompt: None,
                api_key: None,
                temperature: None,
                max_depth: 3,
                agentic: false,
                allowed_tools: Vec::new(),
                max_iterations: 10,
                remote: None,
            },
        );
        let tool = DelegateTool::new(agents, None, test_security());
        let result = tool
            .execute(json!({
                "agent": "tester",
                "prompt": "do something",
                "context": "some context data"
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("Failed to create provider"));
    }

    #[tokio::test]
    async fn delegate_empty_context_omits_prefix() {
        let mut agents = HashMap::new();
        agents.insert(
            "tester".to_string(),
            DelegateAgentConfig {
                provider: "invalid-for-test".to_string(),
                model: "test-model".to_string(),
                system_prompt: None,
                api_key: None,
                temperature: None,
                max_depth: 3,
                agentic: false,
                allowed_tools: Vec::new(),
                max_iterations: 10,
                remote: None,
            },
        );
        let tool = DelegateTool::new(agents, None, test_security());
        let result = tool
            .execute(json!({
                "agent": "tester",
                "prompt": "do something",
                "context": ""
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("Failed to create provider"));
    }

    #[test]
    fn delegate_depth_construction() {
        let tool = DelegateTool::with_depth(sample_agents(), None, test_security(), 5);
        assert_eq!(tool.depth, 5);
    }

    #[tokio::test]
    async fn delegate_no_agents_configured() {
        let tool = DelegateTool::new(HashMap::new(), None, test_security());
        let result = tool
            .execute(json!({"agent": "any", "prompt": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("none configured"));
    }

    #[tokio::test]
    async fn agentic_mode_rejects_empty_allowed_tools() {
        let mut agents = HashMap::new();
        agents.insert("agentic".to_string(), agentic_config(Vec::new(), 10));

        let tool = DelegateTool::new(agents, None, test_security());
        let result = tool
            .execute(json!({"agent": "agentic", "prompt": "test"}))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("allowed_tools is empty"));
    }

    #[tokio::test]
    async fn agentic_mode_rejects_unmatched_allowed_tools() {
        let mut agents = HashMap::new();
        agents.insert(
            "agentic".to_string(),
            agentic_config(vec!["missing_tool".to_string()], 10),
        );

        let tool = DelegateTool::new(agents, None, test_security())
            .with_parent_tools(Arc::new(vec![Arc::new(EchoTool)]));
        let result = tool
            .execute(json!({"agent": "agentic", "prompt": "test"}))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("no executable tools"));
    }

    #[tokio::test]
    async fn execute_agentic_runs_tool_call_loop_with_filtered_tools() {
        let config = agentic_config(vec!["echo_tool".to_string()], 10);
        let tool = DelegateTool::new(HashMap::new(), None, test_security()).with_parent_tools(
            Arc::new(vec![
                Arc::new(EchoTool),
                Arc::new(DelegateTool::new(HashMap::new(), None, test_security())),
            ]),
        );

        let provider = OneToolThenFinalProvider;
        let result = tool
            .execute_agentic("agentic", &config, &provider, "run", 0.2)
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.output.contains("(openrouter/model-test, agentic)"));
        assert!(result.output.contains("done"));
    }

    #[tokio::test]
    async fn execute_agentic_excludes_delegate_even_if_allowlisted() {
        let config = agentic_config(vec!["delegate".to_string()], 10);
        let tool = DelegateTool::new(HashMap::new(), None, test_security()).with_parent_tools(
            Arc::new(vec![Arc::new(DelegateTool::new(
                HashMap::new(),
                None,
                test_security(),
            ))]),
        );

        let provider = OneToolThenFinalProvider;
        let result = tool
            .execute_agentic("agentic", &config, &provider, "run", 0.2)
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("no executable tools"));
    }

    #[tokio::test]
    async fn execute_agentic_respects_max_iterations() {
        let config = agentic_config(vec!["echo_tool".to_string()], 2);
        let tool = DelegateTool::new(HashMap::new(), None, test_security())
            .with_parent_tools(Arc::new(vec![Arc::new(EchoTool)]));

        let provider = InfiniteToolCallProvider;
        let result = tool
            .execute_agentic("agentic", &config, &provider, "run", 0.2)
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("maximum tool iterations (2)"));
    }

    #[tokio::test]
    async fn execute_agentic_propagates_provider_errors() {
        let config = agentic_config(vec!["echo_tool".to_string()], 10);
        let tool = DelegateTool::new(HashMap::new(), None, test_security())
            .with_parent_tools(Arc::new(vec![Arc::new(EchoTool)]));

        let provider = FailingProvider;
        let result = tool
            .execute_agentic("agentic", &config, &provider, "run", 0.2)
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("provider boom"));
    }
}
