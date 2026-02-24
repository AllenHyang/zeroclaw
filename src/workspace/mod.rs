use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

use crate::config::schema::{
    decrypt_optional_secret, decrypt_secret, default_config_dir,
    persist_active_workspace_config_dir, Config,
};
use crate::onboard::{scaffold_workspace, ProjectContext};
use crate::security::SecretStore;
use crate::WorkspaceCommands;

// ── Shared types ────────────────────────────────────────────────

/// Run-state of a workspace daemon.
#[derive(Debug)]
pub(crate) enum RunStatus {
    Running { pid: u32 },
    Stopped,
    /// PID file exists but process is dead.
    Stale,
}

/// Discovered workspace entry.
#[derive(Debug)]
pub(crate) struct WorkspaceInfo {
    pub name: String,
    pub config_dir: PathBuf,
    pub port: u16,
    pub host: String,
    pub is_active: bool,
    pub running: RunStatus,
}

// ── Dispatch ────────────────────────────────────────────────────

/// Dispatch workspace subcommands.
pub async fn handle_command(cmd: WorkspaceCommands, config: &Config) -> Result<()> {
    match cmd {
        WorkspaceCommands::Clone {
            name,
            port,
            switch,
        } => clone_workspace(config, &name, port, switch).await,
        WorkspaceCommands::List => list_workspaces(config),
        WorkspaceCommands::Switch { name } => switch_workspace(&name).await,
        WorkspaceCommands::Start { name } => start_workspace(name.as_deref(), config),
        WorkspaceCommands::Stop { name, all } => stop_workspace(name.as_deref(), all, config),
    }
}

// ── Name validation ─────────────────────────────────────────────

fn validate_workspace_name(name: &str) -> Result<()> {
    if name.is_empty() {
        bail!("Workspace name must not be empty");
    }
    if name == "default" {
        bail!("\"default\" is reserved for the primary workspace (~/.zeroclaw/)");
    }
    if name.starts_with('.') || name.starts_with('-') {
        bail!("Workspace name must not start with '.' or '-'");
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        bail!("Workspace name may only contain [a-zA-Z0-9_-]");
    }
    Ok(())
}

// ── Discovery infrastructure ────────────────────────────────────

/// Derive workspace name from directory name.
/// `.zeroclaw` → `"default"`, `.zeroclaw-X` → `"X"`.
pub(crate) fn workspace_name_from_dir(dir_name: &str) -> String {
    if dir_name == ".zeroclaw" {
        "default".to_string()
    } else {
        dir_name
            .strip_prefix(".zeroclaw-")
            .unwrap_or(dir_name)
            .to_string()
    }
}

/// Check whether a process with the given PID is alive.
#[cfg(unix)]
pub(crate) fn is_pid_alive(pid: u32) -> bool {
    // SAFETY: kill(pid, 0) is a standard POSIX existence check.
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}

#[cfg(not(unix))]
pub(crate) fn is_pid_alive(_pid: u32) -> bool {
    false
}

/// Read and parse `daemon.pid` inside a config directory.
pub(crate) fn read_pid_file(config_dir: &Path) -> Option<u32> {
    let pid_path = config_dir.join("daemon.pid");
    std::fs::read_to_string(pid_path)
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|&pid| pid > 0)
}

/// Check whether the daemon's flock on `daemon.pid` is currently held.
/// Returns `true` if the lock is held (daemon is running), `false` otherwise.
#[cfg(unix)]
fn is_daemon_lock_held(config_dir: &Path) -> bool {
    use std::os::unix::io::AsRawFd;
    let pid_path = config_dir.join("daemon.pid");
    let file = match std::fs::OpenOptions::new().read(true).open(&pid_path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let fd = file.as_raw_fd();
    // Try to acquire a non-blocking exclusive lock
    let ret = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
    if ret == 0 {
        // Successfully acquired = daemon is not holding the lock
        unsafe { libc::flock(fd, libc::LOCK_UN) };
        false
    } else {
        // Failed to acquire = daemon holds the lock
        true
    }
}

#[cfg(not(unix))]
fn is_daemon_lock_held(_config_dir: &Path) -> bool {
    false
}

/// Scan all `.zeroclaw*` directories under home and return workspace metadata.
pub(crate) fn discover_workspaces(active_config_dir: Option<&Path>) -> Result<Vec<WorkspaceInfo>> {
    let default_dir = default_config_dir()?;
    let home = default_dir
        .parent()
        .context("Home directory not found")?
        .to_path_buf();

    let mut workspaces = Vec::new();

    for entry in std::fs::read_dir(&home).context("Failed to read home directory")? {
        let entry = entry?;
        let dir_name = entry.file_name();
        let dir_name_str = dir_name.to_string_lossy().to_string();
        // Match exactly `.zeroclaw` or `.zeroclaw-{name}` (not `.zeroclaw-backup` etc.)
        if dir_name_str != ".zeroclaw" && !dir_name_str.starts_with(".zeroclaw-") {
            continue;
        }
        // Must contain a config.toml to be a valid workspace
        let dir_path = entry.path();
        let config_path = dir_path.join("config.toml");
        if !config_path.exists() {
            continue;
        }

        let ws_name = workspace_name_from_dir(&dir_name_str);

        // Read port and host from config
        let (port, host) = std::fs::read_to_string(&config_path)
            .ok()
            .and_then(|s| toml::from_str::<Config>(&s).ok())
            .map(|c| (c.gateway.port, c.gateway.host.clone()))
            .unwrap_or((0, "127.0.0.1".to_string()));

        let is_active = active_config_dir.is_some_and(|active| active == dir_path);

        let running = match read_pid_file(&dir_path) {
            Some(pid) if is_daemon_lock_held(&dir_path) && is_pid_alive(pid) => {
                RunStatus::Running { pid }
            }
            Some(_) => RunStatus::Stale,
            None => RunStatus::Stopped,
        };

        workspaces.push(WorkspaceInfo {
            name: ws_name,
            config_dir: dir_path,
            port,
            host,
            is_active,
            running,
        });
    }

    // Sort: default first, then alphabetically
    workspaces.sort_by(|a, b| {
        if a.name == "default" {
            std::cmp::Ordering::Less
        } else if b.name == "default" {
            std::cmp::Ordering::Greater
        } else {
            a.name.cmp(&b.name)
        }
    });

    Ok(workspaces)
}

/// Find workspace index by name, or the active one if name is None.
fn resolve_target(
    name: Option<&str>,
    workspaces: &[WorkspaceInfo],
) -> Result<usize> {
    match name {
        Some(n) => workspaces
            .iter()
            .position(|w| w.name == n)
            .with_context(|| format!("Workspace '{n}' not found")),
        None => workspaces
            .iter()
            .position(|w| w.is_active)
            .context("No active workspace found — specify a workspace name"),
    }
}

// ── Clone ───────────────────────────────────────────────────────

async fn clone_workspace(
    source_config: &Config,
    name: &str,
    port_override: Option<u16>,
    switch: bool,
) -> Result<()> {
    validate_workspace_name(name)?;

    let home = default_config_dir()?
        .parent()
        .context("Home directory not found")?
        .to_path_buf();
    let new_config_dir = home.join(format!(".zeroclaw-{name}"));
    let new_workspace_dir = new_config_dir.join("workspace");

    if new_config_dir.exists() {
        bail!(
            "Workspace directory already exists: {}",
            new_config_dir.display()
        );
    }

    // Read and decrypt source config
    let source_config_dir = source_config
        .config_path
        .parent()
        .context("Source config path must have a parent directory")?;
    let source_toml =
        std::fs::read_to_string(&source_config.config_path).with_context(|| {
            format!(
                "Failed to read source config: {}",
                source_config.config_path.display()
            )
        })?;
    let mut new_config: Config =
        toml::from_str(&source_toml).context("Failed to parse source config.toml")?;

    // Decrypt secrets from source store so they can be re-encrypted with the new key
    let source_store = SecretStore::new(source_config_dir, source_config.secrets.encrypt);
    decrypt_optional_secret(&source_store, &mut new_config.api_key, "api_key")?;
    decrypt_optional_secret(
        &source_store,
        &mut new_config.composio.api_key,
        "composio.api_key",
    )?;
    decrypt_optional_secret(
        &source_store,
        &mut new_config.browser.computer_use.api_key,
        "browser.computer_use.api_key",
    )?;
    decrypt_optional_secret(
        &source_store,
        &mut new_config.web_search.brave_api_key,
        "web_search.brave_api_key",
    )?;
    decrypt_optional_secret(
        &source_store,
        &mut new_config.storage.provider.config.db_url,
        "storage.provider.config.db_url",
    )?;
    for agent in new_config.agents.values_mut() {
        decrypt_optional_secret(&source_store, &mut agent.api_key, "agents.*.api_key")?;
    }
    if let Some(ref mut ns) = new_config.channels_config.nostr {
        decrypt_secret(&source_store, &mut ns.private_key, "nostr.private_key")?;
    }

    // Determine gateway port
    let port = match port_override {
        Some(p) => p,
        None => auto_assign_port(&home)?,
    };
    new_config.gateway.port = port;

    // Clear paired tokens — will be populated by mutual trust below
    new_config.gateway.paired_tokens = vec![];

    // Update paths
    new_config.config_path = new_config_dir.join("config.toml");
    new_config.workspace_dir = new_workspace_dir.clone();

    // Save — this creates new_config_dir, generates a new .secret_key, and re-encrypts
    new_config.save().await?;

    // Scaffold workspace files
    let default_ctx = ProjectContext::default();
    scaffold_workspace(&new_workspace_dir, &default_ctx).await?;

    // ── Mutual trust: exchange pairing tokens ───────────────────
    exchange_peer_tokens(source_config_dir, &new_config_dir, name)?;

    // Optionally switch
    if switch {
        persist_active_workspace_config_dir(&new_config_dir).await?;
    }

    let source_name = workspace_name_from_dir(
        &source_config_dir
            .file_name()
            .unwrap_or_default()
            .to_string_lossy(),
    );

    println!("Workspace cloned successfully:");
    println!("  Name:      {name}");
    println!("  Path:      {}", new_config_dir.display());
    println!("  Port:      {port}");
    println!("  Peers:     mutual trust established with '{source_name}'");
    if switch {
        println!("  Active:    yes (switched)");
    } else {
        println!("  Active:    no (use `zeroclaw workspace switch {name}` to activate)");
    }
    println!();
    println!("Start the daemon:");
    println!("  zeroclaw workspace start {name}");

    Ok(())
}

/// Scan all `.zeroclaw*` directories and pick max(port) + 1.
fn auto_assign_port(home: &Path) -> Result<u16> {
    let mut max_port: u16 = 0;

    for entry in std::fs::read_dir(home).context("Failed to read home directory")? {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.starts_with(".zeroclaw") {
            continue;
        }
        let config_path = entry.path().join("config.toml");
        if !config_path.exists() {
            continue;
        }
        if let Ok(contents) = std::fs::read_to_string(&config_path) {
            if let Ok(cfg) = toml::from_str::<Config>(&contents) {
                if cfg.gateway.port > max_port {
                    max_port = cfg.gateway.port;
                }
            }
        }
    }

    if max_port == 0 {
        max_port = 37173; // default port
    }

    max_port
        .checked_add(1)
        .context("Port overflow — specify --port manually")
}

// ── Peer token exchange ─────────────────────────────────────────

/// Generate two independent tokens for bidirectional peer authentication.
///
/// - `token_ab` (source -> new): plaintext stored in source's `.peer_tokens/{new_name}.token`
/// - `token_ba` (new -> source): plaintext stored in new's `.peer_tokens/{source_name}.token`
///
/// Both SHA-256 hashes are written to both configs' `paired_tokens` so each
/// gateway accepts requests authenticated with either token.
fn exchange_peer_tokens(
    source_config_dir: &Path,
    new_config_dir: &Path,
    new_name: &str,
) -> Result<()> {
    use sha2::{Digest, Sha256};

    // Two independent tokens: one per direction
    let token_ab = uuid::Uuid::new_v4().to_string(); // source uses to call new
    let token_ba = uuid::Uuid::new_v4().to_string(); // new uses to call source

    let hash_ab = format!("{:x}", Sha256::digest(token_ab.as_bytes()));
    let hash_ba = format!("{:x}", Sha256::digest(token_ba.as_bytes()));

    let source_name = workspace_name_from_dir(
        &source_config_dir
            .file_name()
            .unwrap_or_default()
            .to_string_lossy(),
    );

    // Source stores token_ab (what it presents when calling new)
    write_peer_token(source_config_dir, new_name, &token_ab)?;
    // New stores token_ba (what it presents when calling source)
    write_peer_token(new_config_dir, &source_name, &token_ba)?;

    // Both gateways must accept both hashes for bidirectional auth
    append_paired_token_hash(source_config_dir, &hash_ab)?;
    append_paired_token_hash(source_config_dir, &hash_ba)?;
    append_paired_token_hash(new_config_dir, &hash_ab)?;
    append_paired_token_hash(new_config_dir, &hash_ba)?;

    Ok(())
}

fn write_peer_token(config_dir: &Path, peer_name: &str, plaintext: &str) -> Result<()> {
    let tokens_dir = config_dir.join(".peer_tokens");
    std::fs::create_dir_all(&tokens_dir)
        .with_context(|| format!("Failed to create {}", tokens_dir.display()))?;
    let token_path = tokens_dir.join(format!("{peer_name}.token"));
    std::fs::write(&token_path, plaintext)
        .with_context(|| format!("Failed to write {}", token_path.display()))?;

    // Restrict token file to owner-only read/write (0600)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&token_path, std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("Failed to set permissions on {}", token_path.display()))?;
    }

    Ok(())
}

/// Append a hash to `gateway.paired_tokens` via text-level editing to preserve
/// comments, field ordering, and encrypted secrets in the original TOML.
fn append_paired_token_hash(config_dir: &Path, hash_hex: &str) -> Result<()> {
    let config_path = config_dir.join("config.toml");
    let toml_str = std::fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read {}", config_path.display()))?;

    // Minimal struct to extract only paired_tokens without requiring full Config parse.
    #[derive(Default, serde::Deserialize)]
    struct GatewayTokens {
        #[serde(default)]
        paired_tokens: Vec<String>,
    }
    #[derive(Default, serde::Deserialize)]
    struct MinimalConfig {
        #[serde(default)]
        gateway: GatewayTokens,
    }

    let existing: MinimalConfig =
        toml::from_str(&toml_str).context("Failed to parse config.toml for token injection")?;

    if existing.gateway.paired_tokens.contains(&hash_hex.to_string()) {
        return Ok(()); // already present
    }

    // Build the updated paired_tokens array value
    let mut new_tokens = existing.gateway.paired_tokens.clone();
    new_tokens.push(hash_hex.to_string());
    let new_value = format!(
        "paired_tokens = [{}]",
        new_tokens
            .iter()
            .map(|t| format!("\"{}\"", t))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Text-level replacement: find and replace the existing paired_tokens line,
    // or insert one under [gateway] if absent.
    let updated_toml = if let Some(start) = toml_str.find("paired_tokens") {
        let line_start = toml_str[..start].rfind('\n').map_or(0, |i| i + 1);
        let line_end = toml_str[start..]
            .find('\n')
            .map_or(toml_str.len(), |i| start + i);
        format!(
            "{}{}{}",
            &toml_str[..line_start],
            new_value,
            &toml_str[line_end..]
        )
    } else if let Some(gw_pos) = toml_str.find("[gateway]") {
        // No paired_tokens line — insert after [gateway] header
        let after_header = toml_str[gw_pos..]
            .find('\n')
            .map_or(toml_str.len(), |i| gw_pos + i + 1);
        format!(
            "{}{}\n{}",
            &toml_str[..after_header],
            new_value,
            &toml_str[after_header..]
        )
    } else {
        // No [gateway] section at all — append one
        format!("{}\n[gateway]\n{}\n", toml_str, new_value)
    };

    std::fs::write(&config_path, updated_toml)
        .with_context(|| format!("Failed to write {}", config_path.display()))?;

    Ok(())
}

// ── List ────────────────────────────────────────────────────────

fn list_workspaces(current_config: &Config) -> Result<()> {
    let active_config_dir = current_config
        .config_path
        .parent()
        .map(|p| p.to_path_buf());

    let workspaces = discover_workspaces(active_config_dir.as_deref())?;

    println!("Workspaces:\n");
    let hdr = |n: &str, p: &str, s: &str, t: &str| {
        format!("       {n:<16} {p:>6}  {s:<15} {t}")
    };
    println!("{}", hdr("NAME", "PORT", "STATUS", "PATH"));
    println!("{}", hdr("────", "────", "──────", "────"));

    if workspaces.is_empty() {
        println!("  (no workspaces found)");
    } else {
        for ws in &workspaces {
            let marker = if ws.is_active { "[*]" } else { "   " };
            let status = match &ws.running {
                RunStatus::Running { pid } => format!("running ({pid})"),
                RunStatus::Stopped => "stopped".to_string(),
                RunStatus::Stale => "stale".to_string(),
            };
            println!(
                "  {marker} {:<16} {:>6}  {:<15} {}",
                ws.name,
                ws.port,
                status,
                ws.config_dir.display()
            );
        }
    }

    println!();
    println!("  [*] = active workspace");

    Ok(())
}

// ── Start ───────────────────────────────────────────────────────

fn start_workspace(name: Option<&str>, config: &Config) -> Result<()> {
    let active_config_dir = config.config_path.parent().map(|p| p.to_path_buf());
    let workspaces = discover_workspaces(active_config_dir.as_deref())?;

    let targets: Vec<usize> = match name {
        Some(n) => {
            let idx = resolve_target(Some(n), &workspaces)?;
            vec![idx]
        }
        None => {
            // Start all stopped workspaces
            workspaces
                .iter()
                .enumerate()
                .filter(|(_, w)| matches!(w.running, RunStatus::Stopped | RunStatus::Stale))
                .map(|(i, _)| i)
                .collect()
        }
    };

    if targets.is_empty() {
        println!("All workspaces are already running.");
        return Ok(());
    }

    let exe = std::env::current_exe().context("Failed to determine current executable path")?;

    for idx in targets {
        let ws = &workspaces[idx];

        if matches!(ws.running, RunStatus::Running { .. }) {
            println!(
                "Workspace '{}' is already running (PID {}).",
                ws.name,
                match ws.running {
                    RunStatus::Running { pid } => pid,
                    _ => 0,
                }
            );
            continue;
        }

        // Clean stale PID file
        if matches!(ws.running, RunStatus::Stale) {
            let _ = std::fs::remove_file(ws.config_dir.join("daemon.pid"));
        }

        // Ensure logs directory
        let logs_dir = ws.config_dir.join("logs");
        std::fs::create_dir_all(&logs_dir)
            .with_context(|| format!("Failed to create logs dir: {}", logs_dir.display()))?;

        let stdout_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(logs_dir.join("daemon.stdout.log"))
            .context("Failed to open stdout log")?;
        let stderr_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(logs_dir.join("daemon.stderr.log"))
            .context("Failed to open stderr log")?;

        let config_dir_str = ws.config_dir.to_string_lossy().to_string();

        let child = std::process::Command::new(&exe)
            .args(["--config-dir", &config_dir_str, "daemon"])
            .stdin(std::process::Stdio::null())
            .stdout(stdout_file)
            .stderr(stderr_file)
            .spawn()
            .with_context(|| format!("Failed to spawn daemon for workspace '{}'", ws.name))?;

        let spawned_pid = child.id();

        // Wait briefly for the daemon to write its PID file and become alive
        std::thread::sleep(std::time::Duration::from_secs(1));

        let alive = read_pid_file(&ws.config_dir)
            .map(is_pid_alive)
            .unwrap_or(false);

        if alive {
            println!(
                "Started '{}': PID {}, http://{}:{}",
                ws.name, spawned_pid, ws.host, ws.port
            );
        } else {
            println!(
                "Warning: daemon for '{}' may not have started correctly (PID {}).",
                ws.name, spawned_pid
            );
            println!(
                "  Check logs: {}/logs/daemon.stderr.log",
                ws.config_dir.display()
            );
        }
    }

    Ok(())
}

// ── Stop ────────────────────────────────────────────────────────

fn stop_workspace(name: Option<&str>, all: bool, config: &Config) -> Result<()> {
    let active_config_dir = config.config_path.parent().map(|p| p.to_path_buf());
    let workspaces = discover_workspaces(active_config_dir.as_deref())?;

    let targets: Vec<usize> = if all {
        workspaces
            .iter()
            .enumerate()
            .filter(|(_, w)| matches!(w.running, RunStatus::Running { .. }))
            .map(|(i, _)| i)
            .collect()
    } else {
        let idx = resolve_target(name, &workspaces)?;
        vec![idx]
    };

    if targets.is_empty() {
        println!("No running workspaces to stop.");
        return Ok(());
    }

    for idx in targets {
        let ws = &workspaces[idx];

        match ws.running {
            RunStatus::Running { pid } => {
                stop_pid(pid, &ws.name, &ws.config_dir)?;
            }
            RunStatus::Stale => {
                let _ = std::fs::remove_file(ws.config_dir.join("daemon.pid"));
                println!("Workspace '{}': cleaned stale PID file.", ws.name);
            }
            RunStatus::Stopped => {
                println!("Workspace '{}' is not running.", ws.name);
            }
        }
    }

    Ok(())
}

/// Send SIGTERM, wait up to 5s, then SIGKILL if still alive.
/// Cleans up the PID file after successful termination.
#[cfg(unix)]
fn stop_pid(pid: u32, name: &str, config_dir: &Path) -> Result<()> {
    let pid_i32 = pid as libc::pid_t;

    // SIGTERM
    unsafe {
        libc::kill(pid_i32, libc::SIGTERM);
    }

    // Poll up to 5s
    for _ in 0..10 {
        std::thread::sleep(std::time::Duration::from_millis(500));
        if !is_pid_alive(pid) {
            let _ = std::fs::remove_file(config_dir.join("daemon.pid"));
            println!("Stopped '{name}' (PID {pid}).");
            return Ok(());
        }
    }

    // SIGKILL
    unsafe {
        libc::kill(pid_i32, libc::SIGKILL);
    }
    std::thread::sleep(std::time::Duration::from_millis(500));

    if is_pid_alive(pid) {
        bail!("Failed to kill daemon for '{name}' (PID {pid})");
    }

    let _ = std::fs::remove_file(config_dir.join("daemon.pid"));
    println!("Killed '{name}' (PID {pid}, SIGKILL).");
    Ok(())
}

#[cfg(not(unix))]
fn stop_pid(_pid: u32, _name: &str, _config_dir: &Path) -> Result<()> {
    bail!("Stopping daemons is only supported on Unix systems");
}

// ── Switch ──────────────────────────────────────────────────────

async fn switch_workspace(name: &str) -> Result<()> {
    let default_dir = default_config_dir()?;
    let home = default_dir
        .parent()
        .context("Home directory not found")?
        .to_path_buf();

    let target_dir = if name == "default" {
        default_dir
    } else {
        validate_workspace_name(name)?;
        home.join(format!(".zeroclaw-{name}"))
    };

    if !target_dir.join("config.toml").exists() {
        bail!(
            "Workspace not found: {} (no config.toml at {})",
            name,
            target_dir.display()
        );
    }

    persist_active_workspace_config_dir(&target_dir).await?;

    println!("Switched to workspace: {name}");
    println!("  Path: {}", target_dir.display());
    println!();
    println!("Restart the daemon to apply: zeroclaw service restart");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_name_rejects_empty() {
        assert!(validate_workspace_name("").is_err());
    }

    #[test]
    fn validate_name_rejects_default() {
        assert!(validate_workspace_name("default").is_err());
    }

    #[test]
    fn validate_name_rejects_special_chars() {
        assert!(validate_workspace_name("agent 2").is_err());
        assert!(validate_workspace_name("agent/2").is_err());
        assert!(validate_workspace_name(".hidden").is_err());
        assert!(validate_workspace_name("-leading").is_err());
        assert!(validate_workspace_name("agent@2").is_err());
    }

    #[test]
    fn validate_name_accepts_valid() {
        assert!(validate_workspace_name("agent2").is_ok());
        assert!(validate_workspace_name("my-workspace").is_ok());
        assert!(validate_workspace_name("ws_123").is_ok());
        assert!(validate_workspace_name("A").is_ok());
        assert!(validate_workspace_name("test-agent-01").is_ok());
    }

    #[test]
    fn workspace_name_from_dir_default() {
        assert_eq!(workspace_name_from_dir(".zeroclaw"), "default");
    }

    #[test]
    fn workspace_name_from_dir_named() {
        assert_eq!(workspace_name_from_dir(".zeroclaw-agent2"), "agent2");
        assert_eq!(
            workspace_name_from_dir(".zeroclaw-my-ws"),
            "my-ws"
        );
    }

    #[test]
    fn workspace_name_from_dir_unexpected() {
        // Fallback: returns as-is if prefix doesn't match
        assert_eq!(workspace_name_from_dir("other"), "other");
    }

    #[cfg(unix)]
    #[test]
    fn is_pid_alive_current_process() {
        assert!(is_pid_alive(std::process::id()));
    }

    #[cfg(unix)]
    #[test]
    fn is_pid_alive_bogus_pid() {
        // PID 4_000_000 is extremely unlikely to exist
        assert!(!is_pid_alive(4_000_000));
    }

    #[test]
    fn read_pid_file_valid() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("daemon.pid"), "12345").unwrap();
        assert_eq!(read_pid_file(dir.path()), Some(12345));
    }

    #[test]
    fn read_pid_file_missing() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read_pid_file(dir.path()), None);
    }

    #[test]
    fn read_pid_file_invalid_content() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("daemon.pid"), "not-a-number").unwrap();
        assert_eq!(read_pid_file(dir.path()), None);
    }

    #[test]
    fn read_pid_file_zero() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("daemon.pid"), "0").unwrap();
        assert_eq!(read_pid_file(dir.path()), None);
    }

    #[cfg(unix)]
    #[test]
    fn write_peer_token_sets_permissions_0600() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        write_peer_token(dir.path(), "test-peer", "secret-token").unwrap();
        let meta = std::fs::metadata(dir.path().join(".peer_tokens/test-peer.token")).unwrap();
        assert_eq!(meta.permissions().mode() & 0o777, 0o600);
    }

    #[test]
    fn write_peer_token_content() {
        let dir = tempfile::tempdir().unwrap();
        write_peer_token(dir.path(), "peer1", "my-token").unwrap();
        let content = std::fs::read_to_string(dir.path().join(".peer_tokens/peer1.token")).unwrap();
        assert_eq!(content, "my-token");
    }

    #[test]
    fn append_paired_token_hash_adds_to_existing() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.toml"),
            "[gateway]\nport = 37173\npaired_tokens = [\"aaa\"]\n",
        )
        .unwrap();
        append_paired_token_hash(dir.path(), "bbb").unwrap();
        let content = std::fs::read_to_string(dir.path().join("config.toml")).unwrap();
        assert!(content.contains("\"aaa\""));
        assert!(content.contains("\"bbb\""));
    }

    #[test]
    fn append_paired_token_hash_skips_duplicate() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.toml"),
            "[gateway]\nport = 37173\npaired_tokens = [\"aaa\"]\n",
        )
        .unwrap();
        append_paired_token_hash(dir.path(), "aaa").unwrap();
        let content = std::fs::read_to_string(dir.path().join("config.toml")).unwrap();
        // Should still be just one "aaa", not duplicated
        assert_eq!(content.matches("\"aaa\"").count(), 1);
    }

    #[test]
    fn append_paired_token_hash_inserts_under_gateway() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.toml"),
            "api_key = \"test\"\n\n[gateway]\nport = 37173\n",
        )
        .unwrap();
        append_paired_token_hash(dir.path(), "newhash").unwrap();
        let content = std::fs::read_to_string(dir.path().join("config.toml")).unwrap();
        assert!(content.contains("\"newhash\""));
        // Verify original content preserved
        assert!(content.contains("api_key = \"test\""));
        assert!(content.contains("port = 37173"));
    }

    #[test]
    fn append_paired_token_hash_preserves_comments() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.toml"),
            "# Main config\napi_key = \"test\"\n\n[gateway]\n# Gateway settings\nport = 37173\npaired_tokens = []\n",
        )
        .unwrap();
        append_paired_token_hash(dir.path(), "abc123").unwrap();
        let content = std::fs::read_to_string(dir.path().join("config.toml")).unwrap();
        assert!(content.contains("# Main config"));
        assert!(content.contains("# Gateway settings"));
        assert!(content.contains("\"abc123\""));
    }

    #[cfg(unix)]
    #[test]
    fn is_daemon_lock_held_unlocked_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("daemon.pid"), "99999").unwrap();
        // No flock held — should return false
        assert!(!is_daemon_lock_held(dir.path()));
    }

    #[cfg(unix)]
    #[test]
    fn is_daemon_lock_held_no_pid_file() {
        let dir = tempfile::tempdir().unwrap();
        // No daemon.pid file — should return false
        assert!(!is_daemon_lock_held(dir.path()));
    }
}
