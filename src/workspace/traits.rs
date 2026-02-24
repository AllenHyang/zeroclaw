use anyhow::Result;
use std::path::Path;

use super::WorkspaceInfo;

/// Workspace discovery backend.
///
/// Default implementation ([`LocalDiscovery`](super::LocalDiscovery)) scans
/// the local filesystem (`~/.zeroclaw*`).  Future implementations may query
/// an HTTP registry, DNS-SD, or a centralized workspace database.
pub trait WorkspaceDiscovery: Send + Sync {
    /// Enumerate all known workspaces.
    ///
    /// `active_config_dir`, when provided, marks the active workspace in the
    /// returned list so callers can distinguish it visually or logically.
    fn discover(&self, active_config_dir: Option<&Path>) -> Result<Vec<WorkspaceInfo>>;

    /// Human-readable backend name (e.g. `"local"`, `"http"`).
    fn name(&self) -> &str;
}
