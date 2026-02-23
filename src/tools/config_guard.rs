//! Config-file write guard: prevents the agent from modifying protected keys
//! in `config.toml` via `file_write` or `file_edit`.
//!
//! Protected keys are declared in `[autonomy].protected_config_keys`.  When the
//! agent writes new content to a file whose resolved path ends in `config.toml`
//! inside the ZeroClaw home directory, this guard parses old and new TOML,
//! compares top-level keys, and rejects the write if any protected key changed.

use std::path::Path;

/// Check whether writing `new_content` to `resolved_path` would violate
/// protected-config-key policy.
///
/// Returns `Ok(())` when the write is allowed, or `Err(message)` describing
/// which protected keys were modified.
///
/// The guard only activates when **all** of:
/// 1. `protected_keys` is non-empty
/// 2. The resolved path filename is `config.toml`
/// 3. The file already exists (new files are not guarded)
///
/// For `file_edit`, callers should pass the *post-edit* content as `new_content`.
pub fn check_protected_config_keys(
    resolved_path: &Path,
    new_content: &str,
    protected_keys: &[String],
) -> Result<(), String> {
    if protected_keys.is_empty() {
        return Ok(());
    }

    // Only guard files named config.toml
    let file_name = resolved_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    if file_name != "config.toml" {
        return Ok(());
    }

    // Read existing content; if the file doesn't exist yet, allow the write
    let old_content = match std::fs::read_to_string(resolved_path) {
        Ok(c) => c,
        Err(_) => return Ok(()),
    };

    let old_table: toml::Table = match old_content.parse() {
        Ok(t) => t,
        Err(_) => return Ok(()), // unparseable old file — don't block
    };
    let new_table: toml::Table = match new_content.parse() {
        Ok(t) => t,
        Err(_) => return Ok(()), // unparseable new content — let TOML error surface later
    };

    let mut violations = Vec::new();
    for key in protected_keys {
        let old_val = old_table.get(key.as_str());
        let new_val = new_table.get(key.as_str());
        if old_val != new_val {
            let old_display = old_val.map_or("(absent)".to_string(), |v| v.to_string());
            let new_display = new_val.map_or("(absent)".to_string(), |v| v.to_string());
            violations.push(format!("  {key}: {old_display} → {new_display}"));
        }
    }

    if violations.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "Blocked: modifying protected config keys requires human approval.\n\
             Changed keys:\n{}",
            violations.join("\n")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_write_when_no_protected_keys() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "default_model = \"glm-5\"\n").unwrap();
        let result = check_protected_config_keys(tmp.path(), "default_model = \"gpt-4\"\n", &[]);
        assert!(result.is_ok());
    }

    #[test]
    fn allows_write_to_non_config_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("other.toml");
        std::fs::write(&path, "default_model = \"glm-5\"\n").unwrap();
        let result = check_protected_config_keys(
            &path,
            "default_model = \"gpt-4\"\n",
            &["default_model".into()],
        );
        assert!(result.is_ok());
    }

    #[test]
    fn blocks_protected_key_modification() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "default_model = \"glm-5\"\nother = 1\n").unwrap();
        let result = check_protected_config_keys(
            &path,
            "default_model = \"glm-4-flash\"\nother = 1\n",
            &["default_model".into()],
        );
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("default_model"));
        assert!(msg.contains("human approval"));
    }

    #[test]
    fn allows_unprotected_key_change() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "default_model = \"glm-5\"\nother = 1\n").unwrap();
        let result = check_protected_config_keys(
            &path,
            "default_model = \"glm-5\"\nother = 2\n",
            &["default_model".into()],
        );
        assert!(result.is_ok());
    }

    #[test]
    fn allows_new_file_creation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        // file does not exist
        let result = check_protected_config_keys(
            &path,
            "default_model = \"anything\"\n",
            &["default_model".into()],
        );
        assert!(result.is_ok());
    }

    #[test]
    fn blocks_multiple_protected_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "default_model = \"glm-5\"\napi_key = \"secret\"\nother = 1\n",
        )
        .unwrap();
        let result = check_protected_config_keys(
            &path,
            "default_model = \"gpt-4\"\napi_key = \"new-secret\"\nother = 1\n",
            &["default_model".into(), "api_key".into()],
        );
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("default_model"));
        assert!(msg.contains("api_key"));
    }

    #[test]
    fn blocks_protected_table_key_modification() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let old = r#"
default_model = "glm-5"

[reliability.model_fallbacks]
"glm-5" = ["gemini-2.0-flash-exp"]
"#;
        let new = r#"
default_model = "glm-5"

[reliability.model_fallbacks]
"glm-4-flash" = ["gemini-2.5-flash"]
"#;
        std::fs::write(&path, old).unwrap();
        let result = check_protected_config_keys(&path, new, &["reliability".into()]);
        assert!(result.is_err());
    }
}
