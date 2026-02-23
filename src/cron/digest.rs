//! Auto-registered daily digest job.
//!
//! Summarizes the day's completed goals, explorations, and learnings
//! into a single memory entry for easy review.

use crate::config::Config;
use anyhow::Result;

/// Default cron expression: 11:00 PM daily.
const DEFAULT_SCHEDULE_EXPR: &str = "0 23 * * *";

/// Job name marker used to identify digest jobs.
pub const DIGEST_JOB_NAME: &str = "__daily_digest";

/// The prompt instructs the agent to produce a concise daily digest using
/// existing tools (memory_recall, memory_store).
const DIGEST_PROMPT: &str = "\
You are generating a daily digest. Follow these steps exactly:

1. Use `memory_recall` with query \"milestone\" to find today's completed goals \
   and milestones.

2. Use `memory_recall` with query \"exploration-journal\" to find today's \
   exploration journal entries.

3. Write a concise summary (under 500 characters) of what was accomplished \
   and learned today. Focus on outcomes and key insights, not process details.

4. Use `memory_store` with key \"daily-digest-YYYY-MM-DD\" (use today's date), \
   category \"digest\", and your summary as content.

5. If a notification channel is configured, send the digest to the user.

If there is no meaningful activity to summarize (no milestones, no explorations), \
store a brief note confirming the check was performed and skip notification.";

/// Create a daily digest cron agent job.
///
/// Schedule: 11:00 PM daily (local time).
/// Job type: agent with `__daily_digest` marker in the name.
/// Session target: isolated (does not disturb main sessions).
pub fn create_digest_job(config: &Config) -> Result<super::CronJob> {
    create_digest_job_with_schedule(config, DEFAULT_SCHEDULE_EXPR, None)
}

/// Create a digest job with a custom cron expression and optional timezone.
pub fn create_digest_job_with_schedule(
    config: &Config,
    cron_expr: &str,
    tz: Option<String>,
) -> Result<super::CronJob> {
    let schedule = super::Schedule::Cron {
        expr: cron_expr.into(),
        tz,
    };

    super::add_agent_job(
        config,
        Some(DIGEST_JOB_NAME.into()),
        schedule,
        DIGEST_PROMPT,
        super::SessionTarget::Isolated,
        None,  // use default model
        None,  // no delivery config
        false, // recurring job — do not delete after run
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cron::{JobType, Schedule, SessionTarget};
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
    fn create_digest_job_produces_valid_job() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp);

        let job = create_digest_job(&config).unwrap();

        assert_eq!(job.name.as_deref(), Some(DIGEST_JOB_NAME));
        assert_eq!(job.job_type, JobType::Agent);
        assert_eq!(job.session_target, SessionTarget::Isolated);
        assert!(!job.delete_after_run);
        assert!(job.enabled);
    }

    #[test]
    fn create_digest_job_uses_correct_schedule() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp);

        let job = create_digest_job(&config).unwrap();

        match &job.schedule {
            Schedule::Cron { expr, tz } => {
                assert_eq!(expr, DEFAULT_SCHEDULE_EXPR);
                assert!(tz.is_none());
            }
            other => panic!("Expected Cron schedule, got {other:?}"),
        }
    }

    #[test]
    fn create_digest_job_prompt_contains_key_instructions() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp);

        let job = create_digest_job(&config).unwrap();
        let prompt = job.prompt.expect("digest job must have a prompt");

        assert!(
            prompt.contains("memory_recall"),
            "prompt should instruct use of memory_recall"
        );
        assert!(
            prompt.contains("memory_store"),
            "prompt should instruct use of memory_store"
        );
        assert!(
            prompt.contains("daily-digest-YYYY-MM-DD"),
            "prompt should specify key format"
        );
        assert!(
            prompt.contains("digest"),
            "prompt should specify digest category"
        );
        assert!(
            prompt.contains("milestone"),
            "prompt should query for milestones"
        );
        assert!(
            prompt.contains("exploration-journal"),
            "prompt should query for exploration journals"
        );
    }

    #[test]
    fn create_digest_job_with_custom_schedule_applies_tz() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp);

        let job = create_digest_job_with_schedule(
            &config,
            "0 22 * * *",
            Some("Asia/Shanghai".into()),
        )
        .unwrap();

        match &job.schedule {
            Schedule::Cron { expr, tz } => {
                assert_eq!(expr, "0 22 * * *");
                assert_eq!(tz.as_deref(), Some("Asia/Shanghai"));
            }
            other => panic!("Expected Cron schedule, got {other:?}"),
        }
    }
}
