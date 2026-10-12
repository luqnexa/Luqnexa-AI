// SPDX-License-Identifier: AGPL-3.0-only
//! Bounded offline draft assistance. Template generation plus local model ranking,
//! not a general-purpose language model. Drafts have no send/insert capability.
use crate::Model;
#[derive(Clone, Copy, Debug)]
pub enum DeviceBudget {
    Low,
    High,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Draft {
    pub text: String,
    pub label: &'static str,
}
#[derive(Debug, PartialEq, Eq)]
pub enum AssistError {
    Disabled,
    InvalidInput,
    ContextNotAllowed,
}
pub struct Assistant {
    enabled: bool,
    budget: DeviceBudget,
}
impl Assistant {
    pub fn new(budget: DeviceBudget) -> Self {
        Self {
            enabled: false,
            budget,
        }
    }
    pub fn enable(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    /// Context is explicitly supplied by the authorized host, never scraped from an app.
    pub fn absence_email(
        &self,
        recipient: &str,
        day: &str,
        reason: &str,
        context: &str,
        context_allowed: bool,
    ) -> Result<Vec<Draft>, AssistError> {
        if !self.enabled {
            return Err(AssistError::Disabled);
        }
        if !context.is_empty() && !context_allowed {
            return Err(AssistError::ContextNotAllowed);
        }
        let bound = match self.budget {
            DeviceBudget::Low => 256,
            DeviceBudget::High => 1024,
        };
        if recipient.is_empty()
            || day.is_empty()
            || recipient.chars().count() > 32
            || day.chars().count() > 32
            || reason.chars().count() > 128
            || context.chars().count() > bound
            || [recipient, day, reason]
                .iter()
                .any(|v| v.chars().any(char::is_control))
        {
            return Err(AssistError::InvalidInput);
        }
        let reason = if reason.is_empty() {
            String::new()
        } else {
            format!("{reason}のため、")
        };
        let mut drafts = vec![
            Draft {
                label: "丁寧",
                text: format!(
                    "{recipient}\n\nお世話になっております。\n{day}は、{reason}欠席いたします。\nご迷惑をおかけしますが、よろしくお願いいたします。"
                ),
            },
            Draft {
                label: "短め",
                text: format!(
                    "{recipient}、{day}は{reason}欠席いたします。どうぞよろしくお願いいたします。"
                ),
            },
            Draft {
                label: "相談",
                text: format!(
                    "{recipient}\n\n{day}は{reason}出席が難しい状況です。必要な手続きや課題がありましたら、ご教示いただけますと幸いです。"
                ),
            },
        ];
        let model = Model::bundled();
        drafts.sort_by_key(|d| std::cmp::Reverse(model.score(context, &d.text)));
        if matches!(self.budget, DeviceBudget::Low) {
            drafts.truncate(2);
        }
        Ok(drafts)
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Commitment {
    pub day: String,
    pub start_minutes: u16,
    pub end_minutes: u16,
}
/// Conflicts are reported from explicit structured commitments, not inferred facts.
pub fn conflicts(
    existing: &[Commitment],
    proposal: &Commitment,
) -> Result<Vec<usize>, AssistError> {
    if std::iter::once(proposal)
        .chain(existing)
        .any(|p| p.day.is_empty() || p.start_minutes >= p.end_minutes || p.end_minutes > 1440)
    {
        return Err(AssistError::InvalidInput);
    }
    Ok(existing
        .iter()
        .enumerate()
        .filter(|(_, p)| {
            p.day == proposal.day
                && p.start_minutes < proposal.end_minutes
                && proposal.start_minutes < p.end_minutes
        })
        .map(|(i, _)| i)
        .collect())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_enable_context_and_preview_only() {
        let mut a = Assistant::new(DeviceBudget::Low);
        assert_eq!(
            a.absence_email("先生", "明日", "", "", false),
            Err(AssistError::Disabled)
        );
        a.enable(true);
        assert_eq!(
            a.absence_email("先生", "明日", "", "履歴", false),
            Err(AssistError::ContextNotAllowed)
        );
        let drafts = a.absence_email("先生", "明日", "", "", false).unwrap();
        assert_eq!(drafts.len(), 2);
        assert!(
            drafts
                .iter()
                .all(|d| d.text.contains("先生") && d.text.contains("明日"))
        );
        assert!(
            a.absence_email("先生\n追加", "明日", "", "", false)
                .is_err()
        );
    }
    #[test]
    fn detects_only_real_schedule_overlap() {
        let old = vec![Commitment {
            day: "2026-10-13".into(),
            start_minutes: 540,
            end_minutes: 600,
        }];
        let proposal = Commitment {
            day: "2026-10-13".into(),
            start_minutes: 570,
            end_minutes: 630,
        };
        assert_eq!(conflicts(&old, &proposal).unwrap(), vec![0]);
        let next = Commitment {
            start_minutes: 600,
            ..proposal
        };
        assert!(conflicts(&old, &next).unwrap().is_empty());
    }
}
