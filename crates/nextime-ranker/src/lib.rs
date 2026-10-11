// SPDX-License-Identifier: AGPL-3.0-only
//! Integer frequency scoring with deterministic ties and surface deduplication.
use std::collections::HashSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub surface: String,
    pub score: i64,
    pub segments: usize,
}

pub const UNKNOWN_SCORE: i64 = -12_288;

/// Monotonic fixed-point log2 approximation minus a per-token cost.
/// Frequencies are sample weights, not trained probabilities. Larger is better.
pub fn word_score(frequency: u32) -> i64 {
    let value = u64::from(frequency) + 1;
    let exponent = value.ilog2();
    let base = 1_u64 << exponent;
    i64::from(exponent) * 256 + ((value - base) * 256 / base) as i64 - 8_448
}

/// Sort, keep the best path per visible surface, and bound the output.
pub fn rank(candidates: &mut Vec<Candidate>, limit: usize) {
    candidates.sort_unstable_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| a.segments.cmp(&b.segments))
            .then_with(|| a.surface.cmp(&b.surface))
    });
    let mut seen = HashSet::new();
    candidates.retain(|c| seen.insert(c.surface.clone()));
    candidates.truncate(limit);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frequency_order_and_extremes() {
        for pair in [0, 1, 2, 10, 1000, u32::MAX - 1].windows(2) {
            assert!(word_score(pair[0]) < word_score(pair[1]));
        }
        assert!(word_score(u32::MAX) < 0);
        assert!(word_score(0) > UNKNOWN_SCORE);
    }
    #[test]
    fn deduplicate_ties_and_limit() {
        let mut items = vec![
            Candidate {
                surface: "b".into(),
                score: 1,
                segments: 1,
            },
            Candidate {
                surface: "a".into(),
                score: 1,
                segments: 1,
            },
            Candidate {
                surface: "a".into(),
                score: 0,
                segments: 2,
            },
        ];
        rank(&mut items, 2);
        assert_eq!(
            items.iter().map(|c| c.surface.as_str()).collect::<Vec<_>>(),
            ["a", "b"]
        );
        rank(&mut items, 0);
        assert!(items.is_empty());
    }
}
