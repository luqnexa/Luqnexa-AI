// SPDX-License-Identifier: AGPL-3.0-only
//! Bounded, volatile learning from explicitly selected candidates only.
use nextime_ranker::{Candidate, rank};
use std::collections::BTreeMap;

#[derive(Debug, Default)]
pub struct PersonalModel {
    choices: BTreeMap<(String, String), u32>,
    contexts: BTreeMap<(String, String, Vec<u64>), u32>,
}
impl PersonalModel {
    fn features(context: &str) -> Vec<u64> {
        let chars: Vec<_> = context.chars().rev().take(96).collect();
        let mut features: Vec<_> = chars
            .windows(2)
            .filter(|pair| pair.iter().all(|c| c.is_alphanumeric()))
            .map(|pair| ((pair[1] as u64) << 32) | pair[0] as u64)
            .collect();
        features.sort_unstable();
        features.dedup();
        features
    }
    pub fn learn_with_context(&mut self, reading: &str, surface: &str, context: &str) {
        self.learn(reading, surface);
        if !self
            .choices
            .contains_key(&(reading.to_owned(), surface.to_owned()))
        {
            return;
        }
        let features = Self::features(context);
        if features.len() < 2 {
            return;
        }
        let key = (reading.to_owned(), surface.to_owned(), features);
        if !self.contexts.contains_key(&key) && self.contexts.len() >= 256 {
            if let Some(old) = self
                .contexts
                .iter()
                .min_by_key(|(_, count)| **count)
                .map(|(key, _)| key.clone())
            {
                self.contexts.remove(&old);
            }
        }
        let count = self.contexts.entry(key).or_default();
        *count = count.saturating_add(1).min(64);
    }
    pub fn rerank_with_context(
        &self,
        reading: &str,
        context: &str,
        candidates: &mut Vec<Candidate>,
        limit: usize,
    ) {
        if context.is_empty() || !self.contexts.keys().any(|(r, _, _)| r == reading) {
            self.rerank(reading, candidates, limit);
            return;
        }
        let features = Self::features(context);
        for candidate in candidates.iter_mut() {
            let mut best = 0;
            for ((r, surface, previous), count) in &self.contexts {
                if r != reading || surface != &candidate.surface {
                    continue;
                }
                let overlap = previous
                    .iter()
                    .filter(|f| features.binary_search(f).is_ok())
                    .count();
                let union = previous.len() + features.len() - overlap;
                if overlap >= 2 && overlap * 100 >= union * 30 {
                    best = best.max(
                        4096 + (overlap * 1024 / union) as i64 + i64::from((*count).min(8)) * 128,
                    );
                }
            }
            candidate.score += best;
        }
        rank(candidates, limit);
    }
    pub fn learn(&mut self, reading: &str, surface: &str) {
        // Do not retain entire sentences or unrestricted input history.
        if reading.is_empty()
            || surface.is_empty()
            || reading.chars().count() > 32
            || surface.chars().count() > 32
            || reading.chars().chain(surface.chars()).any(char::is_control)
        {
            return;
        }
        let key = (reading.to_owned(), surface.to_owned());
        if !self.choices.contains_key(&key) && self.choices.len() >= 1024 {
            // Evict the least selected entry, with deterministic tie breaking.
            if let Some(key) = self
                .choices
                .iter()
                .min_by_key(|(_, count)| **count)
                .map(|(key, _)| key.clone())
            {
                self.choices.remove(&key);
            }
        }
        let count = self.choices.entry(key).or_default();
        *count = count.saturating_add(1).min(64);
    }
    pub fn rerank(&self, reading: &str, candidates: &mut Vec<Candidate>, limit: usize) {
        for candidate in candidates.iter_mut() {
            let count = self
                .choices
                .get(&(reading.to_owned(), candidate.surface.clone()))
                .copied()
                .unwrap_or(0);
            // Strong enough to make an explicitly chosen homophone reusable;
            // bounded so learned choices cannot grow scores without limit.
            if count > 0 {
                candidate.score += 2048 + i64::from(count.min(8)) * 256;
            }
        }
        rank(candidates, limit);
    }
    pub fn clear(&mut self) {
        self.choices.clear();
        self.contexts.clear();
    }
    pub fn entry_count(&self) -> usize {
        self.choices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn contextual_choices_do_not_force_one_homophone_in_every_topic() {
        let mut model = PersonalModel::default();
        model.learn_with_context("はし", "箸", "夕食の準備ができました。");
        model.learn_with_context("はし", "橋", "川の向こうへ行きます。");
        let original = vec![
            Candidate {
                surface: "橋".into(),
                score: 100,
                segments: 1,
            },
            Candidate {
                surface: "箸".into(),
                score: 0,
                segments: 1,
            },
        ];
        for (context, expected) in [
            ("夕食の準備ができました。", "箸"),
            ("川の向こうへ行きます。", "橋"),
        ] {
            let mut candidates = original.clone();
            model.rerank_with_context("はし", context, &mut candidates, 2);
            assert_eq!(candidates[0].surface, expected);
        }
        let mut unknown = original.clone();
        model.rerank_with_context("はし", "映画を見ています。", &mut unknown, 2);
        assert_eq!(unknown, original);
        model.clear();
        assert!(model.contexts.is_empty());
        for i in 0..300 {
            model.learn_with_context("はし", "橋", &format!("川沿いの地点{i}"));
        }
        assert_eq!(model.contexts.len(), 256);
    }
    #[test]
    fn learns_only_exact_reading_and_is_bounded_and_clearable() {
        let mut model = PersonalModel::default();
        model.learn("はし", "箸");
        let original = vec![
            Candidate {
                surface: "橋".into(),
                score: 0,
                segments: 1,
            },
            Candidate {
                surface: "箸".into(),
                score: -100,
                segments: 1,
            },
        ];
        let mut candidates = original.clone();
        model.rerank("はし", &mut candidates, 2);
        assert_eq!(candidates[0].surface, "箸");
        let mut unrelated = original;
        model.rerank("ばし", &mut unrelated, 2);
        assert_eq!(unrelated[0].surface, "橋");
        model.learn(&"あ".repeat(33), "保存しない");
        model.learn("不正\n", "保存しない");
        assert_eq!(model.entry_count(), 1);
        for i in 0..1100 {
            model.learn(&format!("reading{i}"), &format!("surface{i}"));
        }
        assert_eq!(model.entry_count(), 1024);
        model.clear();
        assert_eq!(model.entry_count(), 0);
    }
}
