// SPDX-License-Identifier: AGPL-3.0-only
//! Small locally trained character n-gram ranking model. No I/O or telemetry.
use nextime_ranker::{Candidate, rank};
use std::{collections::HashMap, sync::OnceLock};
pub mod assist;
mod casing;
mod contrast;
mod local;
#[cfg(feature = "mobile-neural")]
pub mod mobile;
mod native;
pub use casing::case_candidates;
pub use native::typo_readings;
mod personal;
pub use local::{LocalConfigError, LocalLlm};
pub use personal::PersonalModel;

/// Identity of the bytes used by native ranking in this compiled binary.
pub fn native_model_identity() -> (&'static str, &'static str, bool) {
    (
        env!("LUQNEXA_COMPILED_MODEL_SHA256"),
        env!("LUQNEXA_COMPILED_ANCHORS_SHA256"),
        cfg!(feature = "research-model"),
    )
}

#[derive(Debug, Default)]
pub struct Model {
    bigrams: HashMap<(char, char), u32>,
    trigrams: HashMap<(char, char, char), u32>,
    public: Vec<([u32; 3], u32)>,
    literature: Vec<([u32; 3], u32)>,
    contrast_enabled: bool,
}

impl Model {
    pub fn train(corpus: &str) -> Self {
        let mut model = Self::default();
        for line in corpus
            .lines()
            .filter(|s| !s.starts_with('#') && !s.is_empty())
        {
            let chars: Vec<_> = ['\0', '\0'].into_iter().chain(line.chars()).collect();
            for pair in chars.windows(2) {
                *model.bigrams.entry((pair[0], pair[1])).or_default() += 1;
            }
            for triple in chars.windows(3) {
                *model
                    .trigrams
                    .entry((triple[0], triple[1], triple[2]))
                    .or_default() += 1;
            }
        }
        model
    }
    pub fn bundled() -> &'static Self {
        static MODEL: OnceLock<Model> = OnceLock::new();
        MODEL.get_or_init(|| {
            let corpus = format!(
                "{}\n{}",
                include_str!("../../../models/context-corpus.txt"),
                include_str!("../../../models/context-synthetic.txt")
            );
            let mut model = Self::train(&corpus);
            model.public =
                Self::decode_public(include_bytes!("../../../models/context-public.bin"))
                    .expect("validated bundled model");
            model.literature =
                Self::decode_public(include_bytes!("../../../models/context-literature.bin"))
                    .expect("validated bundled literary model");
            model.contrast_enabled = true;
            model
        })
    }
    pub fn feature_count(&self) -> usize {
        self.bigrams.len() + self.trigrams.len() + self.public.len() + self.literature.len()
    }

    /// Session-local online learning of confirmed text; stores counts only.
    /// Each table is limited to 4096 features, each count to 64. At capacity,
    /// existing features still update, new features wait until explicit clear.
    pub fn learn_confirmed(&mut self, context: &str, text: &str) {
        if text.is_empty() || text.chars().count() > 256 || text.chars().any(char::is_control) {
            return;
        }
        let mut tail = context.chars().rev();
        let mut b = tail.next().unwrap_or('\0');
        let mut a = tail.next().unwrap_or('\0');
        for c in text.chars() {
            let bi = (b, c);
            if self.bigrams.len() < 4096 || self.bigrams.contains_key(&bi) {
                let count = self.bigrams.entry(bi).or_default();
                *count = count.saturating_add(1).min(64);
            }
            let tri = (a, b, c);
            if self.trigrams.len() < 4096 || self.trigrams.contains_key(&tri) {
                let count = self.trigrams.entry(tri).or_default();
                *count = count.saturating_add(1).min(64);
            }
            a = b;
            b = c;
        }
    }
    pub fn rerank_online(&self, context: &str, candidates: &mut Vec<Candidate>, limit: usize) {
        for candidate in candidates.iter_mut() {
            candidate.score += (self.score(context, &candidate.surface) / 2).min(1024);
        }
        rank(candidates, limit);
    }

    /// Sorted fixed-width public model. No corpus processing on the input path.
    fn decode_public(bytes: &[u8]) -> Result<Vec<([u32; 3], u32)>, &'static str> {
        if !bytes.starts_with(b"NXAI0001") || (bytes.len() - 8) % 16 != 0 {
            return Err("invalid model framing");
        }
        let mut records = Vec::with_capacity((bytes.len() - 8) / 16);
        for record in bytes[8..].chunks_exact(16) {
            let mut words = [0u32; 4];
            for (word, chunk) in words.iter_mut().zip(record.chunks_exact(4)) {
                *word = u32::from_le_bytes(chunk.try_into().unwrap());
            }
            let key = [words[0], words[1], words[2]];
            if words[3] == 0
                || key
                    .iter()
                    .enumerate()
                    .any(|(i, c)| !(i == 0 && *c == u32::MAX) && char::from_u32(*c).is_none())
                || records.last().is_some_and(|(previous, _)| *previous >= key)
            {
                return Err("invalid model record");
            }
            records.push((key, words[3]));
        }
        Ok(records)
    }

    fn public_count(&self, key: [u32; 3]) -> u32 {
        Self::table_count(&self.public, key)
    }
    fn table_count(table: &[([u32; 3], u32)], key: [u32; 3]) -> u32 {
        table
            .binary_search_by_key(&key, |record| record.0)
            .map(|index| table[index].1)
            .unwrap_or(0)
    }

    /// Log-count features learned from the supplied corpus; not calibrated
    /// probabilities. Unknown contexts contribute no bonus.
    pub fn score(&self, context: &str, text: &str) -> i64 {
        let mut tail = context.chars().rev();
        let mut b = tail.next().unwrap_or('\0');
        let mut a = tail.next().unwrap_or('\0');
        let mut total = 0.0;
        let mut public_total = 0.0;
        let mut literature_total = 0.0;
        let mut count = 0;
        for c in text.chars().take(256) {
            let bi = self.bigrams.get(&(b, c)).copied().unwrap_or(0);
            let tri = self.trigrams.get(&(a, b, c)).copied().unwrap_or(0);
            total += f64::from(bi).ln_1p() * 128.0 + f64::from(tri).ln_1p() * 768.0;
            public_total += f64::from(self.public_count([u32::MAX, b as u32, c as u32])).ln_1p()
                * 128.0
                + f64::from(self.public_count([a as u32, b as u32, c as u32])).ln_1p() * 768.0;
            literature_total += f64::from(Self::table_count(
                &self.literature,
                [u32::MAX, b as u32, c as u32],
            ))
            .ln_1p()
                * 128.0
                + f64::from(Self::table_count(
                    &self.literature,
                    [a as u32, b as u32, c as u32],
                ))
                .ln_1p()
                    * 768.0;
            count += 1;
            a = b;
            b = c;
        }
        if count == 0 {
            0
        } else {
            // Keep broad corpus influence bounded while preserving existing
            // conversion scores and the small curated model's corrections.
            (total / f64::from(count)) as i64
                + (public_total / f64::from(count) / 4.0).min(1024.0) as i64
                + if count >= 3 {
                    (literature_total / f64::from(count) / 8.0).min(256.0) as i64
                } else {
                    // Short homophones need specific context/selection data;
                    // broad literary counts can favor a kana fallback instead.
                    0
                }
        }
    }
    /// Diagnostic contribution of the compiled native model; no mutation or I/O.
    pub fn native_bonus(&self, context: &str, text: &str) -> i64 {
        if self.contrast_enabled {
            native::bonus(context, text)
        } else {
            0
        }
    }

    pub fn rerank(&self, context: &str, candidates: &mut Vec<Candidate>, limit: usize) {
        let contrast_enabled =
            self.contrast_enabled && !context.trim().is_empty() && contrast::applicable(candidates);
        for candidate in candidates.iter_mut() {
            candidate.score += self.native_bonus(context, &candidate.surface);
            if !context.is_empty() || candidate.surface.chars().count() >= 3 {
                candidate.score += self.score(context, &candidate.surface).min(4096);
                candidate.score += Self::context_bonus(context, &candidate.surface);
                if contrast_enabled && contrast::matches(&candidate.surface) {
                    // Untrained alternatives retain their original scores.
                    // Only trained-domain comparisons receive a bounded bonus.
                    candidate.score += contrast::score(context, &candidate.surface) + 1536;
                }
            }
        }
        rank(candidates, limit);
    }

    /// Bounded topic hints over the last 96 confirmed characters. Prefer the
    /// most recent cue among competing senses; do not synthesize candidate text.
    fn context_bonus(context: &str, text: &str) -> i64 {
        if context.is_empty() {
            return 0;
        }
        let start = context.char_indices().rev().nth(95).map_or(0, |(i, _)| i);
        let context = &context[start..];
        static ROWS: OnceLock<Vec<(&str, &str)>> = OnceLock::new();
        let rows = ROWS.get_or_init(|| {
            include_str!("../../../models/context-senses.tsv")
                .lines()
                .filter(|line| !line.starts_with('#'))
                .filter_map(|line| line.split_once('\t'))
                .collect()
        });
        // A single global newest topic would make unrelated words compete.
        // Compare only homophones in the explicit sense families.
        const FAMILIES: &[&[&str]] = &[
            &["箸", "橋", "端"],
            &["雨", "飴"],
            &["花", "鼻"],
            &["紙", "髪", "神"],
            &["会う", "合う"],
            &["撮る", "取る"],
            &["正規", "世紀"],
            &["AI", "愛"],
        ];
        for family in FAMILIES {
            let selected = family.iter().find(|surface| {
                text.strip_prefix(**surface).is_some_and(|suffix| {
                    suffix.is_empty()
                        || ["を", "が", "は", "に", "で", "と", "も", "の"]
                            .iter()
                            .any(|particle| suffix.starts_with(particle))
                })
            });
            if let Some(selected) = selected {
                let mut newest = None;
                let mut selected_position = None;
                for (surface, cues) in rows {
                    if family.contains(surface) {
                        for cue in cues.split(',') {
                            if let Some(position) = context.rfind(cue) {
                                newest =
                                    Some(newest.map_or(position, |old: usize| old.max(position)));
                                if surface == selected {
                                    selected_position = Some(
                                        selected_position
                                            .map_or(position, |old: usize| old.max(position)),
                                    );
                                }
                            }
                        }
                    }
                }
                return if selected_position.is_some() && selected_position == newest {
                    3072
                } else {
                    0
                };
            }
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn held_out_contrast_evaluation_preserves_candidate_text() {
        let model = Model::bundled();
        let mut before = 0;
        let mut after = 0;
        let mut total = 0;
        for line in include_str!("../../../models/contrast-evaluation.tsv").lines() {
            let fields: Vec<_> = line.split('\t').collect();
            assert_eq!(fields.len(), 5);
            let context = fields[2];
            let good = fields[3];
            let bad = fields[4];
            let baseline = |text: &str| {
                model.score(context, text).min(4096) + Model::context_bonus(context, text)
            };
            before += usize::from(baseline(good) > baseline(bad) + 100);
            let mut candidates = vec![
                Candidate {
                    surface: good.into(),
                    score: 0,
                    segments: 1,
                },
                Candidate {
                    surface: bad.into(),
                    score: 100,
                    segments: 1,
                },
            ];
            model.rerank(context, &mut candidates, 2);
            after += usize::from(candidates[0].surface == good);
            assert_eq!(candidates.len(), 2);
            assert!(
                candidates
                    .iter()
                    .all(|c| c.surface == good || c.surface == bad)
            );
            total += 1;
        }
        println!(
            "Synthetic held-out plus regression pairs: before={before}/{total}, after={after}/{total}; wrong candidate base advantage=100; not real-world IME accuracy"
        );
        assert!(
            after > before,
            "contrast training must improve this fixed evaluation"
        );
    }
    #[test]
    fn contrast_training_does_not_change_empty_context_ranking() {
        let model = Model::bundled();
        let mut candidates = vec![
            Candidate {
                surface: "AIによる".into(),
                score: 100,
                segments: 1,
            },
            Candidate {
                surface: "愛による".into(),
                score: 0,
                segments: 1,
            },
        ];
        let expected: Vec<_> = candidates
            .iter()
            .map(|c| c.score + model.score("", &c.surface).min(4096))
            .collect();
        model.rerank("", &mut candidates, 2);
        for candidate in candidates {
            let index = usize::from(candidate.surface == "愛による");
            assert_eq!(candidate.score, expected[index]);
        }
    }
    #[test]
    fn previous_sentence_topics_survive_sentence_end_and_follow_topic_changes() {
        let model = Model::default();
        for (context, expected) in [
            ("夕食の準備ができました。これから使うのは", "箸"),
            ("川の向こうに駅があります。これから渡るのは", "橋"),
            ("食事の話は終わりです。川を見に行きます。これは", "橋"),
        ] {
            let mut candidates = vec![
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
            model.rerank(context, &mut candidates, 2);
            assert_eq!(candidates[0].surface, expected);
            assert_eq!(candidates.len(), 2);
        }
        assert_eq!(Model::context_bonus("夕食", "橋本"), 0);
        assert_eq!(Model::context_bonus("夕食", "箸"), 3072);
        assert_eq!(Model::context_bonus("夕食", "箸を使う"), 3072);
        assert_eq!(Model::context_bonus("夕食の準備。", "箸"), 3072);
        assert_eq!(
            Model::context_bonus(&format!("夕食{}", "あ".repeat(96)), "箸"),
            0
        );
        assert_eq!(Model::context_bonus("話題不明", "箸"), 0);
    }
    #[test]
    fn online_learning_is_bounded_and_learns_commit_boundaries() {
        let mut model = Model::default();
        model.learn_confirmed("食事に", "箸を使います。");
        assert!(model.score("食事に", "箸") > model.score("食事に", "橋"));
        let before = model.feature_count();
        model.learn_confirmed("", "invalid\ntext");
        model.learn_confirmed("", &"長".repeat(257));
        assert_eq!(model.feature_count(), before);
        for i in 0..10_000 {
            model.learn_confirmed("", &format!("{}例文", char::from_u32(0x4E00 + i).unwrap()));
        }
        assert!(model.feature_count() <= 8192);
        for _ in 0..100 {
            model.learn_confirmed("食事に", "箸");
        }
        assert!(
            model
                .bigrams
                .values()
                .chain(model.trigrams.values())
                .all(|c| *c <= 64)
        );
    }
    #[test]
    fn validates_public_model_and_loads_learned_features() {
        assert!(Model::decode_public(b"").is_err());
        assert!(Model::decode_public(b"NXAI0001x").is_err());
        let mut malformed = b"NXAI0001".to_vec();
        for word in [0u32, 0, 0xD800, 1] {
            malformed.extend_from_slice(&word.to_le_bytes());
        }
        assert!(Model::decode_public(&malformed).is_err());
        let model = Model::bundled();
        assert!(model.feature_count() > 100_000);
        assert!(model.public_count([u32::MAX, 'で' as u32, 'す' as u32]) > 0);
        assert!(model.literature.len() > 10_000);
        assert!(Model::table_count(&model.literature, [u32::MAX, 't' as u32, 'h' as u32]) > 0);
        assert_eq!(model.score("", ""), 0);
    }
    #[test]
    fn learns_context_without_changing_candidate_text() {
        let model = Model::train("食事に箸を使う。\n川に橋を架ける。\n食事に箸を使います。");
        let original = vec![
            Candidate {
                surface: "橋".into(),
                score: 0,
                segments: 1,
            },
            Candidate {
                surface: "箸".into(),
                score: -10,
                segments: 1,
            },
        ];
        let mut food = original.clone();
        model.rerank("食事に", &mut food, 2);
        assert_eq!(food[0].surface, "箸");
        let mut river = original;
        model.rerank("川に", &mut river, 2);
        assert_eq!(river[0].surface, "橋");
        assert_eq!(model.score("", ""), 0);
        assert!(model.feature_count() > 0);
    }
}
