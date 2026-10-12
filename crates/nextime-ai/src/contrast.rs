// SPDX-License-Identifier: AGPL-3.0-only
//! Fixed-size locally trained pairwise candidate model. No generated text or I/O.
use std::{collections::BTreeSet, sync::OnceLock};

const DIM: usize = 4096;

/// Only compare domains present in training, with both alternatives actually
/// available. Generic weights must not push unrelated proper names into kana.
pub fn applicable(candidates: &[nextime_ranker::Candidate]) -> bool {
    include_str!("../../../models/contrast-anchors.tsv")
        .lines()
        .any(|line| {
            line.split_once('\t').is_some_and(|(good, bad)| {
                candidates.iter().any(|c| c.surface.contains(good))
                    && candidates.iter().any(|c| c.surface.contains(bad))
            })
        })
}

pub fn matches(text: &str) -> bool {
    include_str!("../../../models/contrast-anchors.tsv")
        .lines()
        .filter_map(|line| line.split_once('\t'))
        .any(|(good, bad)| text.contains(good) || text.contains(bad))
}

fn hash_bytes(hash: u32, bytes: &[u8]) -> u32 {
    bytes.iter().fold(hash, |hash, &byte| {
        (hash ^ u32::from(byte)).wrapping_mul(16777619)
    })
}

fn hash_chars(mut hash: u32, chars: &[char]) -> u32 {
    let mut buffer = [0_u8; 4];
    for c in chars {
        hash = hash_bytes(hash, c.encode_utf8(&mut buffer).as_bytes());
    }
    hash
}

#[cfg(test)]
fn slot(text: &str) -> usize {
    hash_bytes(2166136261, text.as_bytes()) as usize % DIM
}

pub(super) fn counts(context: &str, text: &str) -> Vec<u16> {
    let context: Vec<_> = context
        .chars()
        .rev()
        .take(24)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let text: Vec<_> = text.chars().take(96).collect();
    let mut counts = vec![0_u16; DIM];
    let surface_hash = hash_bytes(2166136261, b"s:");
    for size in 1..=3 {
        for gram in text.windows(size) {
            counts[hash_chars(surface_hash, gram) as usize % DIM] += 1;
        }
    }
    let context_pairs: BTreeSet<_> = context.windows(2).map(|g| [g[0], g[1]]).collect();
    let text_pairs: BTreeSet<_> = text.windows(2).map(|g| [g[0], g[1]]).collect();
    let context_hash = hash_bytes(2166136261, b"c:");
    for left in &context_pairs {
        let prefix = hash_bytes(hash_chars(context_hash, left), b":");
        for right in &text_pairs {
            counts[hash_chars(prefix, right) as usize % DIM] += 1;
        }
    }
    counts
}

pub(super) fn decode(bytes: &[u8]) -> Result<Vec<i16>, &'static str> {
    if bytes.len() != 8 + DIM * 2 || !bytes.starts_with(b"NXCR0001") {
        return Err("invalid contrast model framing");
    }
    Ok(bytes[8..]
        .chunks_exact(2)
        .map(|v| i16::from_le_bytes([v[0], v[1]]))
        .collect())
}

pub fn score(context: &str, text: &str) -> i64 {
    if text.is_empty() {
        return 0;
    }
    static WEIGHTS: OnceLock<Vec<i16>> = OnceLock::new();
    let weights = WEIGHTS.get_or_init(|| {
        decode(include_bytes!("../../../models/context-contrast.bin"))
            .expect("validated bundled contrast model")
    });
    let counts = counts(context, text);
    let norm = counts
        .iter()
        .map(|&v| f64::from(v).powi(2))
        .sum::<f64>()
        .sqrt();
    let dot = counts
        .iter()
        .zip(weights)
        .map(|(&n, &w)| f64::from(n) * f64::from(w))
        .sum::<f64>();
    (dot / norm.max(1.0) / 1024.0 * 768.0).clamp(-1536.0, 1536.0) as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn framing_and_input_bounds() {
        assert!(decode(b"").is_err());
        assert!(decode(b"NXCR0001").is_err());
        assert_eq!(score("", ""), 0);
        assert_eq!(
            score(&format!("古い{}", "あ".repeat(24)), "文章"),
            score(&"あ".repeat(24), "文章")
        );
        assert_eq!(score("", &"あ".repeat(1000)), score("", &"あ".repeat(96)));
    }
    #[test]
    fn fixed_point_features_match_python() {
        assert_eq!(slot("s:がいい"), 1852);
        assert_eq!(score("入力について評価します。", "変換の精度がいい"), 1323);
        assert_eq!(score("文字の読み方を練習します。", "漢字を覚えます"), 1164);
    }
    #[test]
    fn unrelated_names_are_outside_the_training_domain() {
        let candidates =
            ["任天堂", "ニンテンドウ", "にんてんどう"].map(|surface| nextime_ranker::Candidate {
                surface: surface.into(),
                score: 0,
                segments: 1,
            });
        assert!(!applicable(&candidates));
        let greeting = ["Hello", "hello"].map(|surface| nextime_ranker::Candidate {
            surface: surface.into(),
            score: 0,
            segments: 1,
        });
        assert!(!applicable(&greeting));
    }
}
