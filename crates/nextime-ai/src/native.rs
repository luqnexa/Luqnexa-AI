// SPDX-License-Identifier: AGPL-3.0-only
//! Luqnexa's bounded discriminative model, trained locally from authored pairs.
use std::sync::OnceLock;

#[cfg(not(feature = "research-model"))]
const MODEL: &[u8] = include_bytes!("../../../models/native-model.bin");
#[cfg(feature = "research-model")]
const MODEL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/research-model.bin"));
#[cfg(not(feature = "research-model"))]
const ANCHORS: &str = include_str!("../../../models/native-anchors.tsv");
#[cfg(feature = "research-model")]
const ANCHORS: &str = include_str!(concat!(env!("OUT_DIR"), "/research-anchors.tsv"));

pub fn bonus(context: &str, text: &str) -> i64 {
    // Restrict new weights to reviewed lexical domains. No text generation.
    // New contextual anchors must not broaden context-free corrections. Keep
    // the original domain when no preceding sentence was supplied.
    let anchors = if cfg!(feature = "research-model") && context.trim().is_empty() {
        include_str!("../../../models/native-anchors.tsv")
    } else {
        ANCHORS
    };
    let matches = anchors
        .lines()
        .filter_map(|line| line.split_once('\t'))
        .any(|(a, b)| text.contains(a) || text.contains(b));
    if !matches {
        return 0;
    }
    static WEIGHTS: OnceLock<Vec<i16>> = OnceLock::new();
    let weights =
        WEIGHTS.get_or_init(|| super::contrast::decode(MODEL).expect("validated native model"));
    let counts = super::contrast::counts(context, text);
    let norm = counts
        .iter()
        .map(|&n| f64::from(n).powi(2))
        .sum::<f64>()
        .sqrt()
        .max(1.0);
    let dot: f64 = counts
        .iter()
        .zip(weights)
        .map(|(&n, &w)| f64::from(n) * f64::from(w))
        .sum();
    (2048.0 + dot / norm / 1024.0 * 1024.0).clamp(0.0, 4096.0) as i64
}

/// One adjacent QWERTY-key substitution or adjacent transposition. Bounded and
/// deterministic; dictionary verification and literal preservation happen upstream.
pub fn typo_readings(input: &str) -> Vec<String> {
    if !(4..=24).contains(&input.len())
        || !input.bytes().any(|b| b.is_ascii_lowercase())
        || !input
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b";[]'".contains(&b))
    {
        return Vec::new();
    }
    const ROWS: &[(&str, f32)] = &[
        ("qwertyuiop[]", 0.0),
        ("asdfghjkl;'", 0.25),
        ("zxcvbnm", 0.75),
    ];
    let keys: Vec<_> = ROWS
        .iter()
        .enumerate()
        .flat_map(|(y, (row, offset))| {
            row.bytes()
                .enumerate()
                .map(move |(x, b)| (b, x as f32 + offset, y as i32))
        })
        .collect();
    let bytes = input.as_bytes();
    let mut result = Vec::new();
    // A hand starting one key to the left/right produces a whole-row shift.
    // Offer both inverse shifts only when every key has a valid neighbor.
    for direction in [-1isize, 1] {
        let shifted: Option<String> = bytes
            .iter()
            .map(|&b| {
                ROWS.iter().find_map(|(row, _)| {
                    row.bytes().position(|key| key == b).and_then(|index| {
                        row.as_bytes()
                            .get(index.checked_add_signed(direction)?)
                            .copied()
                            .map(char::from)
                    })
                })
            })
            .collect();
        if let Some(shifted) = shifted {
            result.push(shifted);
        }
    }
    for i in 0..bytes.len() {
        // Extra letter or accidental repeated key; dictionary-checked upstream.
        let mut deleted = bytes.to_vec();
        deleted.remove(i);
        result.push(String::from_utf8(deleted).unwrap());
        if i + 1 < bytes.len() && bytes[i] != bytes[i + 1] {
            let mut changed = bytes.to_vec();
            changed.swap(i, i + 1);
            result.push(String::from_utf8(changed).unwrap());
        }
        let (_, x, y) = keys.iter().find(|(b, _, _)| *b == bytes[i]).unwrap();
        for &(b, other_x, other_y) in &keys {
            if b != bytes[i] && (other_y - y).abs() <= 1 && (other_x - x).abs() <= 1.05 {
                let mut changed = bytes.to_vec();
                changed[i] = b;
                result.push(String::from_utf8(changed).unwrap());
            }
        }
    }
    result.sort();
    result.dedup();
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn keyboard_neighbors_and_transpositions_are_bounded() {
        assert!(typo_readings("pythpn").iter().any(|s| s == "python"));
        assert!(typo_readings("pyhton").iter().any(|s| s == "python"));
        assert!(typo_readings("pythonn").iter().any(|s| s == "python"));
        assert!(typo_readings("jr;;p").iter().any(|s| s == "hello"));
        assert!(typo_readings("gwsr").iter().any(|s| s == "hedt"));
        assert!(typo_readings("PYTHON").is_empty());
        assert!(typo_readings("あいうえ").is_empty());
        assert!(typo_readings(&"a".repeat(25)).is_empty());
        assert!(typo_readings(&"a".repeat(24)).len() <= 24 * 10 + 2);
    }
    #[test]
    fn native_weights_distinguish_tools_and_crossings() {
        assert!(bonus("", "箸でご飯を食べる") > bonus("", "橋でご飯を食べる"));
        assert!(bonus("", "橋を渡る") > bonus("", "箸を渡る"));
        assert_eq!(bonus("", "任天堂"), 0);
    }
}
