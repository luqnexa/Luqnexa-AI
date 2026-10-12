// SPDX-License-Identifier: AGPL-3.0-only
//! Luqnexa's bounded discriminative model, trained locally from authored pairs.
use std::sync::OnceLock;

pub fn bonus(context: &str, text: &str) -> i64 {
    // Restrict new weights to reviewed lexical domains. No text generation.
    let matches = include_str!("../../../models/native-anchors.tsv")
        .lines()
        .filter_map(|line| line.split_once('\t'))
        .any(|(a, b)| text.contains(a) || text.contains(b));
    if !matches {
        return 0;
    }
    static WEIGHTS: OnceLock<Vec<i16>> = OnceLock::new();
    let weights = WEIGHTS.get_or_init(|| {
        super::contrast::decode(include_bytes!("../../../models/native-model.bin"))
            .expect("validated native model")
    });
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
    if !(4..=24).contains(&input.len()) || !input.bytes().all(|b| b.is_ascii_lowercase()) {
        return Vec::new();
    }
    const ROWS: &[(&str, f32)] = &[("qwertyuiop", 0.0), ("asdfghjkl", 0.25), ("zxcvbnm", 0.75)];
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
    for i in 0..bytes.len() {
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
        assert!(typo_readings("PYTHON").is_empty());
        assert!(typo_readings("あいうえ").is_empty());
        assert!(typo_readings(&"a".repeat(25)).is_empty());
        assert!(typo_readings(&"a".repeat(24)).len() <= 24 * 9);
    }
    #[test]
    fn native_weights_distinguish_tools_and_crossings() {
        assert!(bonus("", "箸でご飯を食べる") > bonus("", "橋でご飯を食べる"));
        assert!(bonus("", "橋を渡る") > bonus("", "箸を渡る"));
        assert_eq!(bonus("", "任天堂"), 0);
    }
}
