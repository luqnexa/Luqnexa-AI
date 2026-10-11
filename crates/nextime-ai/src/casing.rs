// SPDX-License-Identifier: AGPL-3.0-only
//! Bounded spelling alternatives; the local model selects, never generates text.
use nextime_ranker::Candidate;
use std::{collections::HashMap, sync::OnceLock};

fn names() -> &'static HashMap<String, &'static str> {
    static NAMES: OnceLock<HashMap<String, &'static str>> = OnceLock::new();
    NAMES.get_or_init(|| {
        include_str!("../../../dictionaries/proper-nouns.tsv")
            .lines()
            .filter(|line| !line.starts_with('#'))
            .filter_map(|line| line.split('\t').nth(1))
            .filter(|name| {
                name.bytes().all(|b| b.is_ascii_alphanumeric())
                    && name.bytes().any(|b| b.is_ascii_uppercase())
            })
            .map(|name| (name.to_ascii_lowercase(), name))
            .collect()
    })
}

pub fn case_candidates(candidates: &mut Vec<Candidate>, limit: usize) {
    let originals: Vec<_> = candidates.iter().take(3).cloned().collect();
    for original in originals {
        // Leave identifiers, URLs and paths literal. They can be case sensitive.
        if original
            .surface
            .contains(['/', '\\', '@', '_', '`', ':', '=', '{', '}', '.'])
        {
            continue;
        }
        let mut canonical = String::new();
        let mut known_name = false;
        for piece in original
            .surface
            .split_inclusive(|c: char| !c.is_ascii_alphanumeric())
        {
            let end = piece
                .find(|c: char| !c.is_ascii_alphanumeric())
                .unwrap_or(piece.len());
            let word = &piece[..end];
            let number = word
                .find(|c: char| c.is_ascii_digit())
                .unwrap_or(word.len());
            let name_end = if word[number..].bytes().all(|b| b.is_ascii_digit()) {
                number
            } else {
                word.len()
            };
            known_name |= names().contains_key(&word[..name_end].to_ascii_lowercase());
            canonical.push_str(
                names()
                    .get(&word[..name_end].to_ascii_lowercase())
                    .copied()
                    .unwrap_or(&word[..name_end]),
            );
            canonical.push_str(&word[name_end..]);
            canonical.push_str(&piece[end..]);
        }
        let mut start = true;
        let title: String = original
            .surface
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() {
                    let result = if start {
                        c.to_ascii_uppercase()
                    } else {
                        c.to_ascii_lowercase()
                    };
                    start = false;
                    result
                } else {
                    start = true;
                    c
                }
            })
            .collect();
        let variants = [
            canonical,
            original.surface.to_ascii_lowercase(),
            original.surface.to_ascii_uppercase(),
            title,
        ];
        for (variant, surface) in variants.into_iter().enumerate() {
            // A naive title spelling such as Chatgpt or Github is not an
            // alternative to the dictionary's exact product spelling.
            if variant == 3 && known_name {
                continue;
            }
            if candidates.len() >= limit.min(12) {
                return;
            }
            if !candidates.iter().any(|c| c.surface == surface) {
                candidates.push(Candidate {
                    surface,
                    score: candidates
                        .last()
                        .map_or(original.score, |c| c.score)
                        .saturating_sub(1),
                    segments: original.segments,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn names_gain_case_choices_and_sensitive_identifiers_stay_literal() {
        for (input, expected) in [
            ("chatgptで書く", "ChatGPTで書く"),
            ("vscode", "VSCode"),
            ("github", "GitHub"),
            ("apple", "Apple"),
            ("iphone2026", "iPhone2026"),
        ] {
            let mut choices = vec![Candidate {
                surface: input.into(),
                score: 10,
                segments: 1,
            }];
            case_candidates(&mut choices, 8);
            assert!(choices.iter().any(|c| c.surface == expected), "{input}");
            assert_eq!(choices[0].surface, input);
            assert!(
                choices
                    .iter()
                    .all(|c| c.surface.eq_ignore_ascii_case(input))
            );
        }
        for input in [
            "user_ID",
            "https://example.com/Case",
            "file.rs",
            "me@example.com",
        ] {
            let mut choices = vec![Candidate {
                surface: input.into(),
                score: 0,
                segments: 1,
            }];
            case_candidates(&mut choices, 8);
            assert_eq!(choices.len(), 1);
        }
    }
}
