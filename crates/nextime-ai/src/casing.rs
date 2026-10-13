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

/// Registered spellings only. Does not synthesize title/upper case or change scores.
/// Appends at most 16 alternatives; caller bounds and ranks the combined pool.
pub fn dictionary_case_candidates(candidates: &mut Vec<Candidate>) {
    let originals: Vec<_> = candidates.iter().take(16).cloned().collect();
    for original in originals {
        if let Some(surface) = canonical_name_text(&original.surface) {
            if !candidates.iter().any(|c| c.surface == surface) {
                candidates.push(Candidate {
                    surface,
                    score: original.score.saturating_sub(1),
                    segments: original.segments,
                });
            }
        }
    }
}

fn canonical_name_text(text: &str) -> Option<String> {
    if text.contains(['/', '\\', '@', '_', '`', ':', '=', '{', '}', '.']) {
        return None;
    }
    let mut canonical = String::new();
    for piece in text.split_inclusive(|c: char| !c.is_ascii_alphanumeric()) {
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
        canonical.push_str(
            names()
                .get(&word[..name_end].to_ascii_lowercase())
                .copied()
                .unwrap_or(&word[..name_end]),
        );
        canonical.push_str(&word[name_end..]);
        canonical.push_str(&piece[end..]);
    }
    (canonical != text).then_some(canonical)
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
    fn registered_case_alternatives_work_with_a_full_pool_and_preserve_identifiers() {
        let mut candidates = (0..32)
            .map(|i| Candidate {
                surface: format!("example{i}"),
                score: 100 - i,
                segments: 1,
            })
            .collect::<Vec<_>>();
        candidates[0].surface = "githubで共有する".into();
        dictionary_case_candidates(&mut candidates);
        assert_eq!(candidates.len(), 33);
        assert_eq!(candidates[0].surface, "githubで共有する");
        assert_eq!(candidates[32].surface, "GitHubで共有する");
        assert_eq!(candidates[32].score, 99);
        let before = candidates.clone();
        dictionary_case_candidates(&mut candidates);
        assert_eq!(candidates, before);
        for text in [
            "user_github",
            "https://github.com/Case",
            "unknownterm",
            "file.py",
        ] {
            let mut values = vec![Candidate {
                surface: text.into(),
                score: 0,
                segments: 1,
            }];
            dictionary_case_candidates(&mut values);
            assert_eq!(values.len(), 1);
        }
    }
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
