// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit adapter for the new Rust model. No implicit installation or loading.
use luqnexa_ml::Model;
use nextime_ranker::{Candidate, rank};

pub struct NeuralRanker {
    model: Model,
    temperature: Option<f32>,
}
impl NeuralRanker {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        Ok(Self {
            model: Model::from_bytes(bytes)?,
            temperature: None,
        })
    }
    /// Explicit training-derived score scale; not a calibrated probability.
    pub fn from_bytes_with_temperature(
        bytes: &[u8],
        temperature: f32,
    ) -> Result<Self, &'static str> {
        if !temperature.is_finite() || !(0.0001..=100.0).contains(&temperature) {
            return Err("temperature must be finite and in 0.0001..100");
        }
        Ok(Self {
            model: Model::from_bytes(bytes)?,
            temperature: Some(temperature),
        })
    }
    /// Rank existing candidates only; model influence is bounded to +/-2048.
    /// The host must verify provenance before providing model bytes.
    pub fn rerank(
        &self,
        reading: &str,
        context: &str,
        candidates: &mut Vec<Candidate>,
        limit: usize,
    ) -> Result<(), &'static str> {
        if !(1..=32).contains(&limit) || candidates.len() > 32 {
            return Err("candidate limit exceeded");
        }
        let input = self.model.prepare(reading, context);
        let mut scores = [0.0; 32];
        for (slot, candidate) in scores.iter_mut().zip(candidates.iter()) {
            *slot = self.model.score(&input, &candidate.surface);
        }
        let average = if self.temperature.is_some() && !candidates.is_empty() {
            (scores[..candidates.len()]
                .iter()
                .map(|&s| f64::from(s))
                .sum::<f64>()
                / candidates.len() as f64) as f32
        } else {
            0.0
        };
        for (candidate, score) in candidates.iter_mut().zip(scores) {
            let bonus = score_bonus(score, average, self.temperature.unwrap_or(1.0));
            candidate.score = candidate.score.saturating_add(bonus);
        }
        rank(candidates, limit);
        Ok(())
    }
    pub fn resident_bytes(&self) -> usize {
        self.model.resident_bytes()
    }
}
fn score_bonus(score: f32, average: f32, temperature: f32) -> i64 {
    (((score - average) / temperature).tanh() * 2048.0).round() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_scale_is_bounded_and_invariant_to_common_score_offsets() {
        for score in [-2.0, -0.25, 0.25, 2.0] {
            assert_eq!(
                score_bonus(score, 0.0, 0.5),
                score_bonus(score + 32.0, 32.0, 0.5)
            );
            assert!(score_bonus(score, 0.0, 0.0001).abs() <= 2048);
        }
        let bytes = luqnexa_ml::Trainer::new(7).quantize().to_bytes();
        for scale in [f32::NAN, f32::INFINITY, 0.0, -1.0, 101.0] {
            assert!(NeuralRanker::from_bytes_with_temperature(&bytes, scale).is_err());
        }
        let model = NeuralRanker::from_bytes_with_temperature(&bytes, 0.01).unwrap();
        let mut values = vec![Candidate {
            surface: "原文".into(),
            score: 100,
            segments: 1,
        }];
        model.rerank("genbun", "", &mut values, 1).unwrap();
        assert_eq!(values[0].score, 100);
        assert_eq!(values[0].surface, "原文");
    }
    #[test]
    fn rust_network_adapter_preserves_candidates_and_bounds_influence() {
        let mut trainer = luqnexa_ml::Trainer::new(5);
        for _ in 0..200 {
            trainer.learn("hashi", "夕食", "箸", "橋", 0.1).unwrap();
        }
        let ranker = NeuralRanker::from_bytes(&trainer.quantize().to_bytes()).unwrap();
        let mut candidates = vec![
            Candidate {
                surface: "橋".into(),
                score: 0,
                segments: 1,
            },
            Candidate {
                surface: "箸".into(),
                score: 0,
                segments: 1,
            },
        ];
        ranker.rerank("hashi", "夕食", &mut candidates, 2).unwrap();
        assert_eq!(candidates[0].surface, "箸");
        assert!(candidates.iter().all(|c| c.score.abs() <= 2048));
        assert_eq!(candidates.len(), 2);
        assert!(ranker.rerank("", "", &mut candidates, 0).is_err());
        assert!(ranker.resident_bytes() < 170_000);
    }
}
