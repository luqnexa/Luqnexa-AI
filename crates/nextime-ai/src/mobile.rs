// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit adapter for the new Rust model. No implicit installation or loading.
use luqnexa_ml::Model;
use nextime_ranker::{Candidate, rank};

pub struct NeuralRanker {
    model: Model,
}
impl NeuralRanker {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        Ok(Self {
            model: Model::from_bytes(bytes)?,
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
        for candidate in candidates.iter_mut() {
            let bonus =
                (self.model.score(&input, &candidate.surface).tanh() * 2048.0).round() as i64;
            candidate.score = candidate.score.saturating_add(bonus);
        }
        rank(candidates, limit);
        Ok(())
    }
    pub fn resident_bytes(&self) -> usize {
        self.model.resident_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
