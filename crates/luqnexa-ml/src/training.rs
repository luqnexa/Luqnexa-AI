// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
pub struct Trainer {
    interactions: bool,
    embeddings: Box<[[f32; EMBEDDING]]>,
    hidden: [[f32; INPUT]; HIDDEN],
    biases: [f32; HIDDEN],
    output: [f32; HIDDEN],
}

struct Forward {
    features: Features,
    input: [f32; INPUT],
    raw_input: [f32; INPUT],
    hidden: [f32; HIDDEN],
    score: f32,
}
impl Trainer {
    /// A fresh deterministic initialization, never an implicit warm start.
    pub fn new(seed: u64) -> Self {
        Self::initialize(seed, false)
    }
    /// Experimental interaction architecture, explicitly selected by the caller.
    pub fn new_interaction(seed: u64) -> Self {
        Self::initialize(seed, true)
    }
    /// Explicit legacy initialization for reproducibility, never a warm start.
    pub fn new_legacy(seed: u64) -> Self {
        Self::initialize(seed, false)
    }
    fn initialize(seed: u64, interactions: bool) -> Self {
        let mut state = seed.max(1);
        let mut random = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / ((1_u32 << 24) as f32) * 2.0 - 1.0) * 0.1
        };
        let embeddings = (0..BUCKETS)
            .map(|_| std::array::from_fn(|_| random()))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let hidden = std::array::from_fn(|_| std::array::from_fn(|_| random()));
        let output = std::array::from_fn(|_| random());
        Self {
            interactions,
            embeddings,
            hidden,
            biases: [0.0; HIDDEN],
            output,
        }
    }
    fn forward(&self, reading: &str, context: &str, surface: &str) -> Forward {
        let features = Features::new(reading, context, surface);
        let mut input = [0.0; INPUT];
        for channel in 0..3 {
            let n = features.lengths[channel];
            for &id in &features.ids[channel][..n] {
                for i in 0..EMBEDDING {
                    input[channel * EMBEDDING + i] +=
                        self.embeddings[usize::from(id)][i] / n as f32;
                }
            }
        }
        let raw_input = input;
        if self.interactions {
            interact(&mut input);
        }
        let hidden = std::array::from_fn(|j| {
            (self.biases[j]
                + self.hidden[j]
                    .iter()
                    .zip(input)
                    .map(|(w, x)| w * x)
                    .sum::<f32>())
            .max(0.0)
        });
        let score = hidden.iter().zip(self.output).map(|(x, w)| x * w).sum();
        Forward {
            features,
            input,
            raw_input,
            hidden,
            score,
        }
    }
    pub fn loss(&self, reading: &str, context: &str, good: &str, bad: &str) -> f32 {
        let delta =
            self.forward(reading, context, good).score - self.forward(reading, context, bad).score;
        (-delta).max(0.0) + (-delta.abs()).exp().ln_1p()
    }
    pub fn learn(
        &mut self,
        reading: &str,
        context: &str,
        good: &str,
        bad: &str,
        rate: f32,
    ) -> Result<(), &'static str> {
        if !rate.is_finite() || !(0.0..=1.0).contains(&rate) || rate == 0.0 || good == bad {
            return Err("invalid training sample or learning rate");
        }
        let a = self.forward(reading, context, good);
        let b = self.forward(reading, context, bad);
        let gradient = -1.0 / (1.0 + (a.score - b.score).clamp(-30.0, 30.0).exp());
        let mut ga = [0.0; HIDDEN];
        let mut gb = [0.0; HIDDEN];
        let mut ia = [0.0; INPUT];
        let mut ib = [0.0; INPUT];
        for j in 0..HIDDEN {
            ga[j] = if a.hidden[j] > 0.0 {
                gradient * self.output[j]
            } else {
                0.0
            };
            gb[j] = if b.hidden[j] > 0.0 {
                -gradient * self.output[j]
            } else {
                0.0
            };
            for i in 0..INPUT {
                ia[i] += ga[j] * self.hidden[j][i];
                ib[i] += gb[j] * self.hidden[j][i];
            }
        }
        for j in 0..HIDDEN {
            for i in 0..INPUT {
                self.hidden[j][i] -=
                    rate * (ga[j] * a.input[i] + gb[j] * b.input[i]).clamp(-1.0, 1.0);
            }
            self.biases[j] -= rate * (ga[j] + gb[j]).clamp(-1.0, 1.0);
            self.output[j] -= rate * (gradient * (a.hidden[j] - b.hidden[j])).clamp(-1.0, 1.0);
        }
        for (pass, mut input_gradient) in [(&a, ia), (&b, ib)] {
            if self.interactions {
                for i in 0..EMBEDDING {
                    let product_gradient = input_gradient[EMBEDDING + i];
                    input_gradient[EMBEDDING + i] =
                        product_gradient * pass.raw_input[2 * EMBEDDING + i] * EMBEDDING as f32;
                    input_gradient[2 * EMBEDDING + i] +=
                        product_gradient * pass.raw_input[EMBEDDING + i] * EMBEDDING as f32;
                }
            }
            for channel in 0..3 {
                let n = pass.features.lengths[channel];
                for &id in &pass.features.ids[channel][..n] {
                    for i in 0..EMBEDDING {
                        self.embeddings[usize::from(id)][i] -= rate
                            * input_gradient[channel * EMBEDDING + i].clamp(-1.0, 1.0)
                            / n as f32;
                    }
                }
            }
        }
        Ok(())
    }
    pub fn quantize(&self) -> Model {
        Model {
            interactions: self.interactions,
            embeddings: self
                .embeddings
                .iter()
                .map(Row::quantize)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            hidden: std::array::from_fn(|j| Row::quantize(&self.hidden[j])),
            biases: self.biases,
            output: Row::quantize(&self.output),
            output_bias: 0.0,
        }
    }
    /// Optional offline regularization once per epoch. Biases are not decayed.
    pub fn decay_weights(&mut self, amount: f32) -> Result<(), &'static str> {
        if !amount.is_finite() || !(0.0..=0.1).contains(&amount) {
            return Err("epoch weight decay must be finite in 0..0.1");
        }
        if amount == 0.0 {
            return Ok(());
        }
        let factor = 1.0 - amount;
        for row in self.embeddings.iter_mut() {
            for weight in row {
                *weight *= factor;
            }
        }
        for row in &mut self.hidden {
            for weight in row {
                *weight *= factor;
            }
        }
        for weight in &mut self.output {
            *weight *= factor;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn weight_decay_zero_is_identical_and_invalid_values_do_not_mutate_weights() {
        let mut trainer = Trainer::new_interaction(11);
        let before = trainer.quantize().to_bytes();
        trainer.decay_weights(0.0).unwrap();
        assert_eq!(trainer.quantize().to_bytes(), before);
        for amount in [f32::NAN, f32::INFINITY, -0.001, 0.2] {
            assert!(trainer.decay_weights(amount).is_err());
            assert_eq!(trainer.quantize().to_bytes(), before);
        }
        let weight = trainer.embeddings[0][0];
        trainer.decay_weights(0.001).unwrap();
        assert_eq!(trainer.embeddings[0][0], weight * 0.999);
        assert_ne!(trainer.quantize().to_bytes(), before);
    }
    #[test]
    fn interaction_embedding_gradient_matches_finite_difference() {
        let mut trainer = Trainer::new_interaction(17);
        trainer.hidden = [[0.05; INPUT]; HIDDEN];
        trainer.biases = [0.25; HIDDEN];
        trainer.output = [0.2; HIDDEN];
        let feature = usize::from(Features::new("read", "context", "good").ids[1][0]);
        let before = trainer.embeddings[feature][0];
        let epsilon = 0.01;
        trainer.embeddings[feature][0] = before + epsilon;
        let plus = trainer.loss("read", "context", "good", "bad");
        trainer.embeddings[feature][0] = before - epsilon;
        let minus = trainer.loss("read", "context", "good", "bad");
        trainer.embeddings[feature][0] = before;
        let numerical = (plus - minus) / (2.0 * epsilon);
        let rate = 0.1;
        trainer
            .learn("read", "context", "good", "bad", rate)
            .unwrap();
        let analytical = (before - trainer.embeddings[feature][0]) / rate;
        assert!(numerical.abs() > 0.0001);
        assert!(
            (numerical - analytical).abs() < 0.0001,
            "numerical={numerical}, analytical={analytical}"
        );
    }
}
