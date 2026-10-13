// SPDX-License-Identifier: AGPL-3.0-only
//! Small original neural candidate scorer. Training and int8 inference are Rust.
//! No networking, input persistence, dictionary access or automatic commits.
mod training;
pub use training::Trainer;

pub const BUCKETS: usize = 8192;
pub const EMBEDDING: usize = 16;
pub const HIDDEN: usize = 32;
pub const INPUT: usize = EMBEDDING * 3;
const MAX_FEATURES: usize = 288;
pub const MODEL_BYTES: usize = 20 + BUCKETS * (4 + EMBEDDING) + HIDDEN * (8 + INPUT) + 8 + HIDDEN;

#[derive(Clone)]
pub(crate) struct Features {
    pub ids: [[u16; MAX_FEATURES]; 3],
    pub lengths: [usize; 3],
}
impl Features {
    pub fn new(reading: &str, context: &str, surface: &str) -> Self {
        let mut features = Self {
            ids: [[0; MAX_FEATURES]; 3],
            lengths: [0; 3],
        };
        for (channel, text, bound) in [(0, reading, 64), (1, context, 48), (2, surface, 96)] {
            // Context uses the most recent characters, not the start of history.
            let mut chars = ['\0'; 96];
            let count = if channel == 1 {
                let n = text
                    .chars()
                    .rev()
                    .take(bound)
                    .enumerate()
                    .map(|(i, c)| {
                        chars[i] = c;
                    })
                    .count();
                chars[..n].reverse();
                n
            } else {
                text.chars()
                    .take(bound)
                    .enumerate()
                    .map(|(i, c)| {
                        chars[i] = c;
                    })
                    .count()
            };
            for width in 1..=3 {
                for start in 0..count.saturating_sub(width - 1) {
                    let mut hash = 2166136261_u32 ^ channel as u32;
                    for c in &chars[start..start + width] {
                        for byte in (*c as u32).to_le_bytes() {
                            hash = (hash ^ u32::from(byte)).wrapping_mul(16777619);
                        }
                    }
                    let n = features.lengths[channel];
                    features.ids[channel][n] = (hash as usize % BUCKETS) as u16;
                    features.lengths[channel] += 1;
                }
            }
        }
        features
    }
}

#[derive(Clone, Copy, Debug)]
struct Row<const N: usize> {
    scale: f32,
    values: [i8; N],
}
impl<const N: usize> Row<N> {
    fn quantize(values: &[f32; N]) -> Self {
        let scale = (values.iter().map(|x| x.abs()).fold(0.0_f32, f32::max) / 127.0).max(1e-12);
        Self {
            scale,
            values: std::array::from_fn(|i| (values[i] / scale).round().clamp(-127.0, 127.0) as i8),
        }
    }
    fn value(&self, i: usize) -> f32 {
        f32::from(self.values[i]) * self.scale
    }
}

#[derive(Debug)]
pub struct Model {
    embeddings: Box<[Row<EMBEDDING>]>,
    hidden: [Row<INPUT>; HIDDEN],
    biases: [f32; HIDDEN],
    output: Row<HIDDEN>,
    output_bias: f32,
}
#[derive(Clone, Copy, Debug)]
pub struct PreparedInput {
    values: [f32; INPUT],
}

impl Model {
    pub fn prepare(&self, reading: &str, context: &str) -> PreparedInput {
        let f = Features::new(reading, context, "");
        let mut values = [0.0; INPUT];
        self.embed(&f, 0, &mut values);
        self.embed(&f, 1, &mut values);
        PreparedInput { values }
    }
    fn embed(&self, features: &Features, channel: usize, values: &mut [f32; INPUT]) {
        let n = features.lengths[channel];
        if n == 0 {
            return;
        }
        for &id in &features.ids[channel][..n] {
            let row = &self.embeddings[usize::from(id)];
            for i in 0..EMBEDDING {
                values[channel * EMBEDDING + i] += row.value(i) / n as f32;
            }
        }
    }
    /// Allocation-free after model loading and input preparation. Higher is better.
    pub fn score(&self, input: &PreparedInput, surface: &str) -> f32 {
        let f = Features::new("", "", surface);
        let mut values = input.values;
        self.embed(&f, 2, &mut values);
        let mut result = self.output_bias;
        for j in 0..HIDDEN {
            let sum = self.biases[j]
                + values
                    .iter()
                    .enumerate()
                    .map(|(i, x)| x * self.hidden[j].value(i))
                    .sum::<f32>();
            result += sum.max(0.0) * self.output.value(j);
        }
        result
    }
    pub fn resident_bytes(&self) -> usize {
        self.embeddings.len() * std::mem::size_of::<Row<EMBEDDING>>() + std::mem::size_of::<Self>()
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(MODEL_BYTES);
        bytes.extend_from_slice(b"LQML0001");
        for n in [BUCKETS, EMBEDDING, HIDDEN] {
            bytes.extend_from_slice(&(n as u32).to_le_bytes());
        }
        fn row<const N: usize>(bytes: &mut Vec<u8>, row: &Row<N>) {
            bytes.extend_from_slice(&row.scale.to_le_bytes());
            bytes.extend(row.values.iter().map(|x| *x as u8));
        }
        for embedding in &self.embeddings {
            row(&mut bytes, embedding);
        }
        for j in 0..HIDDEN {
            row(&mut bytes, &self.hidden[j]);
            bytes.extend_from_slice(&self.biases[j].to_le_bytes());
        }
        row(&mut bytes, &self.output);
        bytes.extend_from_slice(&self.output_bias.to_le_bytes());
        bytes
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() != MODEL_BYTES || &bytes[..8] != b"LQML0001" {
            return Err("invalid model format or size");
        }
        for (i, n) in [BUCKETS, EMBEDDING, HIDDEN].into_iter().enumerate() {
            if u32::from_le_bytes(bytes[8 + i * 4..12 + i * 4].try_into().unwrap()) != n as u32 {
                return Err("invalid model dimensions");
            }
        }
        let mut cursor = 20;
        fn scalar(bytes: &[u8], cursor: &mut usize) -> Result<f32, &'static str> {
            let value = f32::from_le_bytes(bytes[*cursor..*cursor + 4].try_into().unwrap());
            *cursor += 4;
            if !value.is_finite() || value.abs() > 1e6 {
                return Err("invalid model scalar");
            }
            Ok(value)
        }
        fn row<const N: usize>(bytes: &[u8], cursor: &mut usize) -> Result<Row<N>, &'static str> {
            let scale = scalar(bytes, cursor)?;
            if scale <= 0.0 {
                return Err("invalid quantization scale");
            }
            let values = std::array::from_fn(|i| bytes[*cursor + i] as i8);
            *cursor += N;
            Ok(Row { scale, values })
        }
        let mut embeddings = Vec::with_capacity(BUCKETS);
        for _ in 0..BUCKETS {
            embeddings.push(row(bytes, &mut cursor)?);
        }
        let mut hidden = [Row {
            scale: 1.0,
            values: [0; INPUT],
        }; HIDDEN];
        let mut biases = [0.0; HIDDEN];
        for j in 0..HIDDEN {
            hidden[j] = row(bytes, &mut cursor)?;
            biases[j] = scalar(bytes, &mut cursor)?;
        }
        let output = row(bytes, &mut cursor)?;
        let output_bias = scalar(bytes, &mut cursor)?;
        Ok(Self {
            embeddings: embeddings.into_boxed_slice(),
            hidden,
            biases,
            output,
            output_bias,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_roundtrip_is_fixed_size_and_corruption_is_rejected() {
        let model = Trainer::new(7).quantize();
        let mut bytes = model.to_bytes();
        assert_eq!(bytes.len(), MODEL_BYTES);
        let loaded = Model::from_bytes(&bytes).unwrap();
        let input = model.prepare("hashi", "食事");
        assert_eq!(model.score(&input, "箸"), loaded.score(&input, "箸"));
        assert!(model.resident_bytes() < 170_000);
        bytes[20..24].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(Model::from_bytes(&bytes).is_err());
        assert!(Model::from_bytes(&[]).is_err());
    }
    #[test]
    fn unicode_and_input_cost_are_bounded() {
        let f = Features::new(&"🙂".repeat(1000), &"語".repeat(1000), &"한국".repeat(1000));
        assert_eq!(f.lengths, [189, 141, 285]);
        let model = Trainer::new(1).quantize();
        assert!(model.score(&model.prepare("", ""), "").is_finite());
    }
    #[test]
    fn rust_training_learns_a_pair_from_fresh_weights() {
        let mut trainer = Trainer::new(5);
        let before = trainer.loss("hashi", "夕食", "箸", "橋");
        for _ in 0..200 {
            trainer.learn("hashi", "夕食", "箸", "橋", 0.1).unwrap();
        }
        assert!(trainer.loss("hashi", "夕食", "箸", "橋") < before);
        let model = trainer.quantize();
        let input = model.prepare("hashi", "夕食");
        assert!(model.score(&input, "箸") > model.score(&input, "橋"));
        assert!(trainer.learn("", "", "a", "b", f32::NAN).is_err());
    }
    #[test]
    fn initialization_and_training_are_reproducible() {
        let mut a = Trainer::new(12);
        let mut b = Trainer::new(12);
        for _ in 0..20 {
            a.learn("ai", "技術", "AI", "愛", 0.1).unwrap();
            b.learn("ai", "技術", "AI", "愛", 0.1).unwrap();
        }
        assert_eq!(a.quantize().to_bytes(), b.quantize().to_bytes());
    }
    #[test]
    fn opposite_contexts_must_learn_opposite_choices() {
        let mut trainer = Trainer::new(1337);
        for _ in 0..2000 {
            trainer
                .learn("same-reading", "meal", "utensil", "bridge", 0.1)
                .unwrap();
            trainer
                .learn("same-reading", "river", "bridge", "utensil", 0.1)
                .unwrap();
        }
        let model = trainer.quantize();
        let meal = model.prepare("same-reading", "meal");
        let river = model.prepare("same-reading", "river");
        assert!(model.score(&meal, "utensil") > model.score(&meal, "bridge"));
        assert!(model.score(&river, "bridge") > model.score(&river, "utensil"));
    }
}
