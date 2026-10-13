// SPDX-License-Identifier: AGPL-3.0-only
use luqnexa_ml::{MODEL_BYTES, Model, Trainer};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fs,
    path::PathBuf,
    time::Instant,
};
struct Sample {
    input: String,
    group_input: Option<String>,
    context: String,
    good: String,
    bad: String,
    split: String,
}
fn shuffle(order: &mut [usize], state: &mut u64) {
    for i in (1..order.len()).rev() {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        order.swap(i, (*state as usize) % (i + 1));
    }
}
// Preserve first-occurrence order; HashMap iteration never controls training.
fn sentence_groups(train: &[&Sample]) -> Vec<Vec<usize>> {
    let mut index = HashMap::new();
    let mut groups = Vec::<Vec<usize>>::new();
    for (i, row) in train.iter().enumerate() {
        let next = groups.len();
        let group = *index
            .entry((
                row.group_input.as_deref().unwrap_or(&row.input),
                row.context.as_str(),
            ))
            .or_insert(next);
        if group == groups.len() {
            groups.push(Vec::new());
        }
        groups[group].push(i);
    }
    groups
}
fn grouped_epoch(
    groups: &[Vec<usize>],
    order: &mut [usize],
    epoch: usize,
    seed: &mut u64,
) -> Vec<usize> {
    shuffle(order, seed);
    order
        .iter()
        .map(|&group| groups[group][(epoch - 1) % groups[group].len()])
        .collect()
}
fn metric(model: &Model, rows: &[Sample], split: &str) -> (usize, usize, f32) {
    let mut wins = 0;
    let mut n = 0;
    let mut loss = 0.0;
    for row in rows.iter().filter(|r| r.split == split) {
        let input = model.prepare(&row.input, &row.context);
        let delta = model.score(&input, &row.good) - model.score(&input, &row.bad);
        wins += usize::from(delta > 0.0);
        n += 1;
        loss += (-delta).max(0.0) + (-delta.abs()).exp().ln_1p();
    }
    (wins, n, loss / n as f32)
}
fn run() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mut data = None;
    let mut output = None;
    let mut epochs = 200_usize;
    let mut seed = 1337_u64;
    let mut interaction = false;
    let mut schedule = "cyclic";
    let mut epoch_decay = 0.0_f32;
    while let Some(arg) = args.next() {
        let value = args.next().ok_or("option needs a value")?;
        match arg.as_str() {
            "--data" => data = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--epochs" => epochs = value.parse()?,
            "--seed" => seed = value.parse()?,
            "--epoch-weight-decay" => epoch_decay = value.parse()?,
            "--schedule" => {
                schedule = match value.as_str() {
                    "cyclic" => "cyclic",
                    "shuffled" => "shuffled",
                    "grouped-shuffled" => "grouped-shuffled",
                    _ => return Err("schedule must be cyclic, shuffled or grouped-shuffled".into()),
                }
            }
            "--architecture" => {
                interaction = match value.as_str() {
                    "v1" => false,
                    "interaction-v2" => true,
                    _ => return Err("architecture must be v1 or interaction-v2".into()),
                }
            }
            _ => return Err("unknown option".into()),
        }
    }
    if !(1..=10_000).contains(&epochs) {
        return Err("epochs must be 1..10000".into());
    }
    if !epoch_decay.is_finite() || !(0.0..=0.1).contains(&epoch_decay) {
        return Err("epoch weight decay must be finite in 0..0.1".into());
    }
    let data = data.ok_or("--data required")?;
    let output =
        output.ok_or("--output required; creates a new run without deleting prior models")?;
    if output.exists() {
        return Err("output already exists; refusing to reset or overwrite existing files".into());
    }
    if fs::metadata(&data)?.len() > 10 * 1024 * 1024 {
        return Err("dataset too large".into());
    }
    let raw = fs::read(&data)?;
    let mut rows = Vec::new();
    let mut inputs = HashMap::new();
    let mut surfaces = HashMap::new();
    let mut seen = HashSet::new();
    for line in std::str::from_utf8(&raw)?.lines().filter(|x| !x.is_empty()) {
        let value: Value = serde_json::from_str(line)?;
        let field = |key: &str| -> Result<String, Box<dyn Error>> {
            let text = value[key].as_str().ok_or("missing string field")?;
            if (text.is_empty() && key != "context")
                || text.chars().count() > 512
                || text.chars().any(char::is_control)
            {
                return Err("invalid sample text".into());
            }
            Ok(text.into())
        };
        let row = Sample {
            input: field("input")?,
            group_input: value
                .get("group_input")
                .map(|_| field("group_input"))
                .transpose()?,
            context: field("context")?,
            good: field("good")?,
            bad: field("bad")?,
            split: field("split")?,
        };
        if !matches!(row.split.as_str(), "train" | "validation" | "test") || row.good == row.bad {
            return Err("invalid split or noncontrastive reference".into());
        }
        if row.good.chars().count() > 96 || row.bad.chars().count() > 96 {
            return Err("candidate exceeds neural input bound".into());
        }
        for (key, map) in [(&row.input, &mut inputs), (&row.good, &mut surfaces)] {
            if map
                .insert(key.clone(), row.split.clone())
                .is_some_and(|prior| prior != row.split)
            {
                return Err("training/evaluation overlap".into());
            }
        }
        if !seen.insert((row.context.clone(), row.good.clone(), row.bad.clone())) {
            return Err("duplicate contrast example".into());
        }
        rows.push(row);
        if rows.len() > 10_000 {
            return Err("too many rows".into());
        }
    }
    // Augmented typos must point to a real, unaugmented training sentence with
    // the same context and answer. Evaluation rows cannot be learning parents.
    let parents: HashSet<_> = rows
        .iter()
        .filter(|r| r.split == "train" && r.group_input.is_none())
        .map(|r| (r.input.as_str(), r.context.as_str(), r.good.as_str()))
        .collect();
    for row in &rows {
        if let Some(parent) = row.group_input.as_deref() {
            if row.split != "train"
                || !parents.contains(&(parent, row.context.as_str(), row.good.as_str()))
            {
                return Err("invalid augmentation parent; requires matching training input, context and answer".into());
            }
        }
    }
    for split in ["train", "validation", "test"] {
        if !rows.iter().any(|r| r.split == split) {
            return Err("all three splits are required".into());
        }
    }
    let started = Instant::now();
    let mut trainer = if interaction {
        Trainer::new_interaction(seed)
    } else {
        Trainer::new(seed)
    };
    let initial = trainer.quantize();
    let mut best = initial.to_bytes();
    let mut best_metric = metric(&initial, &rows, "validation");
    let mut selected = 0;
    let mut trace = Vec::new();
    let train = rows
        .iter()
        .filter(|r| r.split == "train")
        .collect::<Vec<_>>();
    let mut order: Vec<_> = (0..train.len()).collect();
    let groups = sentence_groups(&train);
    let mut group_order: Vec<_> = (0..groups.len()).collect();
    let mut order_seed = (seed ^ 0x9e3779b97f4a7c15).max(1);
    for epoch in 1..=epochs {
        if schedule == "shuffled" {
            shuffle(&mut order, &mut order_seed);
        }
        let epoch_rows = if schedule == "grouped-shuffled" {
            grouped_epoch(&groups, &mut group_order, epoch, &mut order_seed)
        } else {
            (0..train.len())
                .map(|offset| {
                    if schedule == "shuffled" {
                        order[offset]
                    } else {
                        (offset + epoch) % train.len()
                    }
                })
                .collect()
        };
        for row_index in epoch_rows {
            let row = train[row_index];
            trainer.learn(&row.input, &row.context, &row.good, &row.bad, 0.1)?;
        }
        trainer.decay_weights(epoch_decay)?;
        let model = trainer.quantize();
        let result = metric(&model, &rows, "validation");
        if epoch == 1 || epoch % 10 == 0 || epoch == epochs {
            let training = metric(&model, &rows, "train");
            let float_loss = train
                .iter()
                .map(|row| trainer.loss(&row.input, &row.context, &row.good, &row.bad))
                .sum::<f32>()
                / train.len() as f32;
            trace.push(json!({"epoch":epoch,"quantized_train_wins":training.0,
                "quantized_train_loss":training.2,"float_train_loss":float_loss,
                "validation_wins":result.0,"validation_loss":result.2}));
        }
        if result.0 > best_metric.0 || (result.0 == best_metric.0 && result.2 < best_metric.2) {
            best = model.to_bytes();
            best_metric = result;
            selected = epoch;
        }
    }
    let model = Model::from_bytes(&best)?;
    let before: Value = ["train", "validation", "test"]
        .into_iter()
        .map(|split| {
            let (wins, n, loss) = metric(&initial, &rows, split);
            (
                split.to_owned(),
                json!({"wins":wins,"examples":n,"loss":loss}),
            )
        })
        .collect();
    let after: Value = ["train", "validation", "test"]
        .into_iter()
        .map(|split| {
            let (wins, n, loss) = metric(&model, &rows, split);
            (
                split.to_owned(),
                json!({"wins":wins,"examples":n,"loss":loss}),
            )
        })
        .collect();
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&output)?;
    fs::write(output.join("model.lqml"), &best)?;
    fs::write(
        output.join("training-trace.json"),
        serde_json::to_vec_pretty(&trace)?,
    )?;
    let architecture = if interaction {
        "8192x16 hashed Unicode embeddings; context/surface Hadamard interaction; 48->32 ReLU->1 pairwise neural scorer"
    } else {
        "8192x16 hashed Unicode embeddings; 48->32 ReLU->1 pairwise neural scorer"
    };
    let report = json!({"architecture":architecture, "format":std::str::from_utf8(&best[..8])?, "implementation":"Rust training and int8 runtime; no external model", "seed":seed,"epochs":epochs,"selected_epoch":selected,"fresh_initialization":true,"warm_start":false,
        "selection":"validation only; test evaluated after selection", "model_bytes":MODEL_BYTES,"resident_bytes":model.resident_bytes(),"model_sha256":format!("{:x}",Sha256::digest(&best)),"dataset_sha256":format!("{:x}",Sha256::digest(&raw)),"seconds":started.elapsed().as_secs_f64(),"before":before,"after":after,
        "installed":false,"network_used":false,"limits":"Synthetic candidate-pair evaluation, not full IME accuracy. Bounded mean-pooled character features, not general text generation. No A13 or phone measurement."});
    let report = report_with_schedule(&report, schedule, epoch_decay, &groups, epochs);
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{report}");
    Ok(())
}
fn report_with_schedule(
    report: &Value,
    schedule: &str,
    epoch_decay: f32,
    groups: &[Vec<usize>],
    epochs: usize,
) -> Value {
    let mut report = report.clone();
    report["schedule"] = json!(schedule);
    report["epoch_weight_decay"] = json!(epoch_decay);
    report["epoch_coverage"] = json!(if schedule == "grouped-shuffled" {
        "Each input/context group once; negative contrasts rotate; no evaluation sampling"
    } else {
        "Every training example exactly once; no evaluation sampling"
    });
    report["training_sentence_context_groups"] = json!(groups.len());
    report["largest_group_pairs"] = json!(groups.iter().map(Vec::len).max().unwrap_or(0));
    report["gradient_updates"] = json!(
        epochs
            * if schedule == "grouped-shuffled" {
                groups.len()
            } else {
                groups.iter().map(Vec::len).sum()
            }
    );
    report
}
fn main() {
    if let Err(error) = run() {
        eprintln!("luqnexa-train: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grouped_epochs_balance_sentences_rotate_all_contrasts_and_are_reproducible() {
        let groups = vec![vec![0], vec![1, 2, 3], vec![4, 5]];
        let mut order_a = vec![0, 1, 2];
        let mut order_b = order_a.clone();
        let mut seed_a = 37;
        let mut seed_b = 37;
        let mut seen = HashSet::new();
        let mut visits = [0; 3];
        for epoch in 1..=6 {
            let a = grouped_epoch(&groups, &mut order_a, epoch, &mut seed_a);
            assert_eq!(a, grouped_epoch(&groups, &mut order_b, epoch, &mut seed_b));
            assert_eq!(a.len(), 3);
            for row in a {
                seen.insert(row);
                let group = groups.iter().position(|g| g.contains(&row)).unwrap();
                visits[group] += 1;
            }
        }
        assert_eq!(visits, [6, 6, 6]);
        assert_eq!(seen.len(), 6);
        let samples = [
            Sample {
                input: "same".into(),
                group_input: None,
                context: "one".into(),
                good: "x".into(),
                bad: "y".into(),
                split: "train".into(),
            },
            Sample {
                input: "same".into(),
                group_input: None,
                context: "two".into(),
                good: "y".into(),
                bad: "x".into(),
                split: "train".into(),
            },
            Sample {
                input: "same".into(),
                group_input: None,
                context: "one".into(),
                good: "x".into(),
                bad: "z".into(),
                split: "train".into(),
            },
        ];
        assert_eq!(
            sentence_groups(&samples.iter().collect::<Vec<_>>()),
            vec![vec![0, 2], vec![1]]
        );
        let augmented = Sample {
            input: "typo".into(),
            group_input: Some("same".into()),
            context: "one".into(),
            good: "x".into(),
            bad: "typed".into(),
            split: "train".into(),
        };
        assert_eq!(
            sentence_groups(&[&samples[0], &augmented, &samples[1]]),
            vec![vec![0, 1], vec![2]]
        );
    }
    #[test]
    fn shuffled_epochs_cover_each_example_once_and_repeat_from_the_same_seed() {
        let mut a: Vec<_> = (0..72).collect();
        let mut b = a.clone();
        let mut seed_a = 13;
        let mut seed_b = 13;
        let original = a.clone();
        for _ in 0..20 {
            shuffle(&mut a, &mut seed_a);
            shuffle(&mut b, &mut seed_b);
            assert_eq!(a, b);
            let mut sorted = a.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, original);
        }
        assert_ne!(a, original);
        shuffle(&mut [], &mut seed_a);
        shuffle(&mut [0], &mut seed_a);
    }
}
