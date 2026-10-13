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
    context: String,
    good: String,
    bad: String,
    split: String,
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
    while let Some(arg) = args.next() {
        let value = args.next().ok_or("option needs a value")?;
        match arg.as_str() {
            "--data" => data = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--epochs" => epochs = value.parse()?,
            "--seed" => seed = value.parse()?,
            _ => return Err("unknown option".into()),
        }
    }
    if !(1..=10_000).contains(&epochs) {
        return Err("epochs must be 1..10000".into());
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
    for split in ["train", "validation", "test"] {
        if !rows.iter().any(|r| r.split == split) {
            return Err("all three splits are required".into());
        }
    }
    let started = Instant::now();
    let mut trainer = Trainer::new(seed);
    let initial = trainer.quantize();
    let mut best = initial.to_bytes();
    let mut best_metric = metric(&initial, &rows, "validation");
    let mut selected = 0;
    let train = rows
        .iter()
        .filter(|r| r.split == "train")
        .collect::<Vec<_>>();
    for epoch in 1..=epochs {
        for offset in 0..train.len() {
            let row = train[(offset + epoch) % train.len()];
            trainer.learn(&row.input, &row.context, &row.good, &row.bad, 0.1)?;
        }
        let model = trainer.quantize();
        let result = metric(&model, &rows, "validation");
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
    let report = json!({"architecture":"8192x16 hashed Unicode embeddings; 48->32 ReLU->1 pairwise neural scorer", "implementation":"Rust training and int8 runtime; no external model", "seed":seed,"epochs":epochs,"selected_epoch":selected,"fresh_initialization":true,"warm_start":false,
        "selection":"validation only; test evaluated after selection", "model_bytes":MODEL_BYTES,"resident_bytes":model.resident_bytes(),"model_sha256":format!("{:x}",Sha256::digest(&best)),"dataset_sha256":format!("{:x}",Sha256::digest(&raw)),"seconds":started.elapsed().as_secs_f64(),"before":before,"after":after,
        "installed":false,"network_used":false,"limits":"Synthetic candidate-pair evaluation, not full IME accuracy. Bounded mean-pooled character features, not general text generation. No A13 or phone measurement."});
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{report}");
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("luqnexa-train: {error}");
        std::process::exit(1);
    }
}
