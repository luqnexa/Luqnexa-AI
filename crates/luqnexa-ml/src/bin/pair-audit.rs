// SPDX-License-Identifier: AGPL-3.0-only
//! Inspect an already trained model against supplied pairs; never updates weights.
use luqnexa_ml::Model;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: pair-audit MODEL JSONL".into());
    }
    let bytes = std::fs::read(&args[0])?;
    let data = std::fs::read(&args[1])?;
    if data.len() > 10 * 1024 * 1024 {
        return Err("dataset too large".into());
    }
    let model = Model::from_bytes(&bytes)?;
    let mut rows = Vec::new();
    let mut train_margins = Vec::new();
    for line in std::str::from_utf8(&data)?
        .lines()
        .filter(|s| !s.trim().is_empty())
    {
        let row: Value = serde_json::from_str(line)?;
        let field = |key: &str| -> Result<&str, &'static str> {
            row[key].as_str().ok_or("missing text field")
        };
        let input = field("input")?;
        let context = field("context")?;
        let prepared = model.prepare(input, context);
        let good = model.score(&prepared, field("good")?);
        let bad = model.score(&prepared, field("bad")?);
        if row["split"] == "train" {
            train_margins.push((good - bad).abs());
        }
        rows.push(json!({"input":input,"context":context,"split":row["split"],"group":row["group"],
            "good":row["good"],"bad":row["bad"],"good_score":good,"bad_score":bad,"passed":good>bad}));
    }
    train_margins.sort_by(f32::total_cmp);
    let temperature = if train_margins.is_empty() {
        None
    } else {
        let n = train_margins.len();
        Some(((train_margins[(n - 1) / 2] + train_margins[n / 2]) * 0.5).clamp(0.0001, 100.0))
    };
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"model_sha256":format!("{:x}",Sha256::digest(&bytes)),
        "dataset_sha256":format!("{:x}",Sha256::digest(&data)),"rows":rows,"training":false,"installed":false,
        "ranking_temperature":temperature,"temperature_examples":train_margins.len(),"temperature_source":"Median absolute margin from training pairs only; not probability calibration",
        "scope":"Supplied candidate pairs; not full conversion or independent accuracy"})
        )?
    );
    Ok(())
}
