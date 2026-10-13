// SPDX-License-Identifier: AGPL-3.0-only
use luqnexa_ml::{MODEL_BYTES, Trainer};
use std::{hint::black_box, time::Instant};
fn main() {
    let model = match std::env::var_os("LUQNEXA_BENCH_MODEL") {
        Some(path) => luqnexa_ml::Model::from_bytes(&std::fs::read(path).unwrap()).unwrap(),
        None => Trainer::new(1337).quantize(),
    };
    let candidates = [
        "Pythonで医師に相談した",
        "Pythonで意思に相談した",
        "Pythonで石に相談した",
        "Pythonでいしに相談した",
    ];
    let mut samples = Vec::new();
    for i in 0..2100 {
        let start = Instant::now();
        let input = model.prepare(
            black_box("pythondeishinisoudanshita"),
            black_box("病院の診察について相談した"),
        );
        for text in candidates {
            black_box(model.score(&input, black_box(text)));
        }
        if i >= 100 {
            samples.push(start.elapsed().as_nanos());
        }
    }
    samples.sort_unstable();
    println!(
        "model_bytes={MODEL_BYTES} resident_bytes={} prepare_and_4_candidates_p50_us={:.3} p95_us={:.3} iterations={}; host only, no dictionary/UI/phone measurements",
        model.resident_bytes(),
        samples[samples.len() / 2] as f64 / 1000.0,
        samples[samples.len() * 95 / 100] as f64 / 1000.0,
        samples.len()
    );
}
