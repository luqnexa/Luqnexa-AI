// SPDX-License-Identifier: AGPL-3.0-only
//! Portable on-device latency probe; not a full IME benchmark.
use luqnexa_ml::{MODEL_BYTES, Model};
use std::{hint::black_box, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: mobile-probe MODEL [ITERATIONS:100..100000]")?;
    let iterations: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(2000);
    if !(100..=100_000).contains(&iterations) || args.next().is_some() {
        return Err("invalid iterations or extra arguments".into());
    }
    if std::fs::metadata(&path)?.len() != MODEL_BYTES as u64 {
        return Err("invalid model size".into());
    }
    let bytes = std::fs::read(path)?;
    let load = Instant::now();
    let model = Model::from_bytes(&bytes)?;
    let load_us = load.elapsed().as_secs_f64() * 1_000_000.0;
    drop(bytes);
    // Representative kernel workload plus maximum bounded Unicode features.
    // Repeated supplied candidates isolate ranking cost, not semantic accuracy.
    let normal = [
        "PythonでAIを作る",
        "Pythonで愛を作る",
        "PythonでAIをつくる",
        "pythonでAIを作る",
    ];
    let long_reading = "あ".repeat(64);
    let long_context = "文".repeat(48);
    let long_surface = "候".repeat(96);
    println!(
        "{{\"os\":\"{}\",\"arch\":\"{}\",\"model_bytes\":{},\"model_resident_bytes\":{},\"model_load_us\":{:.3},\"iterations\":{},\"warmup\":100,\"scope\":\"prepare plus supplied candidate scoring; no dictionary/UI/power; repeated candidates are kernel workload\",\"measurements\":[",
        std::env::consts::OS,
        std::env::consts::ARCH,
        MODEL_BYTES,
        model.resident_bytes(),
        load_us,
        iterations
    );
    let mut first = true;
    for bounded_max in [false, true] {
        for count in [1, 8, 32] {
            let mut samples = Vec::with_capacity(iterations);
            for i in 0..iterations + 100 {
                let start = Instant::now();
                let prepared = model.prepare(
                    black_box(if bounded_max {
                        &long_reading
                    } else {
                        "pythondeaiwotsukuru"
                    }),
                    black_box(if bounded_max {
                        &long_context
                    } else {
                        "プログラミングの話をしている"
                    }),
                );
                for n in 0..count {
                    let surface = if bounded_max {
                        long_surface.as_str()
                    } else {
                        normal[n % normal.len()]
                    };
                    black_box(model.score(&prepared, black_box(surface)));
                }
                let elapsed = start.elapsed().as_nanos();
                if i >= 100 {
                    samples.push(elapsed);
                }
            }
            samples.sort_unstable();
            if !first {
                println!(",");
            }
            first = false;
            println!(
                "{{\"bounded_max\":{},\"candidates\":{},\"p50_us\":{:.3},\"p95_us\":{:.3},\"p99_us\":{:.3}}}",
                bounded_max,
                count,
                samples[iterations / 2] as f64 / 1000.0,
                samples[iterations * 95 / 100] as f64 / 1000.0,
                samples[iterations * 99 / 100] as f64 / 1000.0
            );
        }
    }
    println!("]}}");
    Ok(())
}
