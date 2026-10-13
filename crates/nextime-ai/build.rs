// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit compile-time research model snapshot; no inference-time file access.
use sha2::{Digest, Sha256};
use std::{env, fs, path::PathBuf};

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn main() {
    println!("cargo:rerun-if-env-changed=LUQNEXA_RESEARCH_MODEL_DIR");
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("../../models");
    let research = env::var_os("CARGO_FEATURE_RESEARCH_MODEL").is_some();
    let source = if research {
        PathBuf::from(
            env::var_os("LUQNEXA_RESEARCH_MODEL_DIR")
                .expect("research-model requires an explicit LUQNEXA_RESEARCH_MODEL_DIR"),
        )
    } else {
        root
    };
    let model_path = source.join("native-model.bin");
    let anchor_path = source.join("native-anchors.tsv");
    for path in [&model_path, &anchor_path] {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    assert_eq!(
        fs::metadata(&model_path).unwrap().len(),
        8200,
        "invalid model size"
    );
    let model = fs::read(&model_path).unwrap();
    assert!(model.starts_with(b"NXCR0001"), "invalid model header");
    assert!(
        fs::metadata(&anchor_path).unwrap().len() <= 256 * 1024,
        "anchor table too large"
    );
    let anchors = fs::read_to_string(&anchor_path).unwrap();
    assert!(!anchors.is_empty(), "empty anchor table");
    assert!(anchors.lines().count() <= 8192, "too many anchors");
    for line in anchors.lines() {
        let (a, b) = line.split_once('\t').expect("invalid anchor pair");
        assert!(!a.is_empty() && !b.is_empty(), "empty anchor span");
        assert!(
            a.chars().count() <= 160 && b.chars().count() <= 160,
            "anchor span too long"
        );
        assert!(
            !a.chars().chain(b.chars()).any(char::is_control),
            "invalid anchor character"
        );
    }
    let model_sha = sha(&model);
    let anchor_sha = sha(anchors.as_bytes());
    if research {
        let report_path = source.join("report.json");
        println!("cargo:rerun-if-changed={}", report_path.display());
        let report: serde_json::Value =
            serde_json::from_slice(&fs::read(report_path).unwrap()).unwrap();
        assert_eq!(
            report["model_sha256"].as_str(),
            Some(model_sha.as_str()),
            "research model hash mismatch"
        );
        assert_eq!(
            report["reproducible"].as_bool(),
            Some(true),
            "unverified model reproducibility"
        );
        assert_eq!(
            report["passes_candidate_regression_gate"].as_bool(),
            Some(true),
            "model failed regression gate"
        );
        let training_path = source.join("training.jsonl");
        println!("cargo:rerun-if-changed={}", training_path.display());
        assert_eq!(
            report["training_sha256"].as_str(),
            Some(sha(&fs::read(training_path).unwrap()).as_str()),
            "training snapshot mismatch"
        );
        let output = PathBuf::from(env::var_os("OUT_DIR").unwrap());
        fs::write(output.join("research-model.bin"), model).unwrap();
        fs::write(output.join("research-anchors.tsv"), anchors).unwrap();
    }
    println!("cargo:rustc-env=LUQNEXA_COMPILED_MODEL_SHA256={model_sha}");
    println!("cargo:rustc-env=LUQNEXA_COMPILED_ANCHORS_SHA256={anchor_sha}");
}
