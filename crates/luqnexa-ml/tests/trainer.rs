#![cfg(feature = "trainer")]
use std::{fs, process::Command};

#[test]
fn cli_enforces_partitions_and_keeps_existing_runs() {
    let directory =
        std::env::temp_dir().join(format!("luqnexa-neural-test-{}", std::process::id()));
    fs::create_dir(&directory).unwrap();
    let data = directory.join("data.jsonl");
    let output = directory.join("run");
    let rows = [
        r#"{"input":"hashi","context":"","good":"箸","bad":"橋","split":"train"}"#,
        r#"{"input":"ame","context":"雨雲","good":"雨","bad":"飴","split":"validation"}"#,
        r#"{"input":"ai","context":"技術","good":"AI","bad":"愛","split":"test"}"#,
    ];
    fs::write(&data, rows.join("\n")).unwrap();
    let run = || {
        Command::new(env!("CARGO_BIN_EXE_luqnexa-train"))
            .args([
                "--data",
                data.to_str().unwrap(),
                "--output",
                output.to_str().unwrap(),
                "--epochs",
                "2",
            ])
            .output()
            .unwrap()
    };
    assert!(run().status.success());
    let trace: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("training-trace.json")).unwrap()).unwrap();
    assert_eq!(trace.as_array().unwrap().len(), 2);
    assert!(
        trace
            .as_array()
            .unwrap()
            .iter()
            .all(|row| row.get("test").is_none())
    );
    let before = fs::read(output.join("model.lqml")).unwrap();
    let audit = Command::new(env!("CARGO_BIN_EXE_pair-audit"))
        .args([
            output.join("model.lqml").to_str().unwrap(),
            data.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(audit.status.success());
    let inspected: serde_json::Value = serde_json::from_slice(&audit.stdout).unwrap();
    assert_eq!(inspected["training"], false);
    assert_eq!(inspected["temperature_examples"], 1);
    assert_eq!(inspected["rows"].as_array().unwrap().len(), 3);
    assert_eq!(fs::read(output.join("model.lqml")).unwrap(), before);
    let audit_data = directory.join("audit.jsonl");
    let mut alternate: Vec<serde_json::Value> = rows
        .iter()
        .map(|row| serde_json::from_str(row).unwrap())
        .collect();
    alternate[1]["good"] = serde_json::json!("Different validation candidate");
    alternate[2]["good"] = serde_json::json!("Different test candidate");
    fs::write(
        &audit_data,
        alternate
            .iter()
            .map(|row| row.to_string())
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    let other = Command::new(env!("CARGO_BIN_EXE_pair-audit"))
        .args([
            output.join("model.lqml").to_str().unwrap(),
            audit_data.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(other.status.success());
    let other: serde_json::Value = serde_json::from_slice(&other.stdout).unwrap();
    assert_eq!(
        inspected["ranking_temperature"],
        other["ranking_temperature"]
    );
    let experimental = directory.join("interaction-run");
    let result = Command::new(env!("CARGO_BIN_EXE_luqnexa-train"))
        .args([
            "--data",
            data.to_str().unwrap(),
            "--output",
            experimental.to_str().unwrap(),
            "--epochs",
            "2",
            "--architecture",
            "interaction-v2",
        ])
        .output()
        .unwrap();
    assert!(result.status.success());
    assert_eq!(
        &fs::read(experimental.join("model.lqml")).unwrap()[..8],
        b"LQML0002"
    );
    assert_eq!(&before[..8], b"LQML0001");
    let grouped = directory.join("grouped-run");
    let result = Command::new(env!("CARGO_BIN_EXE_luqnexa-train"))
        .args([
            "--data",
            data.to_str().unwrap(),
            "--output",
            grouped.to_str().unwrap(),
            "--epochs",
            "2",
            "--schedule",
            "grouped-shuffled",
        ])
        .output()
        .unwrap();
    assert!(result.status.success());
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(grouped.join("report.json")).unwrap()).unwrap();
    // Validation and test rows never become learning groups.
    assert_eq!(report["training_sentence_context_groups"], 1);
    assert_eq!(report["largest_group_pairs"], 1);
    assert_eq!(report["gradient_updates"], 2);
    assert_eq!(report["schedule"], "grouped-shuffled");
    let augmentation_data = directory.join("augmentation.jsonl");
    let augmentation_run = directory.join("augmentation-run");
    let noisy = r#"{"input":"hasgi","group_input":"hashi","context":"","good":"箸","bad":"hasgi","split":"train"}"#;
    fs::write(&augmentation_data, format!("{}\n{noisy}", rows.join("\n"))).unwrap();
    let augment = || {
        Command::new(env!("CARGO_BIN_EXE_luqnexa-train"))
            .args([
                "--data",
                augmentation_data.to_str().unwrap(),
                "--output",
                augmentation_run.to_str().unwrap(),
                "--epochs",
                "2",
                "--schedule",
                "grouped-shuffled",
            ])
            .output()
            .unwrap()
    };
    assert!(augment().status.success());
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(augmentation_run.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["training_sentence_context_groups"], 1);
    assert_eq!(report["largest_group_pairs"], 2);
    assert_eq!(report["gradient_updates"], 2);
    for file in ["model.lqml", "report.json", "training-trace.json"] {
        fs::remove_file(augmentation_run.join(file)).unwrap();
    }
    fs::remove_dir(&augmentation_run).unwrap();
    for parent in ["ame", "absent"] {
        fs::write(
            &augmentation_data,
            format!(
                "{}\n{}",
                rows.join("\n"),
                noisy.replace(
                    "\"group_input\":\"hashi\"",
                    &format!("\"group_input\":\"{parent}\"")
                )
            ),
        )
        .unwrap();
        assert!(!augment().status.success());
        assert!(!augmentation_run.exists());
    }
    fs::remove_file(augmentation_data).unwrap();
    assert!(!run().status.success());
    assert_eq!(fs::read(output.join("model.lqml")).unwrap(), before);
    let bad_output = directory.join("bad-run");
    fs::write(
        &data,
        rows.join("\n")
            .replace("\"input\":\"ame\"", "\"input\":\"hashi\""),
    )
    .unwrap();
    let bad = Command::new(env!("CARGO_BIN_EXE_luqnexa-train"))
        .args([
            "--data",
            data.to_str().unwrap(),
            "--output",
            bad_output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!bad.status.success());
    assert!(!bad_output.exists());
    // Explicitly checked temporary files only; no recursive removal or user data.
    fs::remove_file(output.join("model.lqml")).unwrap();
    fs::remove_file(output.join("report.json")).unwrap();
    fs::remove_file(output.join("training-trace.json")).unwrap();
    fs::remove_dir(output).unwrap();
    for file in ["model.lqml", "report.json", "training-trace.json"] {
        fs::remove_file(experimental.join(file)).unwrap();
    }
    fs::remove_dir(experimental).unwrap();
    for file in ["model.lqml", "report.json", "training-trace.json"] {
        fs::remove_file(grouped.join(file)).unwrap();
    }
    fs::remove_dir(grouped).unwrap();
    fs::remove_file(data).unwrap();
    fs::remove_file(audit_data).unwrap();
    fs::remove_dir(directory).unwrap();
}
