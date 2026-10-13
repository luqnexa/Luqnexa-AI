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
    let before = fs::read(output.join("model.lqml")).unwrap();
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
    fs::remove_dir(output).unwrap();
    fs::remove_file(data).unwrap();
    fs::remove_dir(directory).unwrap();
}
