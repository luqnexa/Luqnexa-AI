// SPDX-License-Identifier: AGPL-3.0-only
use std::process::Command;
#[test]
fn portable_probe_rejects_invalid_model_and_bounds_work() {
    let path = std::env::temp_dir().join(format!("luqnexa-probe-{}.lqml", std::process::id()));
    std::fs::write(&path, luqnexa_ml::Trainer::new(9).quantize().to_bytes()).unwrap();
    let bin = env!("CARGO_BIN_EXE_mobile-probe");
    let ok = Command::new(bin).arg(&path).arg("100").output().unwrap();
    assert!(
        ok.status.success(),
        "{}",
        String::from_utf8_lossy(&ok.stderr)
    );
    let text = String::from_utf8(ok.stdout).unwrap();
    assert_eq!(text.matches("\"p95_us\"").count(), 6);
    assert!(text.contains("\"candidates\":32"));
    assert!(text.contains("\"bounded_max\":true"));
    assert!(
        !Command::new(bin)
            .arg(&path)
            .arg("100001")
            .output()
            .unwrap()
            .status
            .success()
    );
    std::fs::write(&path, b"corrupt").unwrap();
    assert!(
        !Command::new(bin)
            .arg(&path)
            .output()
            .unwrap()
            .status
            .success()
    );
    std::fs::remove_file(&path).unwrap();
}
