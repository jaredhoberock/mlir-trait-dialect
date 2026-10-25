//! The lit suites under `cargo test`. Each suite runs from the execution root
//! build.rs wrote its site configuration into, which names the toolchain and the
//! plugins this build made; lit finds the rows through the site configuration's
//! source root.

use std::process::Command;

/// Runs lit over the suite whose execution root is `suite` and fails when lit
/// reports a failing row.
fn run_lit(suite: &str) {
    let status = Command::new("lit")
        .arg("-sv")
        .arg(suite)
        .status()
        .expect("lit runs");
    assert!(status.success(), "lit reported failing rows in {suite}");
}

/// The rows under cpp/tests pass.
#[test]
fn lit_rows_pass() {
    run_lit(env!("LIT_SUITE"));
}
