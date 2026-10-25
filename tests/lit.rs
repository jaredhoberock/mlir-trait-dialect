//! The lit suite under `cargo test`, run from the execution root build.rs wrote
//! its site configuration into.

#[test]
fn lit_rows_pass() {
    build_support::run_lit(env!("LLVM_LIT_PY"), env!("LIT_SUITE"));
}
