// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Get absolute path to the crate root
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // Path to cpp directory
    let cpp_dir = crate_dir.join("cpp");

    // Always run make in cpp/
    let status = Command::new("make")
        .arg("-j")
        .current_dir(&cpp_dir)
        .status()
        .expect("Failed to run make in cpp/");

    if !status.success() {
        panic!("C++ build failed");
    }

    // Link against the static library in cpp/
    println!("cargo:rustc-link-search=native={}", cpp_dir.display());
    println!("cargo:rustc-link-lib=static=trait_dialect");

    // Add rpath so cargo test can find the MLIR shared library at runtime.
    let mlir_prefix = env::var("MLIR_SYS_220_PREFIX")
        .expect("MLIR_SYS_220_PREFIX must be set in the environment");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib", mlir_prefix);

    // Ensure rebuild if anything in cpp/ changes
    println!("cargo:rerun-if-changed=cpp/");
}
