use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let mlir_prefix =
        env::var("MLIR_SYS_220_PREFIX").expect("MLIR_SYS_220_PREFIX must be set");
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cpp_dir = crate_dir.join("cpp");
    let build_dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("cpp");
    let lowering_driver_source_dir = env::var("DEP_LOWERING_DRIVER_SOURCE_DIR")
        .expect("DEP_LOWERING_DRIVER_SOURCE_DIR must be set by mlir-lowering-driver");

    let status = Command::new("make")
        .arg("-j")
        .arg(format!("BUILD_DIR={}", build_dir.display()))
        .arg(format!("LOWERING_DRIVER_SOURCE_DIR={lowering_driver_source_dir}"))
        .current_dir(&cpp_dir)
        .status()
        .expect("Failed to run make in cpp/");

    if !status.success() {
        panic!("C++ build failed");
    }

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=trait_dialect");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{mlir_prefix}/lib");

    println!("cargo:source_dir={}", cpp_dir.display());
    println!("cargo:include_dir={}", build_dir.display());
    println!("cargo:lib_dir={}", build_dir.display());

    println!("cargo:rerun-if-env-changed=MLIR_SYS_220_PREFIX");
    println!("cargo:rerun-if-changed=cpp");
    // This dialect compiles against the driver's contribution headers rather
    // than linking them, so a change to either must recompile its C++.
    println!("cargo:rerun-if-changed={lowering_driver_source_dir}/LoweringContribution.hpp");
    println!("cargo:rerun-if-changed={lowering_driver_source_dir}/c_api.h");
}
