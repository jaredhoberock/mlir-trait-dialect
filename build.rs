use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let mlir_prefix =
        env::var("MLIR_SYS_230_PREFIX").expect("MLIR_SYS_230_PREFIX must be set");
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

    // The dialect plugins this crate's lit rows load, each after the plugins it
    // references, published for the lit rows of the crates that depend on this one.
    let plugin = build_dir.join("libtrait_dialect.so");
    let plugins = dialect_plugins(&[], plugin.clone());
    println!("cargo:plugin={}", plugin.display());
    println!("cargo:plugins={}", env::join_paths(&plugins).unwrap().to_string_lossy());
    let lit_suite = write_lit_site_config(
        &mlir_prefix,
        "lit",
        "cpp/tests",
        &[
            ("dialect_plugins", python_paths(&plugins)),
        ],
    );
    println!("cargo:rustc-env=LIT_SUITE={}", lit_suite.display());

    println!("cargo:rerun-if-env-changed=MLIR_SYS_230_PREFIX");
    println!("cargo:rerun-if-changed=cpp");
    // This dialect compiles against the driver's contribution headers rather
    // than linking them, so a change to either must recompile its C++.
    println!("cargo:rerun-if-changed={lowering_driver_source_dir}/LoweringContribution.hpp");
    println!("cargo:rerun-if-changed={lowering_driver_source_dir}/c_api.h");
}

/// The dialect plugins a process loads to hold this crate's dialect, in load
/// order: the list each crate in `dependencies` (named by its `links` key)
/// publishes, in the order given, then `own`, each plugin kept at its first
/// appearance. A plugin's data references to a dependency's symbols bind when
/// the loader opens it, so each plugin follows the plugins it references; every
/// published list is so ordered, and keeping first appearances of their
/// concatenation preserves the order.
fn dialect_plugins(dependencies: &[&str], own: PathBuf) -> Vec<PathBuf> {
    let mut plugins: Vec<PathBuf> = Vec::new();
    for dependency in dependencies {
        let published = env::var_os(format!("DEP_{dependency}_PLUGINS"))
            .unwrap_or_else(|| panic!("DEP_{dependency}_PLUGINS must be set by its crate"));
        for plugin in env::split_paths(&published) {
            if !plugins.contains(&plugin) {
                plugins.push(plugin);
            }
        }
    }
    plugins.push(own);
    plugins
}

/// A Python string literal naming `path`.
fn python_path(path: &Path) -> String {
    format!("{:?}", path.display().to_string())
}

/// A Python list literal naming `paths` in order.
fn python_paths(paths: &[PathBuf]) -> String {
    let literals: Vec<String> = paths.iter().map(|path| python_path(path)).collect();
    format!("[{}]", literals.join(", "))
}

/// Writes `<OUT_DIR>/<suite>/lit.site.cfg.py`, the site configuration of the lit
/// suite whose `lit.cfg.py` stands in the crate's `source` directory, in upstream
/// MLIR's `lit.site.cfg.py.in` shape: it records the toolchain's directories and
/// `values` (each a `config` attribute and the Python expression it holds), makes
/// the directory the suite's execution root, and loads the suite's `lit.cfg.py`,
/// which takes every tool and plugin path from these values and nothing else.
/// Returns the directory, which lit runs on.
fn write_lit_site_config(mlir_prefix: &str, suite: &str, source: &str, values: &[(&str, String)]) -> PathBuf {
    let prefix = Path::new(mlir_prefix);
    let exec_root = PathBuf::from(env::var("OUT_DIR").unwrap()).join(suite);
    let lit_cfg = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join(source).join("lit.cfg.py");
    let mut site = String::from("# Written by build.rs; lit.cfg.py reads these values and nothing else.\n");
    site.push_str(&format!("config.llvm_tools_dir = {}\n", python_path(&prefix.join("bin"))));
    site.push_str(&format!("config.llvm_lib_dir = {}\n", python_path(&prefix.join("lib"))));
    site.push_str("# FileCheck and not: the toolchain install carries neither (LLVM_INSTALL_UTILS is off),\n");
    site.push_str("# so they come from the build tree the install was made from, beside it.\n");
    site.push_str(&format!("config.llvm_utils_dir = {}\n", python_path(&llvm_utils_dir(prefix))));
    for (name, value) in values {
        site.push_str(&format!("config.{name} = {value}\n"));
    }
    site.push_str(&format!("config.test_exec_root = {}\n", python_path(&exec_root)));
    site.push_str(&format!("lit_config.load_config(config, {})\n", python_path(&lit_cfg)));
    std::fs::create_dir_all(&exec_root).expect("the lit execution root can be created");
    std::fs::write(exec_root.join("lit.site.cfg.py"), site).expect("lit.site.cfg.py can be written");
    exec_root
}

/// The directory of LLVM's test utilities (FileCheck, not): the build
/// tree the toolchain install was made from, which stands beside the install.
///
/// XXX TODO the install carries none of these because the toolchain is configured
/// with LLVM_INSTALL_UTILS off; this path goes when the recorded toolchain
/// configuration installs them (LLVM_INSTALL_UTILS=ON).
fn llvm_utils_dir(prefix: &Path) -> PathBuf {
    prefix.join("..").join("build-release-asserts").join("bin")
}
