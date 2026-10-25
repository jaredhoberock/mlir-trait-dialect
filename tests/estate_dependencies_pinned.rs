//! The committed Cargo.lock pins every estate dependency to a commit of its git
//! repository. A package the lock records without that source -- a path, from a
//! `[patch]` override written into the lock -- builds from whatever checkout the
//! path names, so a clone of this repository would not build what was tested.

/// Fails when Cargo.lock records an estate package other than this crate (the
/// compiler, the lowering driver, or a dialect) with a source that is not its
/// repository over ssh.
#[test]
fn every_estate_dependency_is_pinned_to_its_git_repository() {
    let lock = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.lock"))
        .expect("Cargo.lock is tracked");
    for package in lock.split("[[package]]").skip(1) {
        let field = |key: &str| {
            let prefix = format!("{key} = \"");
            package.lines().find_map(|line| Some(line.strip_prefix(&prefix)?.trim_end_matches('"')))
        };
        let name = field("name").expect("every locked package has a name");
        let estate = name == "screech"
            || name == "mlir-lowering-driver"
            || (name.starts_with("mlir-") && name.ends_with("-dialect"));
        if estate && name != env!("CARGO_PKG_NAME") {
            let source = field("source").unwrap_or("a path");
            assert!(
                source.starts_with("git+ssh://git@github.com/jaredhoberock/"),
                "Cargo.lock takes {name} from {source}, not from its git repository"
            );
        }
    }
}
