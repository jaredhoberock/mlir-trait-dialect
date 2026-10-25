//! A clone of this repository builds what was tested: every dependency cargo
//! resolves under the committed lock comes from this repository or from its own
//! pinned git repository.

#[test]
fn every_estate_dependency_is_pinned_to_its_git_repository() {
    build_support::check_estate_pins(env!("CARGO_MANIFEST_DIR"));
}
