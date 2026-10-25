//! A clone of this repository builds what was tested: every dependency cargo
//! resolves under the committed lock comes from the source its manifest
//! declares, and no path package lies outside this repository.

#[test]
fn builds_only_from_what_the_lock_pins() {
    build_support::check_pinned_sources(env!("CARGO_MANIFEST_DIR"));
}
