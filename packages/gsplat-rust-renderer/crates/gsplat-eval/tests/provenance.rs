//! Source archives must retain honest build provenance without a checkout.
#[path = "../build_support.rs"]
mod build_support;

#[test]
fn archive_uses_explicit_source_or_records_unknown() {
    let root =
        std::env::temp_dir().join(format!("gsplat-archive-provenance-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    assert_eq!(build_support::source_sha(&root, None), "unknown");
    assert_eq!(
        build_support::source_sha(&root, Some("archived-commit")),
        "archived-commit"
    );
    std::fs::remove_dir(&root).unwrap();
}

#[test]
fn checkout_identity_retains_dirty_detection() {
    let root =
        std::env::temp_dir().join(format!("gsplat-checkout-provenance-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    assert!(build_support::git(&root, &["init", "-q"]).is_some());
    std::fs::write(root.join("source"), "original").unwrap();
    assert!(build_support::git(&root, &["add", "source"]).is_some());
    assert!(
        build_support::git(
            &root,
            &[
                "-c",
                "user.name=Provenance test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "Source fixture"
            ]
        )
        .is_some()
    );
    let clean = build_support::source_sha(&root, None);
    assert_ne!(clean, "unknown");
    assert!(!clean.ends_with("-dirty"));
    std::fs::write(root.join("source"), "modified").unwrap();
    assert_eq!(
        build_support::source_sha(&root, None),
        format!("{clean}-dirty")
    );
    std::fs::remove_dir_all(&root).unwrap();
}
