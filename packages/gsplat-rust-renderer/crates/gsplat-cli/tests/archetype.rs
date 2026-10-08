//! The quantization lane crosses the native archetype boundary before conversion.
#[test]
fn native_ply_round_trip_preserves_geometry_and_quantizes_opacity() {
    let path = std::env::temp_dir().join(format!("archetype-{}.ply", std::process::id()));
    std::fs::write(
        &path,
        concat!(
            include_str!("one-vertex-ply-header.txt"),
            "1 2 3 -2 -2 -2 1 0 0 0 0 0 0 0\n",
        ),
    )
    .unwrap();
    let splats = gsplat_cli::engines::archetype_splats(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    assert_eq!(&splats.transforms[0][..3], &[1.0, 2.0, 3.0]);
    assert!((splats.transforms[0][7] + 2.0).abs() < 1e-6);
    assert_eq!(splats.sh_degree, 0);
    // Rerun rounds 0.5 * 255 to 128. This is distinct from the raw logit 0.
    assert!((splats.raw_opacities[0] - 0.007843178).abs() < 1e-6);
}
