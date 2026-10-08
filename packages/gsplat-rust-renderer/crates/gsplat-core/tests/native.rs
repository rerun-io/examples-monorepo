use gsplat_core::native::NativeSplats;

#[test]
fn native_upload_preserves_alpha_endpoints_and_decodes_dc() {
    let splats = NativeSplats {
        centers: &[[1.0, 2.0, 3.0], [0.0; 3]],
        scales: &[[2.0, 1.0, 0.5]],
        quaternions: &[[0.0, 0.0, 0.0, 1.0]],
        colors: &[0xff0000ff, 0x00ff0000],
        sh: &[],
        degree: 0,
    }
    .to_core();
    assert_eq!(
        splats.transforms[0][..7],
        [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
    );
    assert!((splats.transforms[0][7] - 2.0_f32.ln()).abs() < 1e-6);
    assert_eq!(splats.raw_opacities, [100.0, -100.0]);
    assert!((splats.sh_coefficients[0][0] - 1.7724539).abs() < 1e-6);
    assert!((splats.sh_coefficients[0][1] + 1.7724539).abs() < 1e-6);
}

#[test]
fn native_degree_selects_complete_bands_and_repeats_short_attributes() {
    use half::f16;
    for degree in 0..=3 {
        let splats = NativeSplats {
            centers: &[[0.0; 3]; 2],
            scales: &[],
            quaternions: &[],
            colors: &[0x80808080],
            sh: &[[[f16::from_f32(0.25); 3]; 15]],
            degree,
        }
        .to_core();
        assert_eq!(
            splats.sh_coefficients.len(),
            2 * (degree as usize + 1).pow(2)
        );
        assert_eq!(splats.raw_opacities[0], splats.raw_opacities[1]);
        if degree > 0 {
            assert_eq!(splats.sh_coefficients[1], [0.25; 3]);
        }
    }
}

#[test]
fn missing_sh_is_dc_only_and_excess_degree_is_capped() {
    use half::f16;
    for sh in [vec![], vec![[[f16::ZERO; 3]; 15]]] {
        let splats = NativeSplats {
            centers: &[[0.0; 3]],
            scales: &[],
            quaternions: &[],
            colors: &[],
            sh: &sh,
            degree: 99,
        }
        .to_core();
        assert_eq!(splats.sh_degree, if sh.is_empty() { 0 } else { 3 });
    }
}
