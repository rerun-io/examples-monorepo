//! Brush training observations converted to the native Rerun wire format.

/// Keep the first, periodic and final snapshots, each once.
pub fn retains(step: u32, final_step: u32, first: u32, every: u32) -> bool {
    step > 0 && (step == final_step || step == first || (every > 0 && step.is_multiple_of(every)))
}

/// Convert CPU readbacks to the native wire format; SH is coefficient-major RGB.
pub fn to_archetype(
    transforms: &[f32],
    coefficients: &[f32],
    raw_opacities: &[f32],
    coefficient_count: usize,
    full_sh: bool,
) -> Result<rerun::GaussianSplats3D, ConversionError> {
    let count = raw_opacities.len();
    if transforms.len() != count * 10
        || coefficient_count == 0
        || coefficients.len() != count * coefficient_count * 3
    {
        return Err(ConversionError::Dimensions);
    }
    let transforms = transforms.as_chunks::<10>().0.iter();
    let centers = transforms.clone().map(|t| [t[0], t[1], t[2]]);
    let scales = transforms
        .clone()
        .map(|t| [t[7].exp(), t[8].exp(), t[9].exp()]);
    let rotations = transforms.map(|t| {
        let norm = (t[3] * t[3] + t[4] * t[4] + t[5] * t[5] + t[6] * t[6]).sqrt();
        rerun::Quaternion::from_xyzw(if norm > 0.0 {
            [t[4] / norm, t[5] / norm, t[6] / norm, t[3] / norm]
        } else {
            [0.0, 0.0, 0.0, 1.0]
        })
    });
    let c0 = 0.5 * (1.0 / std::f32::consts::PI).sqrt();
    let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    let colors = coefficients
        .chunks_exact(coefficient_count * 3)
        .zip(raw_opacities)
        .map(|(sh, opacity)| {
            rerun::Color::new((
                byte(0.5 + c0 * sh[0]),
                byte(0.5 + c0 * sh[1]),
                byte(0.5 + c0 * sh[2]),
                byte(1.0 / (1.0 + (-opacity).exp())),
            ))
        });
    let mut result = rerun::GaussianSplats3D::new(centers)
        .with_scales(scales)
        .with_quaternions(rotations)
        .with_colors(colors);
    if full_sh && coefficient_count > 1 {
        let sh = coefficients
            .chunks_exact(coefficient_count * 3)
            .map(|coefficients| {
                let mut rest = [[0.0; 3]; 15];
                for (target, source) in rest
                    .iter_mut()
                    .zip(coefficients[3..].as_chunks::<3>().0.iter())
                {
                    target.copy_from_slice(source);
                }
                rerun::components::SphericalHarmonics3Rgb::from(rest)
            });
        result = result.with_sh_coefficients(sh);
    }
    let degree = if full_sh {
        coefficient_count.isqrt().saturating_sub(1).min(3) as u32
    } else {
        0
    };
    Ok(result.with_spherical_harmonics_degree(degree))
}

#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    #[error("inconsistent Brush splat tensor dimensions")]
    Dimensions,
    #[error("Brush tensor readback failed: {0}")]
    Readback(String),
}

/// Read a captured slot value asynchronously, folding Brush's training scale floor
/// exactly as its PLY exporter does. No readback runs on the training thread.
pub async fn read_splats(
    splats: brush_render::gaussian_splats::Splats,
    full_sh: bool,
) -> Result<rerun::GaussianSplats3D, ConversionError> {
    use burn::tensor::s;
    let training = splats.min_scale.is_some();
    let splats = splats.bake_min_scale();
    let coefficients = if full_sh {
        splats.sh_coeffs.val()
    } else {
        splats.sh_coeffs.val().slice(s![.., 0..1])
    };
    let coefficient_count = coefficients.dims()[1];
    let (transforms, coefficients, raw_opacities) = tokio::try_join!(
        splats.transforms.val().into_data_async(),
        coefficients.into_data_async(),
        splats.raw_opacities.val().into_data_async()
    )
    .map_err(|e| ConversionError::Readback(e.to_string()))?;
    let mut transforms = transforms
        .try_into_vec::<f32>()
        .map_err(|e| ConversionError::Readback(e.to_string()))?;
    let coefficients = coefficients
        .convert::<f32>()
        .try_into_vec::<f32>()
        .map_err(|e| ConversionError::Readback(e.to_string()))?;
    let raw_opacities = raw_opacities
        .try_into_vec::<f32>()
        .map_err(|e| ConversionError::Readback(e.to_string()))?;
    if training {
        // Brush normalizes training rotations before export; Rerun normalizes
        // the exported values again. Match that path, including f32 rounding.
        for t in transforms.as_chunks_mut::<10>().0 {
            let norm = (t[3] * t[3] + t[4] * t[4] + t[5] * t[5] + t[6] * t[6])
                .sqrt()
                .max(1e-12);
            for value in &mut t[3..7] {
                *value /= norm;
            }
        }
    }
    to_archetype(
        &transforms,
        &coefficients,
        &raw_opacities,
        coefficient_count,
        full_sh,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn conversion_matches_rerun_ply_activation() {
        let ply = concat!(
            include_str!("../../gsplat-cli/tests/one-vertex-ply-header.txt"),
            "1 2 3 0 0.6931472 -0.6931472 0 3 0 4 0 -4 0 4\n",
        )
        .as_bytes();
        let expected = rerun::GaussianSplats3D::from_ply_file_contents(ply, None).unwrap();
        use std::f32::consts::LN_2;
        let converted = to_archetype(
            &[1.0, 2.0, 3.0, 0.0, 3.0, 0.0, 4.0, 0.0, LN_2, -LN_2],
            &[-4.0, 0.0, 4.0],
            &[0.0],
            1,
            true,
        )
        .unwrap();
        assert_eq!(converted.centers, expected.centers);
        assert_eq!(converted.scales, expected.scales);
        assert_eq!(converted.quaternions, expected.quaternions);
        assert_eq!(converted.colors, expected.colors);
    }
    #[test]
    fn final_snapshot_records_the_available_sh_degree() {
        for (coefficient_count, degree) in [(1, 0), (4, 1), (9, 2), (16, 3), (25, 3)] {
            let snapshot = to_archetype(
                &[0.0; 10],
                &vec![0.0; coefficient_count * 3],
                &[0.0],
                coefficient_count,
                true,
            )
            .unwrap();
            let expected =
                rerun::GaussianSplats3D::new([[0.0; 3]]).with_spherical_harmonics_degree(degree);
            assert_eq!(
                snapshot.spherical_harmonics_degree,
                expected.spherical_harmonics_degree
            );
        }
    }

    #[test]
    fn sh_is_coefficient_major_truncated_and_final_only() {
        let mut coefficients = vec![0.0; 75];
        coefficients[3..6].copy_from_slice(&[1.0, 2.0, 3.0]);
        coefficients[45..48].copy_from_slice(&[4.0, 5.0, 6.0]);
        coefficients[48..51].copy_from_slice(&[99.0, 99.0, 99.0]);
        let mut expected = [[0.0; 3]; 15];
        expected[0] = [1.0, 2.0, 3.0];
        expected[14] = [4.0, 5.0, 6.0];
        let expected = rerun::GaussianSplats3D::new([[0.0; 3]])
            .with_sh_coefficients([rerun::components::SphericalHarmonics3Rgb::from(expected)]);
        assert_eq!(
            to_archetype(&[0.0; 10], &coefficients, &[0.0], 25, true)
                .unwrap()
                .sh_coefficients,
            expected.sh_coefficients
        );
        assert!(
            to_archetype(&[0.0; 10], &coefficients, &[0.0], 25, false)
                .unwrap()
                .sh_coefficients
                .is_none()
        );
    }
}
