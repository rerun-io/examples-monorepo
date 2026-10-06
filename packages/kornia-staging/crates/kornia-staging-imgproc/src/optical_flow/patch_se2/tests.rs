#![allow(clippy::unwrap_used)]
use super::{Pattern, Pattern51};
use kornia_image::Image;
use nalgebra::Vector2;

/// Compare the public scalar and four-point patch operations at the bit level.
fn assert_group_bits<P: Pattern>(image: &Image<u16, 1>, positions: [Vector2<f32>; 4], angle: f32) {
    use super::oracle::{build_patch, patch_increment, OpticalFlowPatch};
    use super::se2_exp;
    use super::MAX_PATTERN_SIZE;
    use super::{build_patch_group, patch_increment_rows, patch_residual_taps};
    let mut data = vec![0.0; 4 * P::SIZE];
    let mut jacobian = vec![0.0; 12 * P::SIZE];
    let (means, valid) =
        build_patch_group::<P>(image, positions.map(Into::into), &mut data, &mut jacobian);
    let transforms = positions.map(|pos| {
        let mut transform = se2_exp(&[0.0, 0.0, angle]);
        transform.translation = (pos + Vector2::new(0.21, -0.37)).into();
        transform
    });
    for lane in 0..4 {
        let mut scalar = OpticalFlowPatch::<P>::default();
        (scalar.mean, scalar.valid) = build_patch::<P, _>(
            image,
            positions[lane].as_ref(),
            &mut scalar.data,
            1,
            scalar.h_se2_inv_j_se2_t.as_flattened_mut(),
            1,
            MAX_PATTERN_SIZE,
        );
        assert_eq!(valid[lane], scalar.valid, "lane {lane}");
        assert_eq!(means[lane].to_bits(), scalar.mean.to_bits());
        for tap in 0..P::SIZE {
            assert_eq!(
                data[4 * tap + lane].to_bits(),
                scalar.data[tap].to_bits(),
                "data lane {lane}, tap {tap}"
            );
            for row in 0..3 {
                assert_eq!(
                    jacobian[4 * (row * P::SIZE + tap) + lane].to_bits(),
                    scalar.h_se2_inv_j_se2_t[row][tap].to_bits(),
                    "factor lane {lane}, row {row}, tap {tap}"
                );
            }
        }
        let mut residual = [0.0; MAX_PATTERN_SIZE];
        let survived = scalar.residual(image, &transforms[lane], &mut residual);
        let mut tap_residual = [0.0; MAX_PATTERN_SIZE];
        assert_eq!(
            survived,
            patch_residual_taps::<P>(
                &data[lane..],
                4,
                image,
                &transforms[lane],
                &mut tap_residual
            )
        );
        assert_eq!(residual.map(f32::to_bits), tap_residual.map(f32::to_bits));
        let increment = patch_increment::<P>(
            scalar.h_se2_inv_j_se2_t.as_flattened(),
            1,
            MAX_PATTERN_SIZE,
            &residual,
        );
        let row_increment = patch_increment_rows::<P>(&jacobian[lane..], 4, 4 * P::SIZE, &residual);
        assert_eq!(row_increment.map(f32::to_bits), increment.map(f32::to_bits));
    }
}

/// SIMD lanes must retain the scalar tap order, including partial border patches.
#[test]
fn four_patch_builds_match_scalar_bits() {
    use super::Pattern52;
    let mut image = Image::from_size_val(
        kornia_image::ImageSize {
            width: 80,
            height: 64,
        },
        0,
    )
    .unwrap();
    let mut state = 0x7d91_230bu32;
    let mut random = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        state
    };
    for y in 0..64 {
        for x in 0..80 {
            image.set_pixel(x, y, 0, (random() >> 16) as u16).unwrap();
        }
    }
    for _ in 0..256 {
        let positions = std::array::from_fn(|_| {
            Vector2::new(
                (random() % 84_000) as f32 / 1000.0 - 2.0,
                (random() % 68_000) as f32 / 1000.0 - 2.0,
            )
        });
        let angle = (random() % 600) as f32 / 1000.0 - 0.3;
        assert_group_bits::<Pattern51>(&image, positions, angle);
        assert_group_bits::<Pattern52>(&image, positions, angle);
    }
}

#[test]
fn four_patch_degenerate_and_invalid_lanes_match_scalar_bits() {
    let mut image = Image::from_size_val(
        kornia_image::ImageSize {
            width: 80,
            height: 64,
        },
        0,
    )
    .unwrap();
    let positions = [
        Vector2::new(32.25, 30.5),
        Vector2::new(2.0, 2.0),
        Vector2::new(-100.0, -100.0),
        Vector2::new(f32::NAN, f32::INFINITY),
    ];
    assert_group_bits::<Pattern51>(&image, positions, 0.0);
    for y in 0..64 {
        for x in 0..80 {
            image.set_pixel(x, y, 0, 12_345).unwrap();
        }
    }
    assert_group_bits::<Pattern51>(&image, positions, -0.0);
}
