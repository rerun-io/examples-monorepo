//! BRIEF and matcher parity with Basalt's own `computeAngles`, `computeDescriptors` and `matchDescriptors`.
//!
//! The reference is the output of `fixtures/basalt_brief/basalt_brief_dump.cpp`, built against RootSLAM's
//! `keypoints.cpp` (branch pixi @ 839629d) and run on `fixtures/frames/*.pgm` plus a synthetic u16 image, stored as
//! an `.rrd` on the NAS. Run with the `kornia-staging-golden` Pixi task (release, `--ignored`).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use kornia_image::{Image, ImageSize};
use kornia_staging_golden::BasaltBrief;
use kornia_staging_imgproc::features::{
    describe_keypoints_u16, match_mutual, BriefConfig, Descriptor256, MatchConfig,
};
use kornia_staging_imgproc::test_fixtures::read_pgm;

/// One image's angle and descriptor per keypoint, `None` outside the border.
type Described = Vec<Option<(f64, Descriptor256)>>;

/// The dump's inputs: each frame fixture widened with `v << 8`, then the synthetic image.
fn images() -> Vec<Image<u16, 1>> {
    let frames = Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/frames");
    let mut images: Vec<Image<u16, 1>> = [
        "000_cam0", "000_cam1", "001_cam0", "001_cam1", "002_cam0", "002_cam1",
    ]
    .iter()
    .map(|name| {
        let pgm = read_pgm(&frames.join(format!("frame_{name}.pgm")));
        let size = ImageSize {
            width: pgm.width,
            height: pgm.height,
        };
        Image::new(
            size,
            pgm.pixels.iter().map(|&v| u16::from(v) << 8).collect(),
        )
        .unwrap()
    })
    .collect();
    images.push(synthetic(320, 240));
    images
}

/// xorshift32 (seed 12345) noise `state >> 16`, then an integer 3x3 box mean clipped at the edges.
fn synthetic(width: usize, height: usize) -> Image<u16, 1> {
    let mut state = 12345u32;
    let noise: Vec<u32> = (0..width * height)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state >> 16
        })
        .collect();
    let mut out = vec![0u16; width * height];
    for y in 0..height as i64 {
        for x in 0..width as i64 {
            let (mut sum, mut count) = (0u32, 0u32);
            for (xx, yy) in (-1..=1).flat_map(|dy| (-1..=1).map(move |dx| (x + dx, y + dy))) {
                if xx >= 0 && yy >= 0 && xx < width as i64 && yy < height as i64 {
                    sum += noise[yy as usize * width + xx as usize];
                    count += 1;
                }
            }
            out[y as usize * width + x as usize] = (sum / count) as u16;
        }
    }
    Image::new(ImageSize { width, height }, out).unwrap()
}

/// Our descriptors of every image, in keypoint order, with the Basalt configuration.
fn described(golden: &BasaltBrief, images: &[Image<u16, 1>]) -> Vec<Described> {
    golden
        .images
        .iter()
        .zip(images)
        .map(|(g, image)| {
            assert_eq!((g.width, g.height), (image.width(), image.height()));
            let positions: Vec<[f32; 2]> = g.keypoints.iter().map(|k| k.position).collect();
            describe_keypoints_u16(image, &positions, BriefConfig::basalt())
                .into_iter()
                .map(|d| d.map(|d| (d.angle, d.descriptor)))
                .collect()
        })
        .collect()
}

#[test]
#[ignore = "needs the Basalt BRIEF golden .rrd on the NAS (kornia-staging-golden task)"]
fn angles_and_descriptors_are_bit_identical_to_basalt() {
    let golden = BasaltBrief::load_default();
    let images = images();
    assert_eq!(golden.images.len(), images.len());
    let ours = described(&golden, &images);
    let (mut inside, mut textured) = (0usize, 0usize);
    for (g, o) in golden.images.iter().zip(&ours) {
        for (k, d) in g.keypoints.iter().zip(o) {
            // B7: Basalt skips these keypoints too; we return None instead of a placeholder descriptor.
            assert_eq!(
                k.described.is_some(),
                d.is_some(),
                "border test at {:?}",
                k.position
            );
            if let (Some((angle, descriptor)), Some((our_angle, our_descriptor))) = (k.described, d)
            {
                assert_eq!(
                    angle.to_bits(),
                    our_angle.to_bits(),
                    "angle at {:?}",
                    k.position
                );
                assert_eq!(
                    Descriptor256(descriptor),
                    *our_descriptor,
                    "descriptor at {:?}",
                    k.position
                );
                inside += 1;
                textured += usize::from(descriptor != [0; 4]);
            }
        }
    }
    eprintln!("{inside} described keypoints, {textured} with a non-zero descriptor");
    assert!(
        inside > 30_000 && textured > 20_000,
        "{inside} described, {textured} non-zero"
    );
}

#[test]
#[ignore = "needs the Basalt BRIEF golden .rrd on the NAS (kornia-staging-golden task)"]
fn matches_are_basalts_minus_ties_and_unverified_mutual_checks() {
    let golden = BasaltBrief::load_default();
    let images = images();
    let descriptors: Vec<Vec<Descriptor256>> = described(&golden, &images)
        .into_iter()
        .map(|o| o.into_iter().flatten().map(|(_, d)| d).collect())
        .collect();
    let config = MatchConfig::default();
    let (mut total, mut basalt_only) = (0, 0);
    assert_eq!(golden.matches.len(), 5);
    for m in &golden.matches {
        let (a, b, basalt) = (m.image_a, m.image_b, &m.pairs);
        let (da, db) = (&descriptors[a], &descriptors[b]);
        let ours = match_mutual(da, db, config);
        // Ours is a subset of Basalt's; each pair only Basalt keeps is a tie for the best (B17), or a
        // pair whose reverse search found nothing and passed by reading index 0 (B6).
        assert!(
            ours.iter().all(|m| basalt.contains(m)),
            "images {a}-{b}: a match Basalt does not have"
        );
        for &(i, j) in basalt.iter().filter(|m| !ours.contains(m)) {
            let best = |q: &Descriptor256, set: &[Descriptor256]| {
                let mut d: Vec<u32> = set.iter().map(|x| q.hamming(x)).collect();
                d.sort_unstable();
                (d[0], d.get(1).copied())
            };
            let (forward, forward_second) = best(&da[i], db);
            let (reverse, reverse_second) = best(&db[j], da);
            let tie = forward_second == Some(forward) || reverse_second == Some(reverse);
            assert!(
                tie || i == 0,
                "images {a}-{b}: Basalt-only pair ({i}, {j}) is neither a tie nor B6"
            );
        }
        total += ours.len();
        basalt_only += basalt.len() - ours.len();
    }
    eprintln!("{total} matches over 5 image pairs, {basalt_only} more in Basalt");
    assert!(total > 250, "{total} matches");
}
