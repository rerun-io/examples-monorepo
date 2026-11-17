//! BRIEF and matcher parity with Basalt's own `computeAngles`, `computeDescriptors` and `matchDescriptors`.
//!
//! The reference is the output of `fixtures/basalt_brief/basalt_brief_dump.cpp`, built against RootSLAM's
//! `keypoints.cpp` (branch pixi @ 839629d) and run on `fixtures/frames/*.pgm` plus a synthetic u16 image. It lives on
//! the NAS; the tests skip, naming it, when it is absent.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use kornia_image::{Image, ImageSize};

use super::{describe_keypoints_u16, match_mutual, BriefConfig, Descriptor256, MatchConfig};
use crate::test_fixtures::read_pgm;

/// The golden file: `KORNIA_STAGING_BASALT_BRIEF`, else the RootSLAM port's scratch folder on the NAS.
pub(crate) fn golden_path() -> Option<PathBuf> {
    let path = std::env::var_os("KORNIA_STAGING_BASALT_BRIEF")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("/mnt/nas/ssd-8tb/scratch/slam-rs-rootslam/fixtures/b1/basalt_brief.txt")
        });
    if path.is_file() {
        Some(path)
    } else {
        eprintln!("skipped: Basalt BRIEF golden {} is absent", path.display());
        None
    }
}

/// One keypoint line: position, and Basalt's angle and descriptor when it was inside the 19-pixel border.
struct GoldenKeypoint {
    position: [f32; 2],
    described: Option<(f64, Descriptor256)>,
}

struct GoldenImage {
    keypoints: Vec<GoldenKeypoint>,
}

/// Matched index pairs `(a, b)`.
type Pairs = Vec<(usize, usize)>;
/// One image's angle and descriptor per keypoint, `None` outside the border.
type Described = Vec<Option<(f64, Descriptor256)>>;

struct Golden {
    images: Vec<GoldenImage>,
    /// `(image a, image b, matches)` as Basalt's `matchDescriptors` returned them, sorted by the `a` index.
    matches: Vec<(usize, usize, Pairs)>,
}

fn parse(path: &Path) -> Golden {
    let text = std::fs::read_to_string(path).unwrap();
    let mut golden = Golden {
        images: Vec::new(),
        matches: Vec::new(),
    };
    for line in text.lines() {
        let fields: Vec<&str> = line.split(' ').collect();
        match fields[0] {
            "image" => golden.images.push(GoldenImage {
                keypoints: Vec::new(),
            }),
            "kp" => {
                let position = [fields[1].parse().unwrap(), fields[2].parse().unwrap()];
                let described = (fields[3] != "out").then(|| {
                    let words: Vec<u64> = fields[4..8]
                        .iter()
                        .map(|w| u64::from_str_radix(w, 16).unwrap())
                        .collect();
                    (
                        fields[3].parse().unwrap(),
                        Descriptor256(words.try_into().unwrap()),
                    )
                });
                golden
                    .images
                    .last_mut()
                    .unwrap()
                    .keypoints
                    .push(GoldenKeypoint {
                        position,
                        described,
                    });
            }
            "match" => {
                let pairs = fields[4..]
                    .iter()
                    .map(|p| {
                        let (a, b) = p.split_once(':').unwrap();
                        (a.parse().unwrap(), b.parse().unwrap())
                    })
                    .collect::<Vec<_>>();
                assert_eq!(pairs.len(), fields[3].parse::<usize>().unwrap());
                golden.matches.push((
                    fields[1].parse().unwrap(),
                    fields[2].parse().unwrap(),
                    pairs,
                ));
            }
            _ => {}
        }
    }
    golden
}

/// The dump's inputs: each frame fixture widened with `v << 8`, then the synthetic image.
fn images() -> Vec<Image<u16, 1>> {
    let frames = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/frames");
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
fn described(golden: &Golden, images: &[Image<u16, 1>]) -> Vec<Described> {
    golden
        .images
        .iter()
        .zip(images)
        .map(|(g, image)| {
            let positions: Vec<[f32; 2]> = g.keypoints.iter().map(|k| k.position).collect();
            describe_keypoints_u16(image, &positions, BriefConfig::basalt())
                .into_iter()
                .map(|d| d.map(|d| (d.angle, d.descriptor)))
                .collect()
        })
        .collect()
}

#[test]
fn angles_and_descriptors_are_bit_identical_to_basalt() {
    let Some(path) = golden_path() else { return };
    let golden = parse(&path);
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
                    descriptor, *our_descriptor,
                    "descriptor at {:?}",
                    k.position
                );
                inside += 1;
                textured += usize::from(descriptor != Descriptor256::default());
            }
        }
    }
    assert!(
        inside > 30_000 && textured > 20_000,
        "{inside} described, {textured} non-zero"
    );
}

#[test]
fn matches_are_basalts_minus_ties_and_unverified_mutual_checks() {
    let Some(path) = golden_path() else { return };
    let golden = parse(&path);
    let images = images();
    let descriptors: Vec<Vec<Descriptor256>> = described(&golden, &images)
        .into_iter()
        .map(|o| o.into_iter().flatten().map(|(_, d)| d).collect())
        .collect();
    let config = MatchConfig::default();
    let (mut total, mut basalt_only) = (0, 0);
    // The first two image pairs: brute force over ~6,700 descriptors each is slow in a debug build.
    for (a, b, basalt) in golden.matches.iter().take(2) {
        let (da, db) = (&descriptors[*a], &descriptors[*b]);
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
    eprintln!("{total} matches, {basalt_only} more in Basalt");
    assert!(total > 250, "{total} matches");
}
