//! Basalt BRIEF at 500 keypoints of a 1280x960 u16 image, and mutual matching of two such sets (one loop-closing
//! keyframe pair of one camera in RootSLAM).

use criterion::{criterion_group, criterion_main, Criterion};
use kornia_image::{Image, ImageSize};
use kornia_staging_imgproc::features::{
    describe_keypoints_u16, match_mutual, BriefConfig, Descriptor256, MatchConfig,
};

fn image(width: usize, height: usize) -> Image<u16, 1> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let pixels = (0..width * height)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 48) as u16
        })
        .collect();
    Image::new(ImageSize { width, height }, pixels).expect("size matches")
}

fn keypoints(count: usize) -> Vec<[f32; 2]> {
    (0..count)
        .map(|i| {
            [
                30.0 + (i * 37 % 1200) as f32 + 0.3,
                30.0 + (i * 53 % 880) as f32 + 0.6,
            ]
        })
        .collect()
}

fn descriptors(
    image: &Image<u16, 1>,
    keypoints: &[[f32; 2]],
    config: BriefConfig,
) -> Vec<Descriptor256> {
    describe_keypoints_u16(image, keypoints, config)
        .into_iter()
        .flatten()
        .map(|d| d.descriptor)
        .collect()
}

fn brief(c: &mut Criterion) {
    let image = image(1280, 960);
    let points = keypoints(500);
    c.bench_function("brief_basalt_500", |b| {
        b.iter(|| describe_keypoints_u16(&image, &points, BriefConfig::basalt()))
    });
    c.bench_function("brief_box5_500", |b| {
        b.iter(|| describe_keypoints_u16(&image, &points, BriefConfig::default()))
    });
    let a = descriptors(&image, &points, BriefConfig::basalt());
    let other: Vec<[f32; 2]> = points.iter().map(|&[x, y]| [x + 1.0, y]).collect();
    let b_set = descriptors(&image, &other, BriefConfig::basalt());
    c.bench_function("match_mutual_500x500", |b| {
        b.iter(|| match_mutual(&a, &b_set, MatchConfig::default()))
    });
}

criterion_group!(benches, brief);
criterion_main!(benches);
