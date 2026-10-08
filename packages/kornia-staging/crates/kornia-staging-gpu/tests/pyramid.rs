//! Exact CPU/GPU pyramid contract, moved from the consumer.
#![allow(unsafe_code)]
#![cfg(feature = "wgpu")]
use images::textured_image;
use kornia_image::{Image, ImageSize};
use kornia_staging_gpu::{
    pyramid::{GpuPyramid, GpuPyramidBuilder, PyramidError},
    runtime::gpu_client,
    GpuRuntime,
};
use kornia_staging_imgproc::pyramid::{PyramidPlanError, PyramidPlanU16};
use kornia_staging_imgproc::test_fixtures as images;
const LEVELS: usize = 3;

/// `image`'s pyramid on both lanes, `LEVELS` deep and the same geometry.
fn both_pyramids(image: &Image<u16, 1>) -> (PyramidPlanU16, GpuPyramid<GpuRuntime>) {
    let (width, height): (usize, usize) = (image.width(), image.height());

    let mut cpu = PyramidPlanU16::new(image.size(), LEVELS).unwrap();
    cpu.run(image).unwrap();

    let mut gpu_builder: GpuPyramidBuilder<GpuRuntime> =
        GpuPyramidBuilder::new(gpu_client().unwrap());
    let mut gpu: GpuPyramid<GpuRuntime> = gpu_builder.allocate(width, height, LEVELS).unwrap();
    gpu_builder.build(0, image, &mut gpu).unwrap();

    (cpu, gpu)
}

/// Every level of the two pyramids holds the same geometry and the same pixels.
///
/// The message names the worst pixel and where it is, so a regression says how
/// far it moved rather than only that it moved. Equality is the bound: the
/// arithmetic is integer on both lanes (module doc).
fn assert_levels_equal(cpu: &PyramidPlanU16, gpu: &GpuPyramid<GpuRuntime>, label: &str) {
    assert_eq!(gpu.num_levels(), cpu.levels().len(), "{label}");
    let mut actual: Image<u16, 1> = Image::from_size_val(
        kornia_image::ImageSize {
            width: 0,
            height: 0,
        },
        0u16,
    )
    .unwrap();
    for level in 0..cpu.levels().len() {
        assert_eq!(
            gpu.level_size(level),
            cpu.levels().get(level).map(|image| image.size()),
            "{label}, level {level}"
        );
        let expected = &cpu.levels()[level];
        gpu.read_level_into(level, &mut actual).unwrap();
        let mut worst: i64 = 0;
        let mut worst_at: (usize, usize) = (0, 0);
        for y in 0..expected.height() {
            for x in 0..expected.width() {
                let difference: i64 = i64::from(actual.get_pixel(x, y, 0).copied().ok().unwrap())
                    - i64::from(expected.get_pixel(x, y, 0).copied().ok().unwrap());
                if difference.abs() > worst {
                    worst = difference.abs();
                    worst_at = (x, y);
                }
            }
        }
        assert_eq!(
            worst,
            0,
            "{label}, level {level} differs: max-abs-diff {worst} at {worst_at:?} \
             ({}x{}); the fused 5x5 pass is integer arithmetic and must be exact",
            expected.width(),
            expected.height()
        );
    }
}

#[test]
fn the_gpu_pyramid_is_bit_exact_with_the_cpu() {
    for (width, height) in [(960, 960), (640, 480), (517, 193), (64, 64)] {
        let image: Image<u16, 1> = textured_image(width, height, 0.0, 0.0);
        let (cpu, gpu) = both_pyramids(&image);
        assert_levels_equal(&cpu, &gpu, &format!("{width}x{height}"));
    }
}

#[test]
fn batched_gpu_pyramids_keep_every_camera_pixel_exact() {
    let client = gpu_client().unwrap();
    let mut builder = GpuPyramidBuilder::new(client.clone());
    for (width, height) in [(640, 480), (517, 193), (65, 67)] {
        let mut pyramids: Vec<_> = (0..4)
            .map(|_| builder.allocate(width, height, LEVELS).unwrap())
            .collect();
        for frame in 0..2 {
            let images: Vec<_> = (0..4)
                .map(|camera| {
                    packed_texture(width, height, (camera * 7 + frame) as f32, camera as f32)
                })
                .collect();
            builder.build_images(&images, &mut pyramids).unwrap();
            for (camera, image) in images.iter().enumerate() {
                let mut expected = PyramidPlanU16::new(image.size(), LEVELS).unwrap();
                expected.run(image).unwrap();
                assert_levels_equal(
                    &expected,
                    &pyramids[camera],
                    &format!("camera {camera}, frame {frame}"),
                );
            }
            let mut bytes: Vec<u8> = images
                .iter()
                .flat_map(|image| image.as_slice().iter().map(|&v| (v >> 8) as u8))
                .collect();
            bytes.resize(bytes.len().next_multiple_of(4), 0);
            let work = builder
                .prepare_packed(
                    &bytes::Bytes::from(bytes),
                    images.iter().map(|image| image.size()),
                    &mut pyramids,
                )
                .unwrap()
                .unwrap();
            // SAFETY: This is the same client used to prepare the batch.
            unsafe {
                work.run(&client);
            }
            for (camera, image) in images.iter().enumerate() {
                let mut expected = PyramidPlanU16::new(image.size(), LEVELS).unwrap();
                expected.run(image).unwrap();
                assert_levels_equal(
                    &expected,
                    &pyramids[camera],
                    &format!("camera {camera}, frame {frame}"),
                );
            }
        }
    }
}

#[test]
fn prepared_gpu_pixels_survive_reusing_the_source_image() {
    let mut image = textured_image(96, 96, 0.0, 0.0);
    let mut builder = GpuPyramidBuilder::new(gpu_client().unwrap());
    let mut pyramid = builder.allocate(96, 96, 1).unwrap();
    let expected = image.clone();
    builder
        .prepare_images(std::slice::from_ref(&image))
        .unwrap();
    image.as_slice_mut().fill(0);
    builder.build(0, &image, &mut pyramid).unwrap();
    let mut actual = Image::from_size_val(
        kornia_image::ImageSize {
            width: 0,
            height: 0,
        },
        0u16,
    )
    .unwrap();
    pyramid.read_level_into(0, &mut actual).unwrap();
    assert_eq!(actual.as_slice(), expected.as_slice());
}

/// A pyramid of level 0 alone is refused rather than allocated empty.
///
/// With `optical_flow_levels = 0` the odd buffer holds no level, and
/// `client.empty(0)` is a zero-sized allocation that wgpu rejects at
/// validation — on cubecl's own worker thread, where a panic reaches the caller
/// as data rather than as an error. That is the failure mode `probe_storage`
/// exists to prevent, reachable through a config value instead, so the geometry
/// is refused the way every other unbuildable one is.
#[test]
fn a_pyramid_of_one_level_is_refused_rather_than_allocated_empty() {
    let builder = GpuPyramidBuilder::new(gpu_client().unwrap());
    let refused = builder.allocate(64, 48, 0);
    assert!(
        matches!(
            refused,
            Err(PyramidError::Geometry(PyramidPlanError::TooSmall {
                width: 64,
                height: 48,
                max_level: 0
            }))
        ),
        "a single-level pyramid was accepted: {refused:?}"
    );
}

/// The pyramid the builder allocates is reused frame after frame, so the second
/// frame must not see the first one's pixels anywhere.
#[test]
fn a_reused_pyramid_carries_only_the_newest_frame() {
    let first: Image<u16, 1> = textured_image(128, 96, 0.0, 0.0);
    let second: Image<u16, 1> = textured_image(128, 96, 7.0, -3.0);

    let mut gpu_builder: GpuPyramidBuilder<GpuRuntime> =
        GpuPyramidBuilder::new(gpu_client().unwrap());
    let mut gpu: GpuPyramid<GpuRuntime> = gpu_builder.allocate(128, 96, LEVELS).unwrap();
    gpu_builder.build(0, &first, &mut gpu).unwrap();
    gpu_builder.build(0, &second, &mut gpu).unwrap();

    let mut cpu = PyramidPlanU16::new(second.size(), LEVELS).unwrap();
    cpu.run(&second).unwrap();

    assert_levels_equal(&cpu, &gpu, "the second frame of a reused pyramid");
}

#[test]
fn dense_u16_inputs_keep_the_general_path_and_all_low_bits() {
    let client = gpu_client().unwrap();
    let mut builder = GpuPyramidBuilder::new(client);
    for shift_only in [true, false] {
        let pixels: Vec<u16> = (0..64 * 48)
            .map(|i| {
                if shift_only {
                    ((i % 256) as u16) << 8
                } else {
                    (i * 17) as u16
                }
            })
            .collect();
        let image = Image::new(
            ImageSize {
                width: 64,
                height: 48,
            },
            pixels,
        )
        .unwrap();
        let mut pyramids = vec![builder.allocate(64, 48, 2).unwrap()];
        builder
            .build_images(std::slice::from_ref(&image), &mut pyramids)
            .unwrap();
        assert!(pyramids[0].arena().is_none());
        let mut level0 = Image::from_size_val(
            ImageSize {
                width: 0,
                height: 0,
            },
            0u16,
        )
        .unwrap();
        pyramids[0].read_level_into(0, &mut level0).unwrap();
        assert_eq!(level0.as_slice(), image.as_slice());
    }
}

#[test]
fn prepared_batches_keep_distinct_pixels_before_dispatch() {
    let client = gpu_client().unwrap();
    let mut builder = GpuPyramidBuilder::new(client.clone());
    let size = ImageSize {
        width: 64,
        height: 48,
    };
    let mut first = [builder.allocate(size.width, size.height, 2).unwrap()];
    let mut second = [builder.allocate(size.width, size.height, 2).unwrap()];
    let a = builder
        .prepare_packed(
            &bytes::Bytes::from(vec![7; size.width * size.height]),
            [size].into_iter(),
            &mut first,
        )
        .unwrap()
        .unwrap();
    let b = builder
        .prepare_packed(
            &bytes::Bytes::from(vec![93; size.width * size.height]),
            [size].into_iter(),
            &mut second,
        )
        .unwrap()
        .unwrap();
    // SAFETY: Both batches were prepared on this same client.
    unsafe {
        a.run(&client);
        b.run(&client);
    }
    for (pyramids, value) in [(first, 7u16 << 8), (second, 93u16 << 8)] {
        let mut actual = Image::from_size_val(size, 0u16).unwrap();
        for level in 0..3 {
            pyramids[0].read_level_into(level, &mut actual).unwrap();
            assert!(
                actual.as_slice().iter().all(|&pixel| pixel == value),
                "prepared frame changed at level {level}"
            );
        }
    }
}

#[test]
fn dense_batches_reject_mismatched_camera_counts() {
    let size = ImageSize {
        width: 64,
        height: 48,
    };
    let mut builder = GpuPyramidBuilder::new(gpu_client().unwrap());
    for (inputs, outputs) in [(1, 2), (2, 1)] {
        let images = vec![Image::from_size_val(size, 1234u16).unwrap(); inputs];
        let mut pyramids: Vec<_> = (0..outputs)
            .map(|_| builder.allocate(size.width, size.height, 2).unwrap())
            .collect();
        assert!(
            builder.build_images(&images, &mut pyramids).is_err(),
            "accepted {inputs} inputs and {outputs} outputs"
        );
    }
}

use kornia_staging_imgproc::test_fixtures::packed_texture;
