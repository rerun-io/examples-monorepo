//! The networks' inputs, written in one pass, are byte-identical to the chains they replace.
//!
//! DetNet: the old chain padded each 640x360 small image into a 640x480 net frame (kornia `spatial_padding`, 60 black rows above
//! and below), then per backend: RKNN pooled it 4x4 (`pool4_u8` for an INT8 model; `pool4_mean_f32` for FP16, then
//! `f16_bytes_from_f32` at 1/255 for the native fp16 feed), ORT took the frame as `u8 / 255` floats. The new pass
//! ([`PooledInput::fill`], [`NetFrame::write_unit_f32`]) reads the small image's rows and never builds the padded frame.
//!
//! KeyNet: [`CropInput::fill`] writes the [0, 1] float crop into the one buffer of the model's feed; its bytes are those of the
//! per-feed conversions the worker made before (`u8_from_unit_f32`; `x * 255`; `f16_bytes_from_f32` at scale 1).
//!
//! Inputs: synthetic images (constants, step edges off the 4-pixel blocks, block sums on the round-half-even ties, noise), the
//! s66 frame pixels in `tests/data/perception`, and, when `ROBOCAP_LIVE_DUMP_DIR` names a `robocap-live-dump/1` directory, every
//! camera of every frameset in it.

use std::path::{Path, PathBuf};

use kornia_image::Image;
use kornia_imgproc::padding::{Padding2D, PaddingMode, spatial_padding};
use robocap_live::downsample::resize_area_u8;
use robocap_live::kornia_ext::pool::{pool4_mean_f32, pool4_u8};
use robocap_live::downsample::{SmallImagePool, small_images};
use robocap_live::frame::{FULL_SIZE, FrameReader, SMALL_SIZE};
use robocap_live::hands::letterbox::{BarLetterbox, NET_SIZE};
use robocap_live::nets::rknn::{
    CropInput, ImageFeed, InputData, POOLED_LEN, PooledInput, f16_bytes_from_f32, u8_from_unit_f32,
};
use robocap_live::nets::{CROP_LEN, DETNET_HEIGHT, DETNET_WIDTH, NetFrame};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// The backend inputs of the old chain: RKNN's three feeds and ORT's tensor.
struct OldInputs {
    pooled_u8: Vec<u8>,
    pooled_f32: Vec<f32>,
    pooled_f16: Vec<u8>,
    ort: Vec<f32>,
}

fn old_detnet_inputs(small: &Image<u8, 1>) -> Result<OldInputs, Box<dyn std::error::Error>> {
    let mut net = Image::<u8, 1>::from_size_val(NET_SIZE, 0)?;
    spatial_padding(
        small,
        &mut net,
        Padding2D {
            top: 60,
            bottom: 60,
            left: 0,
            right: 0,
        },
        PaddingMode::Constant,
        [0u8],
    )?;
    let mut pooled_u8 = vec![0u8; POOLED_LEN];
    pool4_u8(net.as_slice(), DETNET_WIDTH, DETNET_HEIGHT, &mut pooled_u8)?;
    let mut pooled_f32 = vec![0f32; POOLED_LEN];
    pool4_mean_f32(net.as_slice(), DETNET_WIDTH, DETNET_HEIGHT, &mut pooled_f32)?;
    let mut pooled_f16 = vec![0u8; 2 * POOLED_LEN];
    f16_bytes_from_f32(&pooled_f32, 1.0 / 255.0, &mut pooled_f16)?;
    let ort: Vec<f32> = net
        .as_slice()
        .iter()
        .map(|&value| f32::from(value) / 255.0)
        .collect();
    Ok(OldInputs {
        pooled_u8,
        pooled_f32,
        pooled_f16,
        ort,
    })
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// The one-pass inputs of `small` equal the old chain's, byte for byte, in every feed. Each buffer first holds the input of a
/// white full 640x480 frame, so a pass that leaves the bars or any value unwritten fails.
fn assert_detnet_inputs_match(small: &Image<u8, 1>, what: &str) -> TestResult {
    let old: OldInputs = old_detnet_inputs(small)?;
    let letterbox = BarLetterbox::robocap();
    let white = vec![255u8; DETNET_WIDTH * DETNET_HEIGHT];
    let frame = letterbox.net_frame(small)?;
    for feed in [ImageFeed::U8, ImageFeed::F32, ImageFeed::F16Native] {
        let mut input = PooledInput::new(feed);
        input.fill(&NetFrame {
            pixels: &white,
            top: 0,
        })?;
        match input.fill(&frame)? {
            InputData::U8(values) => assert!(
                values == old.pooled_u8.as_slice(),
                "{what}: u8 feed differs"
            ),
            InputData::F32(values) => assert!(
                bits(values) == bits(&old.pooled_f32),
                "{what}: f32 feed differs"
            ),
            InputData::Native(bytes) => assert!(
                bytes == old.pooled_f16.as_slice(),
                "{what}: fp16 feed differs"
            ),
        }
    }
    let mut ort = vec![1f32; DETNET_WIDTH * DETNET_HEIGHT];
    frame.write_unit_f32(&mut ort)?;
    assert!(bits(&ort) == bits(&old.ort), "{what}: ORT tensor differs");
    Ok(())
}

fn small_from(
    value: impl Fn(usize, usize) -> u8,
) -> Result<Image<u8, 1>, kornia_image::ImageError> {
    let (width, height) = (SMALL_SIZE.width, SMALL_SIZE.height);
    Image::new(
        SMALL_SIZE,
        (0..width * height)
            .map(|i| value(i % width, i / width))
            .collect(),
    )
}

#[test]
fn detnet_input_matches_the_padded_chain_on_synthetic_images() -> TestResult {
    let mut state: u64 = 7;
    let mut noise = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 56) as u8
    };
    let noisy: Vec<u8> = (0..SMALL_SIZE.width * SMALL_SIZE.height)
        .map(|_| noise())
        .collect();
    let images: Vec<(&str, Image<u8, 1>)> = vec![
        ("black", small_from(|_, _| 0)?),
        ("white", small_from(|_, _| 255)?),
        ("constant 7", small_from(|_, _| 7)?),
        ("constant 129", small_from(|_, _| 129)?),
        // Steps one pixel off the 4x4 blocks, both axes, and a one-pixel checkerboard of extremes.
        (
            "vertical edge",
            small_from(|x, _| if x >= 321 { 255 } else { 0 })?,
        ),
        (
            "horizontal edge",
            small_from(|_, y| if y >= 179 { 200 } else { 3 })?,
        ),
        (
            "first and last rows",
            small_from(|_, y| if y == 0 || y == 359 { 255 } else { 0 })?,
        ),
        (
            "checkerboard",
            small_from(|x, y| if (x + y) % 2 == 0 { 255 } else { 0 })?,
        ),
        // Block sums of 8 mod 16 (the round-half-even ties of the u8 pool): one pixel of 8, 24, 40 ... per block.
        (
            "ties",
            small_from(|x, y| {
                if x % 4 == 0 && y % 4 == 0 {
                    (8 + 16 * ((x / 4 + y) % 15)) as u8
                } else {
                    0
                }
            })?,
        ),
        ("ramp", small_from(|x, y| ((x + 3 * y) % 256) as u8)?),
        ("noise", Image::new(SMALL_SIZE, noisy)?),
    ];
    for (what, image) in &images {
        assert_detnet_inputs_match(image, what)?;
    }
    Ok(())
}

/// The s66 frames stored in `tests/data/perception`: their pixels inside the stored crop footprints, black elsewhere, as 1080p
/// frames (one per entry).
fn stored_s66_frames() -> Result<Vec<Image<u8, 1>>, Box<dyn std::error::Error>> {
    let dir: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/perception");
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)?;
    let bytes: Vec<u8> = std::fs::read(dir.join("frame_pixels_u8.bin"))?;
    let mut frames: Vec<Image<u8, 1>> = Vec::new();
    for entry in manifest["frame_pixels"]["entries"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let mut inside = vec![false; FULL_SIZE.width * FULL_SIZE.height];
        for rect in entry["rects"].as_array().into_iter().flatten() {
            let r: Vec<usize> = rect
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|v| v.as_u64())
                .map(|v| v as usize)
                .collect();
            for y in r[1]..r[3] {
                inside[y * FULL_SIZE.width + r[0]..y * FULL_SIZE.width + r[2]].fill(true);
            }
        }
        let mut source = bytes[entry["offset"].as_u64().ok_or("offset")? as usize..].iter();
        let mut data = vec![0u8; FULL_SIZE.width * FULL_SIZE.height];
        for (pixel, _) in data.iter_mut().zip(&inside).filter(|(_, inside)| **inside) {
            *pixel = *source.next().ok_or("short frame data")?;
        }
        frames.push(Image::new(FULL_SIZE, data)?);
    }
    Ok(frames)
}

#[test]
fn detnet_input_matches_the_padded_chain_on_stored_s66_frames() -> TestResult {
    let frames: Vec<Image<u8, 1>> = stored_s66_frames()?;
    assert!(!frames.is_empty());
    for (index, full) in frames.iter().enumerate() {
        let mut small = Image::<u8, 1>::from_size_val(SMALL_SIZE, 0)?;
        resize_area_u8(full, &mut small)?;
        assert_detnet_inputs_match(&small, &format!("stored s66 frame {index}"))?;
    }
    Ok(())
}

#[test]
fn detnet_input_matches_the_padded_chain_on_a_dump() -> TestResult {
    let Some(dir) = std::env::var_os("ROBOCAP_LIVE_DUMP_DIR").map(PathBuf::from) else {
        eprintln!(
            "SKIP detnet_input_matches_the_padded_chain_on_a_dump: set ROBOCAP_LIVE_DUMP_DIR to a dump (e.g. the s66 clip)"
        );
        return Ok(());
    };
    let mut reader = FrameReader::open(&dir.join("frames.bin"), FULL_SIZE)?;
    let mut checked: usize = 0;
    while let Some(frameset) = reader.next_frameset()? {
        for (camera, small) in small_images(&frameset, None, &mut SmallImagePool::default())?
            .iter()
            .enumerate()
        {
            if let Some(small) = small {
                assert_detnet_inputs_match(
                    small,
                    &format!("frameset {} camera {camera}", frameset.index),
                )?;
                checked += 1;
            }
        }
    }
    eprintln!("{checked} dump images: one-pass DetNet inputs identical in every feed");
    assert!(checked > 0);
    Ok(())
}

/// KeyNet's crop input in each feed, written into a buffer that first held another crop, equals the per-feed conversion of the
/// crop, byte for byte, in the feed's input type.
#[test]
fn keynet_crop_input_matches_the_per_feed_conversions() -> TestResult {
    let mut state: u64 = 11;
    let mut noise = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    let crops: Vec<(&str, Vec<f32>)> = vec![
        ("black", vec![0.0; CROP_LEN]),
        ("white", vec![1.0; CROP_LEN]),
        (
            "ramp",
            (0..CROP_LEN)
                .map(|i| i as f32 / (CROP_LEN - 1) as f32)
                .collect(),
        ),
        // Halfway between two u8 steps (the rounding ties), and values outside [0, 1] (the u8 feed clamps them).
        (
            "ties",
            (0..CROP_LEN)
                .map(|i| ((i % 255) as f32 + 0.5) / 255.0)
                .collect(),
        ),
        (
            "out of range",
            (0..CROP_LEN)
                .map(|i| if i % 2 == 0 { -0.2 } else { 1.3 })
                .collect(),
        ),
        ("noise", (0..CROP_LEN).map(|_| noise()).collect()),
    ];
    let other: Vec<f32> = vec![0.25; CROP_LEN];
    for (what, crop) in &crops {
        let mut u8_values = vec![0u8; CROP_LEN];
        u8_from_unit_f32(crop, &mut u8_values)?;
        let f32_values: Vec<f32> = crop.iter().map(|&value| value * 255.0).collect();
        let mut f16_bytes = vec![0u8; 2 * CROP_LEN];
        f16_bytes_from_f32(crop, 1.0, &mut f16_bytes)?;
        for feed in [ImageFeed::U8, ImageFeed::F32, ImageFeed::F16Native] {
            let mut input = CropInput::new(feed);
            input.fill(&other)?;
            match (feed, input.fill(crop)?) {
                (ImageFeed::U8, InputData::U8(values)) => {
                    assert!(values == u8_values.as_slice(), "{what}: u8 feed differs")
                }
                (ImageFeed::F32, InputData::F32(values)) => assert!(
                    bits(values) == bits(&f32_values),
                    "{what}: f32 feed differs"
                ),
                (ImageFeed::F16Native, InputData::Native(bytes)) => {
                    assert!(bytes == f16_bytes.as_slice(), "{what}: fp16 feed differs")
                }
                (feed, _) => {
                    return Err(format!("{what}: the {feed:?} feed gave another input type").into());
                }
            }
            assert!(
                input.fill(&crop[1..]).is_err(),
                "{what}: a short crop is refused"
            );
        }
    }
    Ok(())
}
