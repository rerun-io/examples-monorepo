//! Downsample 1920x1080 luma -> 640x360: our exact area /3 (`resize_area_u8`) against kornia-imgproc's `resize_fast_mono_aa`
//! (bicubic + antialias, and plain bilinear), per core, one thread; then a whole 6-camera frameset on a pinned worker pool.
//!
//! Usage: `downsample_bench [--frames <frames.bin of a dump>] [--cores 0,4] [--pool 0-3] [--threads 3] [--iterations 200]`

use std::sync::Arc;
use std::time::Instant;

use kornia_image::Image;
use kornia_imgproc::interpolation::InterpolationMode;
use kornia_imgproc::resize::resize_fast_mono_aa;
use kornia_staging_imgproc::resize::resize_area_u8;
use robocap_live::downsample::{SmallImagePool, small_images};
use robocap_live::frame::{
    CameraFrame, FULL_SIZE, FrameMeta, FrameReader, Frameset, NUM_CAMERAS, SMALL_SIZE,
};
use robocap_live::sched::{parse_cpu_list, pin_current_thread};

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn time_ms(
    iterations: usize,
    mut f: impl FnMut() -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
) -> Result<(f64, f64), Box<dyn std::error::Error + Send + Sync>> {
    f()?;
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let started = Instant::now();
        f()?;
        samples.push(started.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(f64::total_cmp);
    Ok((
        samples.iter().sum::<f64>() / samples.len() as f64,
        samples[samples.len() * 95 / 100],
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let iterations: usize = arg("--iterations").map_or(Ok(200), |s| s.parse())?;
    let cores =
        parse_cpu_list(&arg("--cores").unwrap_or_else(|| "0,4".into()))?.unwrap_or_default();
    let pool_cpus = parse_cpu_list(&arg("--pool").unwrap_or_else(|| "0-3".into()))?;
    let threads: usize = arg("--threads").map_or(Ok(3), |s| s.parse())?;
    let frameset = match arg("--frames") {
        Some(path) => FrameReader::open(std::path::Path::new(&path), FULL_SIZE)?
            .next_frameset()?
            .ok_or("empty frames.bin")?,
        None => {
            let mut state = 1u64;
            let mut cameras: [Option<CameraFrame>; NUM_CAMERAS] = Default::default();
            for (camera, slot) in cameras.iter_mut().enumerate() {
                let data: Vec<u8> = (0..FULL_SIZE.width * FULL_SIZE.height)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        (state >> 56) as u8
                    })
                    .collect();
                let meta = FrameMeta {
                    seq: 0,
                    pts_ns: 0,
                    source_id: camera as u32,
                    turned_180: false,
                };
                *slot = Some(CameraFrame {
                    meta,
                    full: Arc::new(Image::new(FULL_SIZE, data)?),
                });
            }
            Frameset {
                index: 0,
                t_ns: 0,
                cameras,
            }
        }
    };
    let full = frameset
        .cameras
        .iter()
        .flatten()
        .next()
        .ok_or("no camera")?
        .full
        .clone();
    let mut small = Image::<u8, 1>::from_size_val(SMALL_SIZE, 0)?;
    for core in cores {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .start_handler(move |_| {
                let _ = pin_current_thread(&[core]);
            })
            .build()?;
        pool.install(|| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            let (area, area95) = time_ms(iterations, || Ok(resize_area_u8(&full, &mut small)?))?;
            let (cubic, cubic95) = time_ms(iterations / 4 + 1, || Ok(resize_fast_mono_aa(&full, &mut small, InterpolationMode::Bicubic, true)?))?;
            let (linear, linear95) = time_ms(iterations, || Ok(resize_fast_mono_aa(&full, &mut small, InterpolationMode::Bilinear, false)?))?;
            println!(
                "core {core}: area/3 {area:.3} ms (p95 {area95:.3}) | kornia bicubic+aa {cubic:.3} ms (p95 {cubic95:.3}) | kornia bilinear {linear:.3} ms (p95 {linear95:.3})"
            );
            Ok(())
        })?;
    }
    // Streaming from DRAM: copy the six distinct 2 MB planes, and downsample them one after another, on one pinned core.
    let sources: Vec<Arc<Image<u8, 1>>> = frameset
        .cameras
        .iter()
        .flatten()
        .map(|f| f.full.clone())
        .collect();
    for core in
        parse_cpu_list(&arg("--stream-cores").unwrap_or_else(|| "0,4".into()))?.unwrap_or_default()
    {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .start_handler(move |_| {
                let _ = pin_current_thread(&[core]);
            })
            .build()?;
        pool.install(|| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            let mut copies: Vec<Vec<u8>> = sources.iter().map(|s| vec![0u8; s.as_slice().len()]).collect();
            let (copy, copy95) = time_ms(iterations / 4 + 1, || {
                for (dst, src) in copies.iter_mut().zip(&sources) {
                    dst.copy_from_slice(src.as_slice());
                }
                Ok(())
            })?;
            let mut smalls: Vec<Image<u8, 1>> = sources.iter().map(|_| Image::from_size_val(SMALL_SIZE, 0)).collect::<Result<_, _>>()?;
            let (area, area95) = time_ms(iterations / 4 + 1, || {
                for (dst, src) in smalls.iter_mut().zip(&sources) {
                    resize_area_u8(src, dst)?;
                }
                Ok(())
            })?;
            let mb = sources.len() as f64 * 2.0736;
            println!(
                "core {core} streaming {} planes: memcpy {copy:.2} ms (p95 {copy95:.2}, {:.2} GB/s) | area/3 {area:.2} ms (p95 {area95:.2}, {:.2} GB/s read)",
                sources.len(),
                mb / copy,
                mb / area
            );
            Ok(())
        })?;
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .start_handler(move |_| {
            if let Some(cpus) = &pool_cpus {
                let _ = pin_current_thread(cpus);
            }
        })
        .build()?;
    let mut images = SmallImagePool::default();
    let (frameset_ms, frameset95) = pool.install(|| {
        time_ms(iterations, || {
            Ok(small_images(&frameset, None, &mut images).map(|_| ())?)
        })
    })?;
    println!(
        "frameset (6 cameras) on {threads} threads: {frameset_ms:.3} ms (p95 {frameset95:.3}) = {:.0} framesets/s",
        1e3 / frameset_ms
    );
    Ok(())
}
