//! Hand-net bench: loads a backend, checks it against the PyTorch golden files, times DetNet and KeyNet, and (optionally) runs a
//! whole held-out set and writes the raw outputs for host-side scoring (`tools/rknn_convert.py --score-device`).
//!
//! ```text
//! nets_bench rknn <models dir> [--detnet FILE] [--keynet FILE] [--keynet-b4 FILE] [--lib librknnrt.so] [--contexts 2]
//!                              [--iters 300] [--golden DIR] [--heldout DIR --out DIR]
//! nets_bench ort  <models dir> [--device cuda|cpu|auto] [--dylib libonnxruntime.so] [--iters 300] [--golden DIR] [--heldout DIR --out DIR]
//! ```
//!
//! Latencies are wall-clock per call through the `HandNets` trait (DetNet includes the CPU 4x4 pool on rknn). On rknn it also
//! prints the per-phase breakdown of a single context (`rknn_inputs_set` / `rknn_run` / `rknn_outputs_get` / driver NPU time).
//! The SoC temperature is checked as it runs; the bench stops above 85 C.

use std::path::{Path, PathBuf};
use std::time::Instant;

use robocap_live::nets::golden::{Comparison, Golden, default_dir, detnet_row, expand_pooled, f32_values, keynet_row};
use robocap_live::nets::rknn::{InputData, NpuCore, POOLED_HEIGHT, POOLED_WIDTH, RknnModel, RknnNets, RknnRuntime, RunTiming, f16_bytes_from_f32};
use robocap_live::kornia_ext::pool::{pool4_mean_f32, pool4_u8};
use robocap_live::sched::soc_temperature_c;
use robocap_live::nets::{DETNET_HEIGHT, DETNET_WIDTH, HandNets, KEYNET_CROP, KeyNetRaw, NetFrame};

type BenchResult<T> = Result<T, Box<dyn std::error::Error>>;

fn check_temperature() -> BenchResult<()> {
    match soc_temperature_c() {
        Some(t) if t > 85.0 => Err(format!("SoC at {t:.1} C > 85 C: stopping").into()),
        _ => Ok(()),
    }
}

struct Args {
    backend: String,
    models: PathBuf,
    options: Vec<(String, String)>,
}

impl Args {
    fn parse() -> BenchResult<Self> {
        let mut raw: Vec<String> = std::env::args().skip(1).collect();
        if raw.len() < 2 {
            return Err("usage: nets_bench <rknn|ort> <models dir> [--key value ...]".into());
        }
        let backend: String = raw.remove(0);
        let models: PathBuf = PathBuf::from(raw.remove(0));
        let mut options: Vec<(String, String)> = Vec::new();
        let mut items = raw.into_iter();
        while let Some(key) = items.next() {
            let value: String = items.next().ok_or_else(|| format!("{key} needs a value"))?;
            options.push((key.trim_start_matches("--").to_string(), value));
        }
        Ok(Self { backend, models, options })
    }

    fn get(&self, key: &str) -> Option<&str> {
        self.options.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }
}

/// p50 / p90 / mean / min of microsecond samples, as one line.
fn stats(name: &str, samples: &mut [f64]) -> String {
    if samples.is_empty() {
        return format!("{name:<36} (no samples)");
    }
    samples.sort_by(f64::total_cmp);
    let n: usize = samples.len();
    let mean: f64 = samples.iter().sum::<f64>() / n as f64;
    format!(
        "{name:<36} n={n:<5} p50 {:>7.3} ms  p90 {:>7.3} ms  mean {:>7.3} ms  min {:>7.3} ms",
        samples[n / 2] / 1e3,
        samples[(n * 9 / 10).min(n - 1)] / 1e3,
        mean / 1e3,
        samples[0] / 1e3
    )
}

fn time_calls<E: std::error::Error + 'static>(iters: usize, mut call: impl FnMut() -> Result<(), E>) -> BenchResult<Vec<f64>> {
    for _ in 0..iters.min(20) {
        call()?;
    }
    let mut samples: Vec<f64> = Vec::with_capacity(iters);
    for i in 0..iters {
        if i % 100 == 0 {
            check_temperature()?;
        }
        let started: Instant = Instant::now();
        call()?;
        samples.push(started.elapsed().as_secs_f64() * 1e6);
    }
    Ok(samples)
}

fn print_comparison(label: &str, c: &Comparison) {
    println!("golden vs PyTorch [{label}]:");
    println!(
        "  detnet: centre max {:.3} px, radius max {:.3} px, presence logit max {:.4}, presence flips {}",
        c.detnet_centre_px_max, c.detnet_radius_px_max, c.detnet_presence_logit_max, c.detnet_presence_flips
    );
    println!(
        "  keynet: heatmap max {:.5}, distance max {:.5}, keypoint shift mean {:.4} / max {:.4} crop px, presence logit max {:.4}, pinch prob max {:.5}",
        c.keynet_heatmap_max,
        c.keynet_distance_max,
        c.keynet_keypoint_px_mean,
        c.keynet_keypoint_px_max,
        c.keynet_presence_logit_max,
        c.keynet_pinch_probability_max
    );
}

fn golden_check(nets: &mut dyn HandNets, dir: &Path) -> BenchResult<()> {
    let golden: Golden = Golden::load(dir)?;
    let frames: Vec<Vec<u8>> = golden.detnet_frames();
    let frame_refs: Vec<NetFrame<'_>> = frames.iter().map(|frame| NetFrame { pixels: frame, top: 0 }).collect();
    let detnet = nets.detnet(&frame_refs)?;
    let crops: Vec<&[f32]> = golden.keynet_crops.iter().map(Vec::as_slice).collect();
    let batched: Vec<KeyNetRaw> = nets.keynet(&crops, &golden.keynet_keypoints)?;
    print_comparison("keynet as one batch", &Comparison::of(&golden, &detnet, &batched)?);
    let mut single: Vec<KeyNetRaw> = Vec::new();
    for (crop, prior) in crops.iter().zip(&golden.keynet_keypoints) {
        single.extend(nets.keynet(&[crop], std::slice::from_ref(prior))?);
    }
    let same: bool = single.iter().zip(&batched).all(|(a, b)| a == b);
    println!("  keynet batch == one-by-one: {same}");
    Ok(())
}

fn latency(nets: &mut dyn HandNets, iters: usize, golden_dir: &Path) -> BenchResult<()> {
    let golden: Golden = Golden::load(golden_dir)?;
    let frames: Vec<Vec<u8>> = golden.detnet_frames();
    let crop: &[f32] = &golden.keynet_crops[0];
    let prior: [f32; 63] = golden.keynet_keypoints[0];
    let mut detnet = time_calls(iters, || nets.detnet(&[NetFrame { pixels: &frames[0], top: 0 }]).map(|_| ()))?;
    println!("{}", stats("detnet 1 frame (trait, incl. pool)", &mut detnet));
    let frame_refs: Vec<NetFrame<'_>> = (0..6).map(|i| NetFrame { pixels: &frames[i % frames.len()], top: 0 }).collect();
    let mut detnet6 = time_calls(iters / 2, || nets.detnet(&frame_refs).map(|_| ()))?;
    println!("{}", stats("detnet 6 frames (trait)", &mut detnet6));
    for count in [1usize, 2, 4, 8] {
        let crops: Vec<&[f32]> = vec![crop; count];
        let priors: Vec<[f32; 63]> = vec![prior; count];
        let mut samples = time_calls(iters, || nets.keynet(&crops, &priors).map(|_| ()))?;
        let line: String = stats(&format!("keynet {count} crop(s) (trait)"), &mut samples);
        let p50: f64 = samples[samples.len() / 2];
        println!("{line}  -> {:.0} crops/s at p50", count as f64 / (p50 / 1e6));
    }
    Ok(())
}

fn rknn_phases(model: &mut RknnModel, inputs: &[InputData<'_>], iters: usize, label: &str) -> BenchResult<()> {
    let mut buffers: Vec<Vec<f32>> = model.outputs().iter().map(|o| vec![0.0; o.n_elems]).collect();
    let mut timings: Vec<RunTiming> = Vec::with_capacity(iters);
    for i in 0..iters + 20 {
        if i % 100 == 0 {
            check_temperature()?;
        }
        let mut outputs: Vec<&mut [f32]> = buffers.iter_mut().map(Vec::as_mut_slice).collect();
        let timing: RunTiming = model.run_timed(inputs, &mut outputs)?;
        if i >= 20 {
            timings.push(timing);
        }
    }
    let column = |f: fn(&RunTiming) -> f64| -> Vec<f64> { timings.iter().map(f).collect() };
    let mut total: Vec<f64> = column(|t| t.inputs_set_us + t.run_us + t.outputs_get_us);
    let mut set: Vec<f64> = column(|t| t.inputs_set_us);
    let mut run: Vec<f64> = column(|t| t.run_us);
    let mut get: Vec<f64> = column(|t| t.outputs_get_us);
    let mut npu: Vec<f64> = column(|t| t.npu_us);
    println!("{label} ({}):", model.name());
    for (name, values) in [("  api total", &mut total), ("  inputs_set", &mut set), ("  rknn_run", &mut run), ("  outputs_get", &mut get), ("  npu (driver)", &mut npu)] {
        println!("{}", stats(name, values));
    }
    Ok(())
}

fn parallel_keynet(runtime: &RknnRuntime, file: &Path, iters: usize) -> BenchResult<()> {
    // Two contexts on cores 1 and 2, each running single crops back to back in its own thread: the KeyNet throughput ceiling.
    let crop: Vec<u8> = vec![100; KEYNET_CROP * KEYNET_CROP];
    let crop_f32: Vec<f32> = vec![100.0; KEYNET_CROP * KEYNET_CROP];
    let prior: Vec<f32> = vec![0.25; 63];
    let mut models: Vec<RknnModel> = vec![RknnModel::load(runtime, file, NpuCore::Core1)?, RknnModel::load(runtime, file, NpuCore::Core2)?];
    let quantised: bool = models[0].inputs()[0].dtype == 2 || models[0].inputs()[0].dtype == 3;
    let started: Instant = Instant::now();
    let results: Vec<Result<f64, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = models
            .iter_mut()
            .map(|model| {
                let (crop, crop_f32, prior) = (&crop, &crop_f32, &prior);
                scope.spawn(move || -> Result<f64, String> {
                    let mut buffers: Vec<Vec<f32>> = model.outputs().iter().map(|o| vec![0.0; o.n_elems]).collect();
                    let begun: Instant = Instant::now();
                    for _ in 0..iters {
                        let mut outputs: Vec<&mut [f32]> = buffers.iter_mut().map(Vec::as_mut_slice).collect();
                        let image: InputData<'_> = if quantised { InputData::U8(crop) } else { InputData::F32(crop_f32) };
                        model.run(&[image, InputData::F32(prior)], &mut outputs).map_err(|e| e.to_string())?;
                    }
                    Ok(iters as f64 / begun.elapsed().as_secs_f64())
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("thread panicked".into()))).collect()
    });
    let wall: f64 = started.elapsed().as_secs_f64();
    let mut total: f64 = 0.0;
    for (core, result) in [1, 2].iter().zip(results) {
        let rate: f64 = result?;
        total += rate;
        println!("  keynet context on core {core}: {rate:.0} crops/s ({:.3} ms/crop)", 1e3 / rate);
    }
    println!("  two cores in parallel: {total:.0} crops/s aggregate, {:.0} crops/s by wall clock", (2 * iters) as f64 / wall);
    Ok(())
}

fn read_f32(path: &Path) -> BenchResult<Vec<f32>> {
    Ok(f32_values(&std::fs::read(path)?).ok_or_else(|| format!("{}: not whole f32 values", path.display()))?)
}

/// Runs a whole evaluation set through the trait and writes the raw outputs. DetNet takes `detnet_frames_u8.bin` (640x480 net
/// frames, the deployed path) when the set has it, else expands `detnet_pooled_u8.bin`; KeyNet takes `keynet_crops_f32.bin`
/// (unrounded crops) when present, else `keynet_crops_u8.bin` / 255.
fn heldout(nets: &mut dyn HandNets, dir: &Path, out: &Path) -> BenchResult<()> {
    std::fs::create_dir_all(out)?;
    let frames_path: PathBuf = dir.join("detnet_frames_u8.bin");
    let frames: Vec<Vec<u8>> = if frames_path.exists() {
        std::fs::read(&frames_path)?.chunks_exact(DETNET_WIDTH * DETNET_HEIGHT).map(<[u8]>::to_vec).collect()
    } else {
        std::fs::read(dir.join("detnet_pooled_u8.bin"))?.chunks_exact(POOLED_WIDTH * POOLED_HEIGHT).map(expand_pooled).collect()
    };
    let mut detnet_rows: Vec<u8> = Vec::new();
    let mut detnet_us: Vec<f64> = Vec::new();
    for (i, frame) in frames.iter().enumerate() {
        if i % 200 == 0 {
            check_temperature()?;
        }
        let started: Instant = Instant::now();
        let raw = nets.detnet(&[NetFrame { pixels: frame, top: 0 }])?;
        detnet_us.push(started.elapsed().as_secs_f64() * 1e6);
        for raw in &raw {
            detnet_rows.extend(detnet_row(raw).iter().flat_map(|v| v.to_le_bytes()));
        }
    }
    std::fs::write(out.join("detnet_out_f32.bin"), &detnet_rows)?;
    println!("{}", stats(if frames_path.exists() { "eval detnet, 640x480 frames (trait)" } else { "eval detnet, expanded pools (trait)" }, &mut detnet_us));
    let crops_f32_path: PathBuf = dir.join("keynet_crops_f32.bin");
    let crop_len: usize = KEYNET_CROP * KEYNET_CROP;
    let crops: Vec<Vec<f32>> = if crops_f32_path.exists() {
        read_f32(&crops_f32_path)?.chunks_exact(crop_len).map(<[f32]>::to_vec).collect()
    } else {
        std::fs::read(dir.join("keynet_crops_u8.bin"))?.chunks_exact(crop_len).map(|crop| crop.iter().map(|&v| f32::from(v) / 255.0).collect()).collect()
    };
    let priors: Vec<f32> = read_f32(&dir.join("keynet_keypoints_f32.bin"))?;
    let mut keynet_rows: Vec<u8> = Vec::new();
    let mut keynet_us: Vec<f64> = Vec::new();
    for (i, (crop, prior)) in crops.iter().zip(priors.chunks_exact(63)).enumerate() {
        if i % 200 == 0 {
            check_temperature()?;
        }
        let prior: [f32; 63] = std::array::from_fn(|k| prior[k]);
        let started: Instant = Instant::now();
        let raw: Vec<KeyNetRaw> = nets.keynet(&[crop], &[prior])?;
        keynet_us.push(started.elapsed().as_secs_f64() * 1e6);
        for raw in &raw {
            keynet_rows.extend(keynet_row(raw).iter().flat_map(|v| v.to_le_bytes()));
        }
    }
    std::fs::write(out.join("keynet_out_f32.bin"), &keynet_rows)?;
    println!("{}", stats(if crops_f32_path.exists() { "eval keynet 1 f32 crop (trait)" } else { "eval keynet 1 u8 crop (trait)" }, &mut keynet_us));
    println!("eval outputs written to {}", out.display());
    Ok(())
}

fn rknn_backend(args: &Args, iters: usize) -> BenchResult<Box<dyn HandNets>> {
    let lib: &str = args.get("lib").unwrap_or(robocap_live::nets::rknn::DEFAULT_LIBRARY);
    let detnet: PathBuf = args.models.join(args.get("detnet").unwrap_or(robocap_live::nets::rknn::DEFAULT_DETNET));
    let keynet: PathBuf = args.models.join(args.get("keynet").unwrap_or(robocap_live::nets::rknn::DEFAULT_KEYNET));
    let contexts: usize = args.get("contexts").map_or(Ok(2), str::parse)?;
    let detnet_contexts: usize = args.get("detnet-contexts").map_or(Ok(3), str::parse)?;
    let rknn: RknnNets = RknnNets::with_files(lib, &detnet, &keynet, detnet_contexts, contexts)?;
    println!("backend: {}", rknn.describe());
    // Phase timings feed each model the way RknnNets does: u8 for INT8 inputs, f32 on the u8 scale for FP16 inputs.
    let pooled: Vec<u8> = vec![90; POOLED_WIDTH * POOLED_HEIGHT];
    let pooled_f32: Vec<f32> = vec![90.0; POOLED_WIDTH * POOLED_HEIGHT];
    let runtime: RknnRuntime = RknnRuntime::open(lib)?;
    let mut detnet_model = RknnModel::load(&runtime, &detnet, NpuCore::Core0)?;
    let detnet_image: InputData<'_> =
        if detnet_model.inputs()[0].dtype == 2 || detnet_model.inputs()[0].dtype == 3 { InputData::U8(&pooled) } else { InputData::F32(&pooled_f32) };
    rknn_phases(&mut detnet_model, &[detnet_image], iters, "detnet core 0 phases")?;
    if detnet_model.inputs()[0].dtype == 1 {
        let mut pooled_f16: Vec<u8> = vec![0; 2 * POOLED_WIDTH * POOLED_HEIGHT];
        f16_bytes_from_f32(&pooled_f32, 1.0 / 255.0, &mut pooled_f16)?;
        rknn_phases(&mut detnet_model, &[InputData::Native(&pooled_f16)], iters, "detnet core 0 phases, fp16 pass-through (deployed)")?;
    }
    let crop: Vec<u8> = vec![100; KEYNET_CROP * KEYNET_CROP];
    let crop_f32: Vec<f32> = vec![100.0; KEYNET_CROP * KEYNET_CROP];
    let prior: Vec<f32> = vec![0.25; 63];
    {
        let mut model = RknnModel::load(&runtime, &keynet, NpuCore::Core1)?;
        let quantised: bool = model.inputs()[0].dtype == 2 || model.inputs()[0].dtype == 3;
        let image: InputData<'_> = if quantised { InputData::U8(&crop) } else { InputData::F32(&crop_f32) };
        rknn_phases(&mut model, &[image, InputData::F32(&prior)], iters, "keynet core 1 phases")?;
        if !quantised {
            let mut crop_f16: Vec<u8> = vec![0; 2 * KEYNET_CROP * KEYNET_CROP];
            f16_bytes_from_f32(&crop_f32, 1.0 / 255.0, &mut crop_f16)?;
            rknn_phases(&mut model, &[InputData::Native(&crop_f16), InputData::F32(&prior)], iters, "keynet core 1 phases, fp16 pass-through (deployed)")?;
        }
    }
    if let Some(file) = args.get("keynet-b4") {
        let mut model: RknnModel = RknnModel::load(&runtime, args.models.join(file), NpuCore::Core1)?;
        let crops: Vec<u8> = vec![100; 4 * KEYNET_CROP * KEYNET_CROP];
        let crops_f32: Vec<f32> = vec![100.0; 4 * KEYNET_CROP * KEYNET_CROP];
        let priors: Vec<f32> = vec![0.25; 4 * 63];
        let image: InputData<'_> =
            if model.inputs()[0].dtype == 2 || model.inputs()[0].dtype == 3 { InputData::U8(&crops) } else { InputData::F32(&crops_f32) };
        rknn_phases(&mut model, &[image, InputData::F32(&priors)], iters, "keynet b4 core 1 phases (4 crops per call)")?;
    }
    println!("keynet throughput, contexts on cores 1 + 2:");
    parallel_keynet(&runtime, &keynet, iters * 3)?;
    let frame: Vec<u8> = vec![90; DETNET_WIDTH * DETNET_HEIGHT];
    let mut pooled_out: Vec<u8> = vec![0; POOLED_WIDTH * POOLED_HEIGHT];
    let mut pool = time_calls(iters, || pool4_u8(&frame, DETNET_WIDTH, DETNET_HEIGHT, &mut pooled_out))?;
    println!("{}", stats("cpu pool4_u8 640x480 -> 120x160", &mut pool));
    let mut pooled_mean: Vec<f32> = vec![0.0; POOLED_WIDTH * POOLED_HEIGHT];
    let mut pool_f32 = time_calls(iters, || pool4_mean_f32(&frame, DETNET_WIDTH, DETNET_HEIGHT, &mut pooled_mean))?;
    println!("{}", stats("cpu pool4_mean_f32 640x480 -> 120x160", &mut pool_f32));
    Ok(Box::new(rknn))
}

#[cfg(feature = "ort")]
fn ort_backend(args: &Args) -> BenchResult<Box<dyn HandNets>> {
    use robocap_live::nets::ort::{OrtDevice, OrtNets, OrtConfig};
    let device: OrtDevice = match args.get("device").unwrap_or("auto") {
        "cuda" => OrtDevice::Cuda(0),
        "cpu" => OrtDevice::Cpu,
        _ => OrtDevice::Auto,
    };
    let options: OrtConfig = OrtConfig { device, dylib: args.get("dylib").map(PathBuf::from), intra_threads: 0 };
    let ort: OrtNets = OrtNets::new(&args.models, &options)?;
    println!("backend: {}", ort.describe());
    Ok(Box::new(ort))
}

#[cfg(not(feature = "ort"))]
fn ort_backend(_args: &Args) -> BenchResult<Box<dyn HandNets>> {
    Err("this build has no ONNX Runtime backend: build with --features ort".into())
}

fn main() -> BenchResult<()> {
    let args: Args = Args::parse()?;
    let iters: usize = args.get("iters").map_or(Ok(300), str::parse)?;
    let golden_dir: PathBuf = args.get("golden").map_or_else(default_dir, PathBuf::from);
    println!("SoC temperature at start: {:?} C", soc_temperature_c());
    let mut nets: Box<dyn HandNets> = match args.backend.as_str() {
        "rknn" => rknn_backend(&args, iters)?,
        "ort" => ort_backend(&args)?,
        other => return Err(format!("unknown backend {other} (rknn or ort)").into()),
    };
    golden_check(nets.as_mut(), &golden_dir)?;
    latency(nets.as_mut(), iters, &golden_dir)?;
    if let (Some(dir), Some(out)) = (args.get("heldout"), args.get("out")) {
        heldout(nets.as_mut(), Path::new(dir), Path::new(out))?;
    }
    println!("SoC temperature at end: {:?} C", soc_temperature_c());
    Ok(())
}
