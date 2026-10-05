//! Probe: does feeding an FP16 RKNN model its image in the native layout (pass-through) skip the runtime's ~1 ms f32 input
//! conversion, and which value scale does the pass-through buffer need?
//!
//! `rknn_native_probe <model.rknn> [iters]` prints the input's native attribute, the max output difference of the
//! pass-through path against the normal f32 path for two value scales (u8 scale and u8 / 255), and the per-phase timings.

use std::path::PathBuf;

use robocap_live::nets::rknn::{DEFAULT_LIBRARY, InputData, NpuCore, RknnModel, RknnRuntime, RunTiming, TensorInfo};

type ProbeResult<T> = Result<T, Box<dyn std::error::Error>>;

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn timed(model: &mut RknnModel, inputs: &[InputData<'_>], buffers: &mut [Vec<f32>], iters: usize) -> ProbeResult<(f64, f64, f64)> {
    let mut set: Vec<f64> = Vec::new();
    let mut run: Vec<f64> = Vec::new();
    let mut total: Vec<f64> = Vec::new();
    for i in 0..iters + 10 {
        let mut outputs: Vec<&mut [f32]> = buffers.iter_mut().map(Vec::as_mut_slice).collect();
        let t: RunTiming = model.run_timed(inputs, &mut outputs)?;
        if i >= 10 {
            set.push(t.inputs_set_us);
            run.push(t.run_us);
            total.push(t.inputs_set_us + t.run_us + t.outputs_get_us);
        }
    }
    Ok((median(&mut set) / 1e3, median(&mut run) / 1e3, median(&mut total) / 1e3))
}

fn main() -> ProbeResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let path: PathBuf = PathBuf::from(args.get(1).ok_or("usage: rknn_native_probe <model.rknn> [iters]")?);
    let iters: usize = args.get(2).map_or(Ok(200), |v| v.parse())?;
    let runtime: RknnRuntime = RknnRuntime::open(DEFAULT_LIBRARY)?;
    let mut model: RknnModel = RknnModel::load(&runtime, &path, NpuCore::Core0)?;
    let info: TensorInfo = model.inputs()[0].clone();
    let native: TensorInfo = model.native_inputs()[0].clone();
    println!("input  {info:?}");
    println!("native {native:?}");
    let count: usize = info.n_elems;
    let values: Vec<f32> = (0..count).map(|i| ((i * 37) % 256) as f32).collect();
    let mut extra: Vec<Vec<f32>> = model.inputs()[1..].iter().map(|t| vec![0.25; t.n_elems]).collect();
    let mut reference: Vec<Vec<f32>> = model.outputs().iter().map(|o| vec![0.0; o.n_elems]).collect();
    let mut inputs: Vec<InputData<'_>> = vec![InputData::F32(&values)];
    inputs.extend(extra.iter_mut().map(|e| InputData::F32(e.as_slice())));
    let (set, run, total) = timed(&mut model, &inputs, &mut reference, iters)?;
    println!("f32 path:          inputs_set {set:.3} ms, run {run:.3} ms, total {total:.3} ms");
    // Native layout: NC1HWC2 [1, C1, H, W, C2] (fmt 2) or NHWC [1, H, W, C] (fmt 1); fp16 (dtype 1) elements; one channel.
    let dims: &[u32] = &native.dims;
    let (height, width, lanes): (usize, usize, usize) = match native.fmt {
        2 => (dims[2] as usize, dims[3] as usize, dims[4] as usize),
        1 => (dims[1] as usize, dims[2] as usize, dims[3] as usize),
        other => return Err(format!("native fmt {other} not handled").into()),
    };
    let stride: usize = if native.w_stride > 0 { native.w_stride as usize } else { width };
    if native.dtype != 1 {
        return Err(format!("native dtype {} is not fp16", native.dtype).into());
    }
    for (label, scale) in [("u8 scale", 1.0_f32), ("u8 / 255", 1.0 / 255.0)] {
        let mut buffer: Vec<u8> = vec![0; native.size_with_stride];
        for y in 0..height {
            for x in 0..width {
                let bits: u16 = half::f16::from_f32(values[y * width + x] * scale).to_bits();
                let at: usize = ((y * stride + x) * lanes) * 2;
                buffer[at..at + 2].copy_from_slice(&bits.to_le_bytes());
            }
        }
        let mut outputs: Vec<Vec<f32>> = model.outputs().iter().map(|o| vec![0.0; o.n_elems]).collect();
        let mut inputs: Vec<InputData<'_>> = vec![InputData::Native(&buffer)];
        inputs.extend(extra.iter().map(|e| InputData::F32(e.as_slice())));
        let (set, run, total) = timed(&mut model, &inputs, &mut outputs, iters)?;
        let diff: f32 = outputs.iter().flatten().zip(reference.iter().flatten()).fold(0.0, |m, (a, b)| m.max((a - b).abs()));
        let scale_of_outputs: f32 = reference.iter().flatten().fold(0.0, |m, v| m.max(v.abs()));
        println!("native ({label}): max |diff| {diff:.5} (outputs up to {scale_of_outputs:.3}); inputs_set {set:.3} ms, run {run:.3} ms, total {total:.3} ms");
    }
    Ok(())
}
