//! GPU de-quantisation (L2) of SOG / ASOG texel planes straight into a splat [`Scene`] (`sog_dequant.wgsl`).
//!
//! Input = the same [`Quantization`] + [`Planes`] as the CPU oracle [`crate::sog::dequantize`]: RGBA texel planes,
//! 4 bytes per splat. They need not come from WebP (ASOG packs its zlib integers as SOG v2 texels). The host packs the
//! first `count` texels of each plane into one u32 storage buffer, builds one f32 table buffer (codebooks, a raw
//! logit per alpha byte and the position of every 16-bit code per axis, all from the CPU decoder's own functions in
//! f64) and runs one compute pass that writes the transforms, raw opacity and SH buffers of the scene.
use crate::Error;
use crate::gpu::{Kernel, bind, storage, uniform, upload};
use crate::scene::{Inputs, Scene};
use crate::sog::{
    Planes, Quantization, Version, check_planes, logit, opacity, position, rest_coefficients,
};

/// Workgroup size of `sog_dequant`.
const WG: u32 = 256;
/// Codebooks (3 x 256) + opacity logits (256) + positions (3 x 65536).
const TABLE_LEN: usize = 1024 + 3 * 65536;

/// The f32 lookup tables of `sog_dequant.wgsl` (opacity as logits).
#[cfg(test)]
pub(crate) fn tables(q: &Quantization) -> Vec<f32> {
    tables_with(q, false)
}

/// Linear opacity of an sh0 alpha byte without the clamp to 1: v3 `(a / 204)^2` reaches 1.5625.
pub(crate) fn linear_opacity(version: Version, byte: u8) -> f32 {
    match version {
        Version::V2 => f32::from(byte) / 255.0,
        Version::V3 => (f64::from(byte) / 204.0).powi(2) as f32,
    }
}

/// The tables with the opacity entries as raw logits (clamped to 1) or as linear opacity (`linear`).
pub(crate) fn tables_with(q: &Quantization, linear: bool) -> Vec<f32> {
    let mut out = Vec::with_capacity(TABLE_LEN);
    out.extend_from_slice(&q.scales);
    out.extend_from_slice(&q.sh0);
    out.extend_from_slice(&q.sh_rest);
    if linear {
        out.extend((0..=255u8).map(|a| linear_opacity(q.version, a)));
    } else {
        out.extend((0..=255u8).map(|a| logit(opacity(q.version, a))));
    }
    for axis in 0..3 {
        out.extend((0..=u16::MAX).map(|code| position(q, axis, code)));
    }
    out
}

/// The scene buffers one or more de-quantisation passes write.
pub(crate) struct Outputs {
    pub transforms: wgpu::Buffer,
    pub opacity: wgpu::Buffer,
    pub sh: wgpu::Buffer,
    /// SH coefficients per channel beyond DC of the scene (a file with fewer leaves the rest zero).
    pub rest: usize,
    /// Write linear opacity (unclamped) instead of logits.
    pub linear_opacity: bool,
}

impl Outputs {
    /// Zeroed buffers for `n` rows with `rest` AC coefficients per channel.
    pub fn new(device: &wgpu::Device, n: usize, rest: usize, limit: u64) -> Result<Self, Error> {
        let checked = |label: &str, bytes: u64| {
            if bytes > limit {
                return Err(Error::Capacity {
                    required: bytes,
                    limit,
                });
            }
            Ok(storage(device, label, bytes))
        };
        Ok(Self {
            transforms: checked("raw transforms", n as u64 * 40)?,
            opacity: checked("raw opacity", n as u64 * 4)?,
            sh: checked("SH RGB coefficients", (n * (rest + 1) * 12) as u64)?,
            rest,
            linear_opacity: false,
        })
    }
}

/// Upload the planes, run the de-quantisation pass (submitted on `queue`) and return the scene it fills.
pub(crate) fn upload_sog(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    kernel: &Kernel,
    q: &Quantization,
    planes: &Planes<'_>,
    limit: u64,
) -> Result<Scene, Error> {
    let rest = planes.sh.map_or(0, |s| rest_coefficients(s.bands));
    let out = Outputs::new(device, q.count, rest, limit)?;
    dequantize_into(device, queue, kernel, q, planes, limit, &out, 0)?;
    Ok(scene(device, out, u32::try_from(q.count)?))
}

/// The scene of filled buffers (no scale floor).
pub(crate) fn scene(device: &wgpu::Device, out: Outputs, n: u32) -> Scene {
    let degree = (1..=4)
        .find(|&b| rest_coefficients(b) == out.rest)
        .unwrap_or(0);
    Scene {
        inputs: Inputs::Splats {
            transforms: out.transforms,
            opacity: out.opacity,
            sh: out.sh,
            min_scale: upload(
                device,
                "3D scale floor",
                bytemuck::bytes_of(&0.0f32),
                wgpu::BufferUsages::STORAGE,
            ),
            has_min_scale: false,
            linear_opacity: out.linear_opacity,
            degree,
            motion: None,
        },
        n,
    }
}

/// One de-quantisation pass (submitted on `queue`) writing the file's rows at `first_row` of `out`.
#[expect(
    clippy::too_many_arguments,
    reason = "one pass per file of a set; the caller owns every input"
)]
pub(crate) fn dequantize_into(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    kernel: &Kernel,
    q: &Quantization,
    planes: &Planes<'_>,
    limit: u64,
    out: &Outputs,
    first_row: u32,
) -> Result<(), Error> {
    let palette_count = check_planes(q, planes)?;
    let n = q.count;
    let count = u32::try_from(n)?;
    let rest = planes.sh.map_or(0, |s| rest_coefficients(s.bands));
    if rest > out.rest || (u64::from(first_row) + n as u64) * 40 > out.transforms.size() {
        return Err(Error::Input(
            "SOG file does not fit the scene buffers (rows or SH bands)",
        ));
    }
    let (labels, centroids) = planes.sh.map_or((&[][..], &[][..]), |s| {
        (&s.labels[..4 * n], &s.centroids[..4 * palette_count * rest])
    });
    let mut words: Vec<u8> = Vec::with_capacity(5 * 4 * n + labels.len() + centroids.len());
    let mut offsets = Vec::new();
    let mut push = |bytes: &[u8]| {
        offsets.push(u32::try_from(words.len() / 4));
        words.extend_from_slice(bytes);
    };
    for plane in [
        planes.means_l,
        planes.means_u,
        planes.quats,
        planes.scales,
        planes.sh0,
    ] {
        push(&plane[..4 * n]);
    }
    push(labels);
    push(centroids);
    let offsets = offsets.into_iter().collect::<Result<Vec<u32>, _>>()?;
    let groups = count.div_ceil(WG).max(1);
    let x = groups.min(65535);
    let y = groups.div_ceil(x);
    let params = [
        count,
        if q.version == Version::V3 { 3 } else { 2 },
        rest as u32,
        u32::try_from(palette_count)?,
        offsets[0],
        offsets[1],
        offsets[2],
        offsets[3],
        offsets[4],
        offsets[5],
        offsets[6],
        x * WG,
        first_row,
        out.rest as u32,
        0,
        0,
    ];
    let planes_buffer = crate::scene::upload(device, "SOG texel planes", &words, limit)?;
    let table_buffer = upload(
        device,
        "SOG tables",
        bytemuck::cast_slice(&tables_with(q, out.linear_opacity)),
        wgpu::BufferUsages::STORAGE,
    );
    let params = uniform(device, &params);
    let group = bind(
        device,
        &kernel.layout,
        &[
            (0, params.as_entire_binding()),
            (1, planes_buffer.as_entire_binding()),
            (2, table_buffer.as_entire_binding()),
            (3, out.transforms.as_entire_binding()),
            (4, out.opacity.as_entire_binding()),
            (5, out.sh.as_entire_binding()),
        ],
    );
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("SOG de-quantisation"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sog_dequant"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(x, y, 1);
    }
    queue.submit([encoder.finish()]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sog::ShPlanes;
    use crate::test_utils::{gpu, read};

    /// Deterministic bytes (xorshift), so both paths see every byte value.
    fn bytes(len: usize, seed: u32) -> Vec<u8> {
        let mut s = seed.max(1);
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                s as u8
            })
            .collect()
    }

    fn quantization(version: Version, count: usize) -> Quantization {
        Quantization {
            version,
            count,
            means_min: [-3.0, -0.5, 0.0],
            means_max: [2.5, 4.0, 0.0],
            scales: std::array::from_fn(|i| -9.0 + i as f32 * 0.04),
            sh0: std::array::from_fn(|i| -2.0 + i as f32 / 64.0),
            sh_rest: std::array::from_fn(|i| -1.0 + i as f32 / 128.0),
        }
    }

    /// Worst |gpu - cpu| per field group: (positions, quaternions, log scales, logits, SH).
    fn compare(version: Version, bands: Option<u32>) -> [f32; 5] {
        let n = 70_001;
        let q = quantization(version, n);
        let mut quats = bytes(4 * n, 3);
        if version == Version::V2 {
            // Every mode byte, including the reserved identity codes below 252.
            for (i, t) in quats.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                t[3] = [252, 253, 254, 255, 17][i % 5];
            }
        }
        let rest = bands.map_or(0, rest_coefficients);
        let palette = 4096;
        let centroids = bytes(4 * palette * rest.max(1), 11);
        let mut labels = bytes(4 * n, 13);
        // Some labels past the palette (zero SH), some inside.
        for (i, t) in labels.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            if i % 7 != 0 {
                t[1] %= 16;
                t[2] = 0;
            }
        }
        let planes = Planes {
            means_l: &bytes(4 * n, 1),
            means_u: &bytes(4 * n, 2),
            quats: &quats,
            scales: &bytes(4 * n, 5),
            sh0: &bytes(4 * n, 7),
            sh: bands.map(|bands| ShPlanes {
                bands,
                palette_count: palette,
                labels: &labels,
                centroids: &centroids,
            }),
        };
        let expected = crate::sog::dequantize(&q, &planes).unwrap();
        let (device, queue) = gpu();
        let kernels = crate::kernels::Kernels::new(&device);
        let scene =
            upload_sog(&device, &queue, &kernels.sog_dequant, &q, &planes, u64::MAX).unwrap();
        let Inputs::Splats {
            transforms,
            opacity,
            sh,
            degree,
            ..
        } = &scene.inputs
        else {
            panic!("splat scene expected");
        };
        assert_eq!(*degree, expected.sh_degree);
        let t: Vec<f32> = read(&device, &queue, transforms, n * 10);
        let o: Vec<f32> = read(&device, &queue, opacity, n);
        let s: Vec<f32> = read(&device, &queue, sh, expected.sh_coefficients.len() * 3);
        let mut worst = [0.0f32; 5];
        for (i, row) in expected.transforms.iter().enumerate() {
            for (k, &e) in row.iter().enumerate() {
                let field = match k {
                    0..3 => 0,
                    3..7 => 1,
                    _ => 2,
                };
                worst[field] = worst[field].max((t[i * 10 + k] - e).abs());
            }
        }
        for (g, e) in o.iter().zip(&expected.raw_opacities) {
            worst[3] = worst[3].max((g - e).abs());
        }
        for (g, e) in s.iter().zip(expected.sh_coefficients.iter().flatten()) {
            worst[4] = worst[4].max((g - e).abs());
        }
        println!("sog_dequant {version:?} bands {bands:?}: worst |gpu - cpu| {worst:?}");
        worst
    }

    #[test]
    #[ignore = "integration: GPU"]
    fn gpu_dequant_matches_the_cpu_decoder() {
        for (version, bands) in [
            (Version::V2, Some(3)),
            (Version::V2, None),
            (Version::V3, Some(4)),
            (Version::V3, Some(1)),
        ] {
            let [positions, quats, scales, logits, sh] = compare(version, bands);
            // Table and codebook lookups are the CPU's own f32 values. Quaternions are f32 arithmetic (the CPU's are
            // f64): v2 `sqrt(1 - sum)` amplifies the f32 rounding of `sum` when the dropped component is small
            // (2.3e-6 measured on the 5090); the projector normalises them.
            assert_eq!(positions, 0.0, "{version:?} positions");
            assert_eq!(scales, 0.0, "{version:?} scales");
            assert_eq!(logits, 0.0, "{version:?} logits");
            assert_eq!(sh, 0.0, "{version:?} SH");
            assert!(quats <= 1e-5, "{version:?} quaternions {quats}");
        }
    }

    #[test]
    fn tables_hold_the_cpu_values() {
        let q = quantization(Version::V3, 1);
        let t = tables(&q);
        assert_eq!(t.len(), TABLE_LEN);
        assert_eq!(t[768 + 204], logit(1.0));
        assert_eq!(t[1024 + 65536 + 12345], position(&q, 1, 12345));
    }
}
