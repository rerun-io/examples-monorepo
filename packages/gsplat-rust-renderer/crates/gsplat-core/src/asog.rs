//! `.asog` v1 (animated-sog, `docs/format.md` at the `asog-src` pin bc976cf) decode into canonical [`crate::Splats`] and
//! the `.asog` motion form ([`AsogMotion`], evaluated by `crate::trigs`).
//!
//! Pure functions over the file bytes. Reference: `asog.container` (container, streams) and `asog.decode.read`
//! (de-quantisation), ported in the same operation order; the motion de-quantisation runs in f64 and is rounded to
//! f32 once.
//!
//! # Container
//! 64-byte header: `"ASOG"` + 12 zero bytes, then u64 LE manifest offset, manifest length, payload offset, payload
//! length. The UTF-8 JSON manifest lists the streams; each is one zlib (RFC 1950) blob of `count x channels`
//! integers stored as byte planes `raw[(p * channels + c) * count + i]`, optionally zigzag- and delta-coded along
//! the splat order (decode: un-zigzag, then a running sum per channel).
//!
//! # Static attributes
//! They use SOG v2 quantisation, so they are packed into SOG v2 texel planes (upstream `layout.js`) and decoded by
//! the shared [`crate::sog::dequantize`]: means 16-bit lo/hi bytes, quats RGB + A = 252 + mode, scales RGB,
//! sh0 RGB + A = opacity, shlabels R/G, shpalette row `n` -> centroid texels `n * C .. n * C + C`.
use crate::sog::{self, Planes, Quantization, ShPlanes, Version};
use crate::trigs::ClipWindow;
use serde::Deserialize;

/// Decode failure, naming the part of the file at fault.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("ASOG header: {0}")]
    Header(&'static str),
    #[error("ASOG manifest: {0}")]
    Manifest(#[from] serde_json::Error),
    #[error("ASOG manifest: format {0:?} v{1}, expected asog v1")]
    Format(String, u32),
    #[error("ASOG stream {0:?} is missing")]
    MissingStream(&'static str),
    #[error("ASOG stream {name:?}: {problem}")]
    Stream { name: String, problem: String },
    #[error("ASOG clip index {clip} of row {row}, but the manifest has {clips} clips")]
    Clip { row: usize, clip: u32, clips: usize },
    #[error(transparent)]
    Sog(#[from] sog::Error),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Timeline {
    pub frame_rate: f64,
    pub total_frames: Option<u64>,
    pub duration_seconds: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClipGlobal {
    pub frame_start: f64,
    pub active_time_start: f64,
    pub active_time_end: f64,
    pub core_time_start: f64,
    pub core_time_end: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClipLocal {
    pub frame_start: f64,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClipRender {
    #[serde(default)]
    pub fade_in_frames: f64,
    #[serde(default)]
    pub fade_out_frames: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Clip {
    pub global: ClipGlobal,
    pub local: ClipLocal,
    #[serde(default)]
    pub render: ClipRender,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Bounds {
    pub mins: [f64; 3],
    pub maxs: [f64; 3],
}

#[derive(Clone, Debug, Deserialize)]
pub struct CodebookMeta {
    pub codebook: Vec<Option<f64>>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ShNMeta {
    pub codebook: Vec<Option<f64>>,
    /// Palette entries K.
    pub count: usize,
    /// AC coefficients per channel (3, 8 or 15).
    pub coeffs: usize,
}

impl ShNMeta {
    /// The SH degree whose AC coefficient count is `coeffs`, if any.
    pub fn bands(&self) -> Option<u32> {
        (1..=4u32).find(|&b| sog::rest_coefficients(b) == self.coeffs)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct MuMeta {
    pub scale: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct LogSigmaMeta {
    pub min: f64,
    pub max: f64,
    pub bits: u32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MotionMeta {
    /// Position step in mm.
    pub qp: f64,
    /// Anchor step in mm.
    pub qa: f64,
    /// Life length in sigmas.
    pub sm: f64,
    #[serde(rename = "midO")]
    pub mid_o: f64,
    #[serde(rename = "midG")]
    pub mid_g: f64,
    pub mref_samples: usize,
    pub r_min: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamDesc {
    pub name: String,
    pub count: usize,
    pub channels: usize,
    pub planes: usize,
    pub delta: bool,
    pub zigzag: bool,
    pub byte_offset: usize,
    pub byte_length: usize,
}

/// The JSON manifest (third-party schema: unknown fields are ignored).
#[derive(Clone, Debug, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub version: u32,
    pub count: usize,
    pub timeline: Timeline,
    pub clips: Vec<Clip>,
    pub means: Bounds,
    pub scales: CodebookMeta,
    pub sh0: CodebookMeta,
    #[serde(rename = "shN")]
    pub sh_n: Option<ShNMeta>,
    pub mu: MuMeta,
    pub logsigma: LogSigmaMeta,
    pub motion: MotionMeta,
    pub streams: Vec<StreamDesc>,
}

impl Manifest {
    /// The clip table in seconds.
    pub fn clip_windows(&self) -> Vec<ClipWindow> {
        let fps = self.timeline.frame_rate;
        self.clips
            .iter()
            .map(|c| ClipWindow {
                offset: ((c.global.frame_start - c.local.frame_start) / fps) as f32,
                active_start: c.global.active_time_start as f32,
                active_end: c.global.active_time_end as f32,
                core_start: c.global.core_time_start as f32,
                core_end: c.global.core_time_end as f32,
                fade_in: (c.render.fade_in_frames / fps) as f32,
                fade_out: (c.render.fade_out_frames / fps) as f32,
            })
            .collect()
    }
}

/// The parsed container: manifest + payload bytes (borrowed from the file).
pub struct Container<'a> {
    pub manifest: Manifest,
    pub payload: &'a [u8],
}

pub const HEADER_SIZE: usize = 64;

fn read_u64(bytes: &[u8], at: usize) -> usize {
    let mut word = [0u8; 8];
    word.copy_from_slice(&bytes[at..at + 8]);
    u64::from_le_bytes(word) as usize
}

/// Header + manifest.
pub fn parse(bytes: &[u8]) -> Result<Container<'_>, Error> {
    if bytes.len() < HEADER_SIZE || &bytes[..4] != b"ASOG" {
        return Err(Error::Header("missing ASOG magic"));
    }
    let span = |offset: usize, length: usize| {
        offset
            .checked_add(length)
            .filter(|&end| end <= bytes.len())
            .map(|end| &bytes[offset..end])
            .ok_or(Error::Header("section runs past the end of the file"))
    };
    let manifest = span(read_u64(bytes, 16), read_u64(bytes, 24))?;
    let payload = span(read_u64(bytes, 32), read_u64(bytes, 40))?;
    let manifest: Manifest = serde_json::from_slice(manifest)?;
    if manifest.format != "asog" || manifest.version != 1 {
        return Err(Error::Format(manifest.format, manifest.version));
    }
    Ok(Container { manifest, payload })
}

impl Container<'_> {
    fn desc(&self, name: &'static str) -> Option<&StreamDesc> {
        self.manifest.streams.iter().find(|s| s.name == name)
    }

    /// Stream `name` as splat-major integers (`out[i * channels + c]`), with its channel count.
    pub fn stream(&self, name: &'static str) -> Result<Option<(Vec<i64>, usize)>, Error> {
        let Some(desc) = self.desc(name) else {
            return Ok(None);
        };
        let problem = |problem: String| Error::Stream {
            name: name.into(),
            problem,
        };
        let data = desc
            .byte_offset
            .checked_add(desc.byte_length)
            .and_then(|end| self.payload.get(desc.byte_offset..end))
            .ok_or_else(|| problem("runs past the payload".into()))?;
        Ok(Some((
            decode_stream(desc, data).map_err(problem)?,
            desc.channels,
        )))
    }
}

/// Every stream [`decode_parts`] reads.
const STREAMS: [&str; 14] = [
    "means",
    "quats",
    "quatmode",
    "scales",
    "sh0",
    "opacity",
    "shlabels",
    "shpalette",
    "mu",
    "logsigma",
    "anchor",
    "omega",
    "g",
    "flags",
];

/// One stream decoded by [`Container::stream`]: absent, or splat-major integers with their channel count.
type Decoded = Result<Option<(Vec<i64>, usize)>, Error>;

/// The streams of [`STREAMS`], each taken once by name.
struct Streams<'c, 'a> {
    container: &'c Container<'a>,
    /// Decoded ahead (parallel); `None` = decode on [`Self::optional`].
    ahead: [Option<Decoded>; STREAMS.len()],
}

impl<'c, 'a> Streams<'c, 'a> {
    /// With `parallel`, decode every stream now, one scoped thread per stream (the zlib members are independent).
    /// Without it, each stream is decoded on the calling thread when it is taken, so at most the streams in use are
    /// held (the web build's memory profile). Both give the same values.
    fn decode(container: &'c Container<'a>, parallel: bool) -> Self {
        if !parallel {
            return Self {
                container,
                ahead: [const { None }; STREAMS.len()],
            };
        }
        let ahead = std::thread::scope(|scope| {
            STREAMS
                .map(|name| scope.spawn(move || container.stream(name)))
                .map(|thread| {
                    Some(
                        thread
                            .join()
                            .unwrap_or_else(|panic| std::panic::resume_unwind(panic)),
                    )
                })
        });
        Self { container, ahead }
    }

    fn optional(&mut self, name: &'static str) -> Decoded {
        let index = STREAMS
            .iter()
            .position(|&s| s == name)
            .expect("a listed stream");
        self.ahead[index]
            .take()
            .unwrap_or_else(|| self.container.stream(name))
    }

    /// Stream `name`, required, with `want` channels over `rows` rows.
    fn shaped(&mut self, name: &'static str, rows: usize, want: usize) -> Result<Vec<i64>, Error> {
        shaped(
            name,
            self.optional(name)?.ok_or(Error::MissingStream(name))?,
            rows,
            want,
        )
    }
}

/// One stream's zlib bytes -> splat-major integers (`out[i * channels + c]`).
pub fn decode_stream(desc: &StreamDesc, data: &[u8]) -> Result<Vec<i64>, String> {
    let (count, channels, planes) = (desc.count, desc.channels, desc.planes);
    if planes == 0 || planes > 8 {
        return Err(format!("{planes} byte planes"));
    }
    let expected = count * channels * planes;
    let raw = miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(data, expected)
        .map_err(|e| format!("zlib: {e:?}"))?;
    if raw.len() != expected {
        return Err(format!("inflated {} bytes, expected {expected}", raw.len()));
    }
    let mut out = vec![0i64; count * channels];
    for c in 0..channels {
        let mut sum = 0i64;
        for i in 0..count {
            let mut z = 0u64;
            for p in 0..planes {
                z |= u64::from(raw[(p * channels + c) * count + i]) << (8 * p);
            }
            let mut x = if desc.zigzag {
                (z >> 1).cast_signed() ^ -(z & 1).cast_signed()
            } else {
                z.cast_signed()
            };
            if desc.delta {
                sum = sum.wrapping_add(x);
                x = sum;
            }
            out[i * channels + c] = x;
        }
    }
    Ok(out)
}

/// A decoded `.asog`: canonical splats (static attributes, f32) and the motion (the CPU oracle of the tests).
#[cfg(any(test, feature = "testing"))]
#[derive(Clone, Debug)]
pub struct Asog {
    pub manifest: Manifest,
    pub canonical: crate::Splats,
    pub motion: crate::trigs::AsogMotion,
}

fn byte(name: &'static str, value: i64) -> Result<u8, Error> {
    u8::try_from(value).map_err(|_range| Error::Stream {
        name: name.into(),
        problem: format!("value {value} does not fit a byte"),
    })
}

/// Check a stream's shape against the splat count.
fn shaped(
    name: &'static str,
    (values, channels): (Vec<i64>, usize),
    rows: usize,
    want: usize,
) -> Result<Vec<i64>, Error> {
    if channels != want || values.len() != rows * want {
        return Err(Error::Stream {
            name: name.into(),
            problem: format!(
                "{} values in {channels} channels, expected {rows} x {want}",
                values.len()
            ),
        });
    }
    Ok(values)
}

/// i32 codes per row of the motion streams in [`AsogParts::motion_codes`].
///
/// Order: opacity, mu, logsigma, anchor (3), omega (9: O0, O1, O2), g (5), flags (static | clip << 1).
/// `asog_motion.wgsl` reads the same layout.
pub const MOTION_CODES: usize = 21;

/// The entropy-decoded file before any de-quantisation.
///
/// SOG v2 texel planes (for `sog_dequant.wgsl` or the CPU [`crate::sog::dequantize`]) and the motion stream integers (for `asog_motion.wgsl` or the CPU
/// [`AsogParts::motion`]).
#[derive(Clone, Debug)]
pub struct AsogParts {
    pub manifest: Manifest,
    pub quantization: Quantization,
    pub means_l: Vec<u8>,
    pub means_u: Vec<u8>,
    pub quats: Vec<u8>,
    pub scales: Vec<u8>,
    pub sh0: Vec<u8>,
    /// SH bands, palette entries, label plane, centroid plane.
    pub sh: Option<(u32, usize, Vec<u8>, Vec<u8>)>,
    /// [`MOTION_CODES`] per row; clip indices are checked against the manifest.
    pub motion_codes: Vec<i32>,
    pub active_threshold: f32,
}

impl AsogParts {
    /// The texel planes as [`Planes`] (borrowed).
    pub fn planes(&self) -> Planes<'_> {
        Planes {
            means_l: &self.means_l,
            means_u: &self.means_u,
            quats: &self.quats,
            scales: &self.scales,
            sh0: &self.sh0,
            sh: self
                .sh
                .as_ref()
                .map(|(bands, count, labels, centroids)| ShPlanes {
                    bands: *bands,
                    palette_count: *count,
                    labels,
                    centroids,
                }),
        }
    }

    /// The motion de-quantised on the CPU (the oracle of `asog_motion.wgsl`; reference `asog.decode.read`, motion
    /// part): f64 arithmetic in its operation order, rounded to f32 once.
    #[cfg(any(test, feature = "testing"))]
    pub fn motion(&self) -> crate::trigs::AsogMotion {
        let m = &self.manifest;
        let n = m.count;
        let mo = &m.motion;
        let ls = &m.logsigma;
        let qpv = mo.qp / 1000.0;
        let w_o = [1.0, mo.mid_o, 1.0];
        let w_g = [1.0, mo.mid_g, mo.mid_g, mo.mid_g, 1.0];
        let samples = mo.mref_samples.max(1);
        let levels = f64::from((1u32 << ls.bits.min(31)) - 1);
        let spans: [f64; 3] = std::array::from_fn(|k| m.means.maxs[k] - m.means.mins[k]);
        let mut out = crate::trigs::AsogMotion {
            opacities: Vec::with_capacity(n),
            times: Vec::with_capacity(n),
            sigmas: Vec::with_capacity(n),
            anchors: Vec::with_capacity(n),
            omega: Vec::with_capacity(n),
            g: Vec::with_capacity(n),
            is_static: Vec::with_capacity(n),
            clip: Vec::with_capacity(n),
            clips: m.clip_windows(),
            active_threshold: self.active_threshold,
        };
        for (i, row) in self
            .motion_codes
            .as_chunks::<MOTION_CODES>()
            .0
            .iter()
            .enumerate()
        {
            let code = |k: usize| f64::from(row[k]);
            let sigma = (ls.min + code(2) / levels * (ls.max - ls.min)).exp();
            let life = mo.sm * sigma;
            let a: [f64; 3] = std::array::from_fn(|k| code(3 + k) * mo.qa / 1000.0);
            // The decoded position in f64 (as the reference; the f32 copy loses ~1e-6 of omega when |x - a| ~ r_min).
            let x: [f64; 3] = std::array::from_fn(|k| {
                let mean =
                    u16::from(self.means_l[4 * i + k]) | (u16::from(self.means_u[4 * i + k]) << 8);
                let v = m.means.mins[k] + f64::from(mean) / 65535.0 * spans[k];
                v.signum() * v.abs().exp_m1()
            });
            let r = (0..3).map(|k| (x[k] - a[k]).powi(2)).sum::<f64>().sqrt();
            let step_o = qpv / r.max(mo.r_min);
            let o_life: [[f64; 3]; 3] = std::array::from_fn(|k| {
                std::array::from_fn(|c| code(6 + 3 * k + c) * step_o * w_o[k])
            });
            let mut min_norm = f64::INFINITY;
            for s in 0..samples {
                let t = if samples > 1 {
                    s as f64 / (samples - 1) as f64
                } else {
                    0.0
                };
                let u = 1.0 - t;
                let (b0, b1, b2) = (u * u, 2.0 * u * t, t * t);
                let norm = (0..3)
                    .map(|c| (b0 * o_life[0][c] + b1 * o_life[1][c] + b2 * o_life[2][c]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                min_norm = min_norm.min(norm);
            }
            let mref = min_norm.max(1e-9) * qpv;
            out.opacities.push((code(0) / 255.0) as f32);
            out.times.push((code(1) / m.mu.scale) as f32);
            out.sigmas.push(sigma as f32);
            out.anchors.push(a.map(|v| v as f32));
            out.omega.push(o_life.map(|o| o.map(|v| (v / life) as f32)));
            out.g.push(std::array::from_fn(|j| {
                (code(15 + j) * mref * w_g[j] / (life * life)) as f32
            }));
            out.is_static.push(row[20] & 1 == 1);
            out.clip.push((row[20] >> 1) as u32);
        }
        out
    }
}

/// The whole file -> [`Asog`], with the static part de-quantised on the CPU.
///
/// `active_threshold` is the temporal cut the motion is evaluated with ([`crate::trigs::ASOG_ACTIVE_THRESHOLD`] = the format's constant).
#[cfg(any(test, feature = "testing"))]
pub fn decode(bytes: &[u8], active_threshold: f32) -> Result<Asog, Error> {
    let parts = decode_parts(bytes, active_threshold)?;
    let canonical = sog::dequantize(&parts.quantization, &parts.planes())?;
    let motion = parts.motion();
    Ok(Asog {
        manifest: parts.manifest,
        canonical,
        motion,
    })
}

/// Entropy decode only (zlib streams, un-delta / un-zigzag).
///
/// Static integers packed as SOG v2 texels, motion
/// integers as [`MOTION_CODES`] i32 per row. No de-quantisation: [`decode`] (CPU) or `sog_dequant.wgsl` +
/// `asog_motion.wgsl` do that. Native builds inflate the streams in parallel (one thread each); the web build, which
/// has no threads, inflates them in order.
pub fn decode_parts(bytes: &[u8], active_threshold: f32) -> Result<AsogParts, Error> {
    decode_parts_with(bytes, active_threshold, cfg!(not(target_arch = "wasm32")))
}

/// [`decode_parts`] with every stream inflated in order on the calling thread (the reference of the parallel path).
#[cfg(any(test, feature = "testing"))]
pub fn decode_parts_serial(bytes: &[u8], active_threshold: f32) -> Result<AsogParts, Error> {
    decode_parts_with(bytes, active_threshold, false)
}

fn decode_parts_with(
    bytes: &[u8],
    active_threshold: f32,
    parallel: bool,
) -> Result<AsogParts, Error> {
    let container = parse(bytes)?;
    let mut streams = Streams::decode(&container, parallel);
    let m = &container.manifest;
    let n = m.count;
    let means = streams.shaped("means", n, 3)?;
    let quats = streams.shaped("quats", n, 3)?;
    let quatmode = streams.shaped("quatmode", n, 1)?;
    let scales = streams.shaped("scales", n, 3)?;
    let sh0 = streams.shaped("sh0", n, 3)?;
    let opacity = streams.shaped("opacity", n, 1)?;

    let mut planes = [(); 5].map(|()| vec![0u8; 4 * n]);
    let [means_l, means_u, quat_plane, scale_plane, sh0_plane] = &mut planes;
    for i in 0..n {
        for a in 0..3 {
            let code = u16::try_from(means[3 * i + a]).map_err(|_range| Error::Stream {
                name: "means".into(),
                problem: format!("value {} does not fit 16 bits", means[3 * i + a]),
            })?;
            means_l[4 * i + a] = (code & 0xff) as u8;
            means_u[4 * i + a] = (code >> 8) as u8;
            quat_plane[4 * i + a] = byte("quats", quats[3 * i + a])?;
            scale_plane[4 * i + a] = byte("scales", scales[3 * i + a])?;
            sh0_plane[4 * i + a] = byte("sh0", sh0[3 * i + a])?;
        }
        let mode = byte("quatmode", quatmode[i])?;
        if mode > 3 {
            return Err(Error::Stream {
                name: "quatmode".into(),
                problem: format!("mode {mode} at row {i}"),
            });
        }
        quat_plane[4 * i + 3] = 252 + mode;
        sh0_plane[4 * i + 3] = byte("opacity", opacity[i])?;
    }

    let sh_n = match (
        &m.sh_n,
        streams.optional("shlabels")?,
        streams.optional("shpalette")?,
    ) {
        (Some(sh_n), Some(labels), Some(palette)) => {
            let rest = sh_n.coeffs;
            let bands = sh_n.bands().ok_or_else(|| Error::Stream {
                name: "shpalette".into(),
                problem: format!("{rest} coefficients is not 3, 8, 15 or 24"),
            })?;
            let labels = shaped("shlabels", labels, n, 1)?;
            let palette = shaped("shpalette", palette, sh_n.count, 3 * rest)?;
            let mut label_plane = vec![0u8; 4 * n];
            for (i, &label) in labels.iter().enumerate() {
                let label = u16::try_from(label).map_err(|_range| Error::Stream {
                    name: "shlabels".into(),
                    problem: format!("label {label} does not fit 16 bits"),
                })?;
                label_plane[4 * i] = (label & 0xff) as u8;
                label_plane[4 * i + 1] = (label >> 8) as u8;
            }
            let mut centroids = vec![255u8; 4 * sh_n.count * rest];
            for (texel, rgb) in palette.as_chunks::<3>().0.iter().enumerate() {
                for c in 0..3 {
                    centroids[4 * texel + c] = byte("shpalette", rgb[c])?;
                }
            }
            Some((bands, sh_n.count, label_plane, centroids))
        }
        _ => None,
    };
    let quantization = Quantization {
        version: Version::V2,
        count: n,
        means_min: m.means.mins,
        means_max: m.means.maxs,
        scales: sog::codebook(&m.scales.codebook, "scales")?,
        sh0: sog::codebook(&m.sh0.codebook, "sh0")?,
        sh_rest: match &m.sh_n {
            Some(sh) => sog::codebook(&sh.codebook, "shN")?,
            None => [0.0; 256],
        },
    };
    let motion_codes = motion_codes(m, opacity, &mut streams)?;
    let [means_l, means_u, quats, scales, sh0] = planes;
    Ok(AsogParts {
        manifest: container.manifest,
        quantization,
        means_l,
        means_u,
        quats,
        scales,
        sh0,
        sh: sh_n,
        motion_codes,
        active_threshold,
    })
}

/// The motion streams as [`MOTION_CODES`] i32 per row, with the clip index of every row checked.
fn motion_codes(
    m: &Manifest,
    opacity: Vec<i64>,
    decoded: &mut Streams<'_, '_>,
) -> Result<Vec<i32>, Error> {
    let n = m.count;
    let streams: [(&'static str, Vec<i64>, usize); 7] = [
        ("opacity", opacity, 1),
        ("mu", decoded.shaped("mu", n, 1)?, 1),
        ("logsigma", decoded.shaped("logsigma", n, 1)?, 1),
        ("anchor", decoded.shaped("anchor", n, 3)?, 3),
        ("omega", decoded.shaped("omega", n, 9)?, 9),
        ("g", decoded.shaped("g", n, 5)?, 5),
        ("flags", decoded.shaped("flags", n, 1)?, 1),
    ];
    let clips = m.clips.len();
    let mut out = Vec::with_capacity(n * MOTION_CODES);
    for i in 0..n {
        for (name, values, width) in &streams {
            for &v in &values[i * width..(i + 1) * width] {
                out.push(i32::try_from(v).map_err(|_range| Error::Stream {
                    name: (*name).into(),
                    problem: format!("value {v} does not fit 32 bits"),
                })?);
            }
        }
        let flag = out[out.len() - 1];
        let clip = u32::try_from(flag >> 1).unwrap_or(u32::MAX);
        if flag < 0 || clip as usize >= clips {
            return Err(Error::Clip {
                row: i,
                clip,
                clips,
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_undoes_planes_zigzag_and_delta() {
        // Two channels over three splats: c0 = [5, 3, 300], c1 = [-1, 0, 2]; delta + zigzag, 2 planes.
        let values = [[5i64, -1], [3, 0], [300, 2]];
        let mut coded = [[0i64; 2]; 3];
        for c in 0..2 {
            let mut prev = 0;
            for i in 0..3 {
                let d = values[i][c] - prev;
                prev = values[i][c];
                coded[i][c] = (d << 1) ^ (d >> 63);
            }
        }
        let mut raw = Vec::new();
        for p in 0..2 {
            for c in 0..2 {
                for row in &coded {
                    raw.push(((row[c] as u64 >> (8 * p)) & 0xff) as u8);
                }
            }
        }
        let desc = StreamDesc {
            name: "x".into(),
            count: 3,
            channels: 2,
            planes: 2,
            delta: true,
            zigzag: true,
            byte_offset: 0,
            byte_length: 0,
        };
        let data = miniz_oxide::deflate::compress_to_vec_zlib(&raw, 6);
        let decoded = decode_stream(&desc, &data).unwrap();
        assert_eq!(decoded, vec![5, -1, 3, 0, 300, 2]);
        assert!(decode_stream(&StreamDesc { count: 4, ..desc }, &data).is_err());
    }

    #[test]
    fn rejects_bad_headers() {
        assert!(matches!(parse(b"nope"), Err(Error::Header(_))));
        let mut bytes = vec![0u8; 64];
        bytes[..4].copy_from_slice(b"ASOG");
        bytes[16..24].copy_from_slice(&64u64.to_le_bytes());
        bytes[24..32].copy_from_slice(&100u64.to_le_bytes());
        assert!(matches!(parse(&bytes), Err(Error::Header(_))));
    }

    #[test]
    fn clip_fade_is_smoothstep_inside_the_active_range() {
        let clip = ClipWindow {
            offset: 0.0,
            active_start: 0.0,
            active_end: 4.0,
            core_start: 1.0,
            core_end: 3.0,
            fade_in: 1.0,
            fade_out: 1.0,
        };
        assert_eq!(clip.fade(-0.1), 0.0);
        assert_eq!(clip.fade(0.5), 0.5);
        assert_eq!(clip.fade(2.0), 1.0);
        assert_eq!(clip.fade(3.5), 0.5);
        assert_eq!(clip.fade(4.0), 0.0);
    }
}
