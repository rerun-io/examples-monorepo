//! SOG (`PlayCanvas` "Spatially Ordered Gaussians") v2 and v3 decode into [`crate::Splats`].
//!
//! Pure functions over bytes, no file or zip I/O: the writer unpacks a bundled `.sog` at log time and the viewer
//! hands in component bytes.
//!
//! Specs: <https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog/> (v3) and `.../sog-v2/`.
//! Reference decode: splat-transform `read-sog.ts` `decodeInto` (5d5be2b1), ported line for line by the Python
//! oracle `gsplat_rust_renderer.sog_decode`. v1 (no `version` key) is out of scope; v3 blind data is not decoded.
//!
//! # Layers
//! - meta: [`Meta::parse`] reads meta.json (unknown fields allowed; codebook entries may be JSON `null`; the SH band
//!   count is inferred from the centroid width when `shN.bands` is absent, as the `PlayCanvas` engine does).
//! - L1: [`decode_webp`]: lossless WebP (VP8L) -> [`Rgba8`] (exact bytes, never premultiplied).
//! - L2: [`dequantize`]: integer texel planes + [`Quantization`] -> [`crate::Splats`].
//! - [`decode_planes`] = meta + L1 for SOG ([`DecodedSog`] keeps the RGBA planes for a GPU upload); [`decode`] = L1 + L2.
//!
//! # The shared de-quantiser ([`dequantize`])
//! It takes texel planes, not WebP, so the ASOG decoder calls it on its zlib integer streams after packing them as
//! SOG v2 texels (the layout of upstream asog `layout.js`), with [`Version::V2`]. Every plane holds 4 bytes (RGBA)
//! per splat, splat `i` at bytes `4i..4i+4` (row-major texture order: texel `i` is splat `i`, the width is irrelevant):
//! - `means_l` / `means_u` RGB: low / high byte of the 16-bit position code `q` per axis (A unused);
//!   `n = min + (max - min) * q / 65535` (a zero span counts as 1, as splat-transform does),
//!   position `= sign(n) (exp|n| - 1)` (evaluated as `exp_m1`).
//! - `quats`: v2 RGB = the three smaller components `c` as `(c sqrt2 + 1) / 2 * 255`, in wxyz order without the
//!   largest, A = 252 + index (wxyz) of the largest (positive, `sqrt(1 - sum c^2)`); A < 252 decodes to identity.
//!   v3 RGB = exp map `p = (b - 128) / 127`, `q = (cos(pi/2 |p|), p sin(pi/2 |p|) / |p|)` (A unused).
//! - `scales` RGB: index into the log-scale codebook (A unused).
//! - `sh0` RGB: index into the DC codebook (the raw SH DC coefficient); A: opacity byte, v2 `a / 255`,
//!   v3 `(a / 204)^2` clamped to 1 (a known deviation: renderers without over-unity support must clamp, per spec).
//! - SH ([`ShPlanes`], optional): `labels` R | G << 8 (| B << 16 in v3) = palette entry; `centroids` texel
//!   `entry * C + k` (C = (bands + 1)^2 - 1, the texture width is 64 C) holds coefficient `k` as RGB indices into
//!   the AC codebook. Entries at or beyond `palette_count` decode to zero SH (as splat-transform does).
//!   For ASOG: shlabels -> labels R,G; shpalette row n (coefficient-major then RGB) -> texels `n*C..n*C+C`.
//!
//! # Output ([`crate::Splats`], gsplat-core layout)
//! f32 means, quaternion wxyz (not renormalised; the projector normalises), log scales (the codebook value), raw
//! logit opacity `ln(a / (1 - a))` with `a` clamped to `[1e-6, 1 - 1e-6]`, SH splat-major then coefficient-major RGB
//! (DC first), `sh_degree` = bands (0 without SH). Arithmetic is f64, rounded to f32 once.
//! Data stays in the file's frame (3DGS PLY: x right, y down, z forward): no flip here.
use serde::Deserialize;
use std::f64::consts::{FRAC_PI_2, SQRT_2};

/// Decode failure, naming the field or plane at fault.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("SOG meta.json: {0}")]
    Meta(#[from] serde_json::Error),
    #[error("SOG version {0:?} is not supported (v2 and v3 only)")]
    Version(Option<u32>),
    #[error("SOG {0} codebook: {1}")]
    Codebook(&'static str, &'static str),
    #[error("SOG WebP: {0}")]
    Webp(#[from] image_webp::DecodingError),
    #[error("SOG WebP plane is lossy or animated; SOG needs lossless still images")]
    NotLossless,
    #[error("SOG {plane} plane holds {texels} texels, fewer than the {count} splats")]
    PlaneSize {
        plane: &'static str,
        texels: usize,
        count: usize,
    },
    #[error(
        "SOG shN centroid width {width} does not fit {bands} SH bands (expected 64 x coefficients)"
    )]
    ShWidth { width: u32, bands: u32 },
}

/// The format revisions this decoder reads; they differ in rotation, opacity and SH label width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Version {
    V2,
    V3,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MeansMeta {
    /// Log-domain bounds per axis.
    pub mins: [f64; 3],
    pub maxs: [f64; 3],
    pub files: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CodebookMeta {
    /// 256 entries; a null `codebook[0]` (an old tool bug) is patched as the `PlayCanvas` engine does.
    pub codebook: Vec<Option<f64>>,
    pub files: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct FilesMeta {
    pub files: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ShNMeta {
    /// Palette entries; absent in older v2 files (then: centroid rows x 64).
    pub count: Option<usize>,
    /// 1-4; absent in older v2 files (then inferred from the centroid width).
    pub bands: Option<u32>,
    pub codebook: Vec<Option<f64>>,
    pub files: Vec<String>,
}

/// meta.json of a v2 or v3 SOG (third-party schema: unknown fields are ignored).
#[derive(Clone, Debug, Deserialize)]
pub struct Meta {
    pub version: u32,
    pub count: usize,
    pub means: MeansMeta,
    pub scales: CodebookMeta,
    pub quats: FilesMeta,
    pub sh0: CodebookMeta,
    #[serde(rename = "shN")]
    pub sh_n: Option<ShNMeta>,
    /// "antialiased" | "2dgs" | absent (ordinary 3DGS). Carried, not acted on.
    pub model: Option<String>,
}

#[derive(Deserialize)]
struct VersionOnly {
    version: Option<u32>,
}

impl Meta {
    /// Parse meta.json; v1 (no `version`) and unknown versions are rejected.
    pub fn parse(json: &str) -> Result<Self, Error> {
        match serde_json::from_str::<VersionOnly>(json)?.version {
            Some(2 | 3) => Ok(serde_json::from_str(json)?),
            other => Err(Error::Version(other)),
        }
    }

    pub fn format_version(&self) -> Version {
        if self.version == 3 {
            Version::V3
        } else {
            Version::V2
        }
    }

    /// The scalar de-quantisation parameters (codebooks patched, `sh_rest` zero without `shN`).
    pub fn quantization(&self) -> Result<Quantization, Error> {
        Ok(Quantization {
            version: self.format_version(),
            count: self.count,
            means_min: self.means.mins,
            means_max: self.means.maxs,
            scales: codebook(&self.scales.codebook, "scales")?,
            sh0: codebook(&self.sh0.codebook, "sh0")?,
            sh_rest: match &self.sh_n {
                Some(sh) => codebook(&sh.codebook, "shN")?,
                None => [0.0; 256],
            },
        })
    }

    /// Upper bound of the SH degree from meta.json alone: `shN.bands`, else the most the version allows (v3: 4,
    /// v2: 3); 0 without `shN`.
    pub fn max_sh_degree(&self) -> u32 {
        match &self.sh_n {
            Some(sh) => sh.bands.unwrap_or(if self.version == 3 { 4 } else { 3 }),
            None => 0,
        }
    }

    /// SH bands: `shN.bands` (checked against the width) or inferred from a centroid width of 192/512/960/1536.
    pub fn sh_bands(&self, centroid_width: u32) -> Result<u32, Error> {
        let inferred = (1..=4).find(|&b| 64 * rest_coefficients(b) as u32 == centroid_width);
        let bands = self.sh_n.as_ref().and_then(|sh| sh.bands);
        match (bands, inferred) {
            (Some(b), Some(i)) if b == i => Ok(b),
            (None, Some(i)) => Ok(i),
            (b, _) => Err(Error::ShWidth {
                width: centroid_width,
                bands: b.unwrap_or(0),
            }),
        }
    }
}

/// SH coefficients per channel beyond DC for `bands` (3, 8, 15, 24 for 1-4).
pub const fn rest_coefficients(bands: u32) -> usize {
    ((bands + 1) * (bands + 1) - 1) as usize
}

/// A 256-entry codebook as f32. A null entry 0 becomes `cb[1] + (cb[1] - cb[255]) / 255` (`PlayCanvas` engine
/// `_patchCodebooks`); any other null, or another length, is an error.
pub fn codebook(entries: &[Option<f64>], name: &'static str) -> Result<[f32; 256], Error> {
    if entries.len() != 256 {
        return Err(Error::Codebook(name, "needs 256 entries"));
    }
    let value = |i: usize| entries[i].ok_or(Error::Codebook(name, "null entry beyond index 0"));
    let first = match entries[0] {
        Some(v) => v,
        None => value(1)? + (value(1)? - value(255)?) / 255.0,
    };
    let mut out = [0.0f32; 256];
    out[0] = first as f32;
    for (i, slot) in out.iter_mut().enumerate().skip(1) {
        *slot = value(i)? as f32;
    }
    Ok(out)
}

/// Scalar parameters of the de-quantiser: what meta.json (SOG) or the manifest (ASOG) carries.
#[derive(Clone, Debug)]
pub struct Quantization {
    pub version: Version,
    pub count: usize,
    /// Log-domain position bounds per axis.
    pub means_min: [f64; 3],
    pub means_max: [f64; 3],
    /// Log scales.
    pub scales: [f32; 256],
    /// SH DC coefficients.
    pub sh0: [f32; 256],
    /// SH AC coefficients (one codebook for every band and channel).
    pub sh_rest: [f32; 256],
}

/// The SH palette planes (see the module docs for the texel layout).
#[derive(Clone, Copy, Debug)]
pub struct ShPlanes<'a> {
    pub bands: u32,
    pub palette_count: usize,
    pub labels: &'a [u8],
    pub centroids: &'a [u8],
}

/// RGBA texel planes, 4 bytes per splat (see the module docs).
#[derive(Clone, Copy, Debug)]
pub struct Planes<'a> {
    pub means_l: &'a [u8],
    pub means_u: &'a [u8],
    pub quats: &'a [u8],
    pub scales: &'a [u8],
    pub sh0: &'a [u8],
    pub sh: Option<ShPlanes<'a>>,
}

fn check_plane(plane: &'static str, bytes: &[u8], count: usize) -> Result<(), Error> {
    let texels = bytes.len() / 4;
    if texels < count {
        return Err(Error::PlaneSize {
            plane,
            texels,
            count,
        });
    }
    Ok(())
}

/// v2 smallest-three: wxyz slots of the three stored components for each index of the dropped largest one.
const QUAT_SLOTS: [[usize; 3]; 4] = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];

/// wxyz from a v2 smallest-three texel (RGB components, A = 252 + largest index; A < 252 = identity).
pub fn quat_smallest_three(texel: [u8; 4]) -> [f64; 4] {
    if texel[3] < 252 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let largest = (texel[3] - 252) as usize;
    let mut q = [0.0f64; 4];
    let mut sum = 0.0;
    for (slot, byte) in QUAT_SLOTS[largest].into_iter().zip(&texel[..3]) {
        let c = (*byte as f64 / 255.0 * 2.0 - 1.0) / SQRT_2;
        q[slot] = c;
        sum += c * c;
    }
    q[largest] = (1.0 - sum).max(0.0).sqrt();
    q
}

/// wxyz from a v3 exp-map texel (RGB).
pub fn quat_exp_map(texel: [u8; 4]) -> [f64; 4] {
    let p = [0, 1, 2].map(|c| (texel[c] as f64 - 128.0) / 127.0);
    let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    let t = FRAC_PI_2 * r;
    let k = if r > 1e-12 { t.sin() / r } else { FRAC_PI_2 };
    [t.cos(), p[0] * k, p[1] * k, p[2] * k]
}

/// Linear opacity of an sh0 alpha byte: v2 `a / 255`, v3 `(a / 204)^2` clamped to 1.
pub fn opacity(version: Version, byte: u8) -> f64 {
    match version {
        Version::V2 => byte as f64 / 255.0,
        Version::V3 => (byte as f64 / 204.0).powi(2).min(1.0),
    }
}

/// Raw logit opacity as gsplat-core stores it.
pub fn logit(alpha: f64) -> f32 {
    let a = alpha.clamp(1e-6, 1.0 - 1e-6);
    ((a / (1.0 - a)).ln() as f32).clamp(-100.0, 100.0)
}

#[cfg(any(test, feature = "testing"))]
fn texel(plane: &[u8], i: usize) -> [u8; 4] {
    [
        plane[4 * i],
        plane[4 * i + 1],
        plane[4 * i + 2],
        plane[4 * i + 3],
    ]
}

/// The position of 16-bit code `code` on `axis` (f64 log map, rounded to f32 once).
pub fn position(q: &Quantization, axis: usize, code: u16) -> f32 {
    let span = q.means_max[axis] - q.means_min[axis];
    let span = if span == 0.0 { 1.0 } else { span };
    let v = q.means_min[axis] + span * (code as f64 / 65535.0);
    let e = v.abs().exp_m1();
    (if v < 0.0 { -e } else { e }) as f32
}

/// Check the plane sizes against `q.count`; returns the usable SH palette entries (0 without SH).
/// Entries whose texels run past the centroid plane count as out of range.
pub fn check_planes(q: &Quantization, planes: &Planes<'_>) -> Result<usize, Error> {
    let n = q.count;
    for (name, plane) in [
        ("means_l", planes.means_l),
        ("means_u", planes.means_u),
        ("quats", planes.quats),
        ("scales", planes.scales),
        ("sh0", planes.sh0),
    ] {
        check_plane(name, plane, n)?;
    }
    Ok(match planes.sh {
        Some(s) => {
            check_plane("shN_labels", s.labels, n)?;
            s.palette_count
                .min(s.centroids.len() / 4 / rest_coefficients(s.bands))
        }
        None => 0,
    })
}

/// L2 on the CPU: texel planes -> [`crate::Splats`], for SOG (decoded WebP) and ASOG (packed zlib integers, [`Version::V2`]).
/// The oracle of `sog_dequant.wgsl` in the tests.
#[cfg(any(test, feature = "testing"))]
pub fn dequantize(q: &Quantization, planes: &Planes<'_>) -> Result<crate::Splats, Error> {
    let n = q.count;
    let palette_count = check_planes(q, planes)?;
    let sh = planes.sh;
    let rest = sh.map_or(0, |s| rest_coefficients(s.bands));
    let mut splats = crate::Splats {
        transforms: Vec::with_capacity(n),
        raw_opacities: Vec::with_capacity(n),
        sh_coefficients: Vec::with_capacity(n * (rest + 1)),
        sh_degree: sh.map_or(0, |s| s.bands),
        min_scale: None,
    };
    for i in 0..n {
        let (lo, hi) = (texel(planes.means_l, i), texel(planes.means_u, i));
        let position = [0, 1, 2].map(|a| position(q, a, u16::from_le_bytes([lo[a], hi[a]])));
        let rotation = match q.version {
            Version::V2 => quat_smallest_three(texel(planes.quats, i)),
            Version::V3 => quat_exp_map(texel(planes.quats, i)),
        };
        let scale = texel(planes.scales, i);
        let color = texel(planes.sh0, i);
        splats.transforms.push([
            position[0],
            position[1],
            position[2],
            rotation[0] as f32,
            rotation[1] as f32,
            rotation[2] as f32,
            rotation[3] as f32,
            q.scales[scale[0] as usize],
            q.scales[scale[1] as usize],
            q.scales[scale[2] as usize],
        ]);
        splats
            .raw_opacities
            .push(logit(opacity(q.version, color[3])));
        splats
            .sh_coefficients
            .push([0, 1, 2].map(|c| q.sh0[color[c] as usize]));
        if let Some(s) = sh {
            let l = texel(s.labels, i);
            let mut label = l[0] as usize | ((l[1] as usize) << 8);
            if q.version == Version::V3 {
                label |= (l[2] as usize) << 16;
            }
            for k in 0..rest {
                splats.sh_coefficients.push(if label < palette_count {
                    let t = texel(s.centroids, label * rest + k);
                    [0, 1, 2].map(|c| q.sh_rest[t[c] as usize])
                } else {
                    [0.0; 3]
                });
            }
        }
    }
    Ok(splats)
}

/// A decoded RGBA8 image, row-major.
#[derive(Clone, Debug)]
pub struct Rgba8 {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// L1: lossless WebP -> RGBA8 exactly as stored (RGB images get A = 255). Lossy or animated WebP is an error.
pub fn decode_webp(bytes: &[u8]) -> Result<Rgba8, Error> {
    let mut decoder = image_webp::WebPDecoder::new(std::io::Cursor::new(bytes))?;
    if decoder.is_lossy() || decoder.is_animated() {
        return Err(Error::NotLossless);
    }
    let (width, height) = decoder.dimensions();
    let mut buffer = vec![0u8; decoder.output_buffer_size().ok_or(Error::NotLossless)?];
    decoder.read_image(&mut buffer)?;
    let data = if decoder.has_alpha() {
        buffer
    } else {
        buffer
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|&[r, g, b]| [r, g, b, 255])
            .collect()
    };
    Ok(Rgba8 {
        width,
        height,
        data,
    })
}

/// The stored WebP bytes of one SOG, by role (the `SogSplats3D` component names). Without both SH planes the
/// splats decode at degree 0 even when meta.json has `shN` (a viewer may skip them).
#[derive(Clone, Copy, Debug)]
pub struct SogWebp<'a> {
    pub means_l: &'a [u8],
    pub means_u: &'a [u8],
    pub quats: &'a [u8],
    pub scales: &'a [u8],
    pub sh0: &'a [u8],
    pub sh_centroids: Option<&'a [u8]>,
    pub sh_labels: Option<&'a [u8]>,
}

/// The SH part of a [`DecodedSog`].
#[derive(Clone, Debug)]
pub struct DecodedSh {
    pub bands: u32,
    pub palette_count: usize,
    pub labels: Rgba8,
    pub centroids: Rgba8,
}

/// Meta + L1 output: the de-quantisation parameters and the RGBA planes (the input of the CPU [`dequantize`] and of
/// a GPU de-quantisation pass).
#[derive(Clone, Debug)]
pub struct DecodedSog {
    pub quantization: Quantization,
    pub means_l: Rgba8,
    pub means_u: Rgba8,
    pub quats: Rgba8,
    pub scales: Rgba8,
    pub sh0: Rgba8,
    pub sh: Option<DecodedSh>,
}

impl DecodedSog {
    pub fn planes(&self) -> Planes<'_> {
        Planes {
            means_l: &self.means_l.data,
            means_u: &self.means_u.data,
            quats: &self.quats.data,
            scales: &self.scales.data,
            sh0: &self.sh0.data,
            sh: self.sh.as_ref().map(|s| ShPlanes {
                bands: s.bands,
                palette_count: s.palette_count,
                labels: &s.labels.data,
                centroids: &s.centroids.data,
            }),
        }
    }
}

/// Meta + L1: parse nothing more than needed, decode each WebP plane once.
pub fn decode_planes(meta: &Meta, webp: &SogWebp<'_>) -> Result<DecodedSog, Error> {
    let sh = match (&meta.sh_n, webp.sh_centroids, webp.sh_labels) {
        (Some(sh_n), Some(centroids), Some(labels)) => {
            let centroids = decode_webp(centroids)?;
            let bands = meta.sh_bands(centroids.width)?;
            let palette_count = sh_n.count.unwrap_or(centroids.height as usize * 64);
            Some(DecodedSh {
                bands,
                palette_count,
                labels: decode_webp(labels)?,
                centroids,
            })
        }
        _ => None,
    };
    Ok(DecodedSog {
        quantization: meta.quantization()?,
        means_l: decode_webp(webp.means_l)?,
        means_u: decode_webp(webp.means_u)?,
        quats: decode_webp(webp.quats)?,
        scales: decode_webp(webp.scales)?,
        sh0: decode_webp(webp.sh0)?,
        sh,
    })
}

/// meta.json + stored WebP planes -> [`crate::Splats`] (CPU L1 + L2; the oracle tests' entry point).
#[cfg(any(test, feature = "testing"))]
pub fn decode(meta_json: &str, webp: &SogWebp<'_>) -> Result<crate::Splats, Error> {
    let meta = Meta::parse(meta_json)?;
    let decoded = decode_planes(&meta, webp)?;
    dequantize(&decoded.quantization, &decoded.planes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quantization(version: Version, count: usize) -> Quantization {
        Quantization {
            version,
            count,
            means_min: [-1.0, -2.0, 0.0],
            means_max: [1.0, 2.0, 0.0],
            scales: std::array::from_fn(|i| -10.0 + i as f32 * 0.05),
            sh0: std::array::from_fn(|i| -2.0 + i as f32 / 64.0),
            sh_rest: std::array::from_fn(|i| -1.0 + i as f32 / 128.0),
        }
    }

    fn near(a: f32, b: f64) -> bool {
        (a as f64 - b).abs() <= 1e-6 * b.abs().max(1.0)
    }

    #[test]
    fn positions_use_the_log_map_and_a_zero_span() {
        let q = quantization(Version::V2, 3);
        // Codes 0, 65535 and 32768 (lo 0x00, hi 0x80).
        let means_l = [0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 255];
        let means_u = [0, 0, 0, 255, 255, 255, 255, 255, 128, 128, 128, 255];
        let rgba = [128u8, 128, 128, 255].repeat(3);
        let planes = Planes {
            means_l: &means_l,
            means_u: &means_u,
            quats: &rgba,
            scales: &rgba,
            sh0: &rgba,
            sh: None,
        };
        let s = dequantize(&q, &planes).unwrap();
        let t = &s.transforms;
        assert!(near(t[0][0], -(1f64.exp_m1())));
        assert!(near(t[0][1], -(2f64.exp_m1())));
        assert!(near(t[1][0], 1f64.exp_m1()));
        assert!(near(t[1][1], 2f64.exp_m1()));
        let mid: f64 = -1.0 + 2.0 * 32768.0 / 65535.0;
        assert!(near(t[2][0], mid.exp_m1()));
        // Zero span on z: splat-transform's `|| 1` gives min + q / 65535.
        assert!(near(t[1][2], 1f64.exp_m1()));
        assert!(near(t[0][2], 0.0));
        assert_eq!(s.sh_degree, 0);
        assert_eq!(s.sh_coefficients.len(), 3);
    }

    #[test]
    fn smallest_three_places_the_largest_component() {
        // Stored components 0 -> -1/sqrt2... use 128 (= ~0) and check each largest slot.
        for largest in 0..4u8 {
            let q = quat_smallest_three([128, 128, 128, 252 + largest]);
            let small = (128.0 / 255.0 * 2.0 - 1.0) / SQRT_2;
            for (slot, &v) in q.iter().enumerate() {
                if slot == largest as usize {
                    assert!((v - (1.0 - 3.0 * small * small).sqrt()).abs() < 1e-15);
                } else {
                    assert!((v - small).abs() < 1e-15);
                }
            }
        }
        assert_eq!(quat_smallest_three([7, 9, 11, 251]), [1.0, 0.0, 0.0, 0.0]);
        // An encoded rotation: 90 degrees about y, wxyz (0.7071, 0, 0.7071, 0): largest = w (index 0, positive),
        // stored x, y, z.
        let enc = |c: f64| ((c * SQRT_2 + 1.0) * 127.5).round() as u8;
        let h = 0.5f64.sqrt();
        let q = quat_smallest_three([enc(0.0), enc(h), enc(0.0), 252]);
        assert!((q[0] - h).abs() < 5e-3 && (q[2] - h).abs() < 5e-3);
    }

    #[test]
    fn exp_map_decodes_identity_and_quarter_turns() {
        assert_eq!(quat_exp_map([128, 128, 128, 0]), [1.0, 0.0, 0.0, 0.0]);
        // p = (0, 0, 1): t = pi/2 -> 180 degrees about z, w = cos(pi/2) = 0.
        let q = quat_exp_map([128, 128, 255, 0]);
        assert!(q[0].abs() < 1e-15 && (q[3] - 1.0).abs() < 1e-15);
        // p = (0.5, 0, 0) (byte 128 + 63.5 is not an integer; 191 -> 63/127).
        let p = 63.0 / 127.0;
        let q = quat_exp_map([191, 128, 128, 0]);
        let t = FRAC_PI_2 * p;
        assert!((q[0] - t.cos()).abs() < 1e-15 && (q[1] - t.sin()).abs() < 1e-15);
        let norm: f64 = q.iter().map(|v| v * v).sum();
        assert!((norm - 1.0).abs() < 1e-12);
    }

    #[test]
    fn opacity_v2_linear_v3_squared_and_clamped() {
        assert_eq!(opacity(Version::V2, 255), 1.0);
        assert_eq!(opacity(Version::V2, 51), 0.2);
        assert!((opacity(Version::V3, 102) - 0.25).abs() < 1e-15);
        assert_eq!(opacity(Version::V3, 204), 1.0);
        // 255 / 204 squared = 1.5625: clamped.
        assert_eq!(opacity(Version::V3, 255), 1.0);
        assert!((logit(0.0) - (1e-6f64 / (1.0 - 1e-6)).ln() as f32).abs() < 1e-5);
        assert!(logit(1.0) > 13.0 && logit(1.0) < 14.0);
        assert!(logit(0.5).abs() < 1e-7);
    }

    #[test]
    fn codebook_patches_a_null_first_entry() {
        let mut entries: Vec<Option<f64>> = (0..256).map(|i| Some(i as f64)).collect();
        entries[0] = None;
        let cb = codebook(&entries, "scales").unwrap();
        assert_eq!(cb[0], (1.0 + (1.0 - 255.0) / 255.0) as f32);
        assert_eq!(cb[255], 255.0);
        entries[7] = None;
        assert!(codebook(&entries, "scales").is_err());
        assert!(codebook(&entries[..10], "scales").is_err());
    }

    fn meta_json(version: u32, sh: &str) -> String {
        let cb = (0..256)
            .map(|i| (f64::from(i) * 0.01).to_string())
            .collect::<Vec<_>>()
            .join(",");
        format!(
            r#"{{"version":{version},"count":2,"extra":{{"a":1}},
            "means":{{"mins":[0,0,0],"maxs":[1,1,1],"files":["means_l.webp","means_u.webp"]}},
            "scales":{{"codebook":[null,{cb_tail}],"files":["scales.webp"]}},
            "quats":{{"files":["quats.webp"]}},
            "sh0":{{"codebook":[{cb}],"files":["sh0.webp"]}}{sh}}}"#,
            cb_tail = cb.split_once(',').unwrap().1
        )
    }

    #[test]
    fn meta_accepts_v2_v3_unknown_fields_and_infers_bands() {
        let cb = (0..256)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let meta = Meta::parse(&meta_json(
            2,
            &format!(r#","shN":{{"codebook":[{cb}],"files":["c.webp","l.webp"]}}"#),
        ))
        .unwrap();
        assert_eq!(meta.format_version(), Version::V2);
        assert_eq!(meta.sh_bands(960).unwrap(), 3);
        assert_eq!(meta.sh_bands(192).unwrap(), 1);
        assert_eq!(meta.sh_bands(1536).unwrap(), 4);
        assert!(meta.sh_bands(100).is_err());
        assert!(meta.quantization().unwrap().scales[0] != 0.0);
        let meta = Meta::parse(&meta_json(
            3,
            &format!(
                r#","shN":{{"count":5,"bands":2,"codebook":[{cb}],"files":["c.webp","l.webp"]}}"#
            ),
        ))
        .unwrap();
        assert_eq!(meta.format_version(), Version::V3);
        assert_eq!(meta.sh_bands(512).unwrap(), 2);
        assert!(meta.sh_bands(960).is_err());
        assert!(matches!(
            Meta::parse(r#"{"count":1,"means":{}}"#),
            Err(Error::Version(None))
        ));
        assert!(matches!(
            Meta::parse(&meta_json(4, "")),
            Err(Error::Version(Some(4)))
        ));
    }

    #[test]
    fn sh_palette_is_coefficient_major_with_label_widths() {
        let bands = 1;
        let rest = rest_coefficients(bands);
        // Palette entry e, coefficient k: RGB indices (10e + k, 10e + k + 100, 10e + k + 200) - small entries only.
        let entries = 3;
        let centroids: Vec<u8> = (0..entries * rest)
            .flat_map(|t| {
                let (e, k) = (t / rest, t % rest);
                let base = (10 * e + k) as u8;
                [base, base + 100, base + 200, 255]
            })
            .collect();
        // Splat 0 -> entry 2; splat 1 -> label 0x010001 (v2 reads 1, v3 reads 65537 = out of range -> zeros);
        // splat 2 -> entry 3 = palette_count -> zeros.
        let labels = [2, 0, 0, 255, 1, 0, 1, 255, 3, 0, 0, 255];
        let rgba = [128u8, 128, 128, 255].repeat(3);
        for version in [Version::V2, Version::V3] {
            let q = quantization(version, 3);
            let planes = Planes {
                means_l: &rgba,
                means_u: &rgba,
                quats: &rgba,
                scales: &rgba,
                sh0: &rgba,
                sh: Some(ShPlanes {
                    bands,
                    palette_count: entries,
                    labels: &labels,
                    centroids: &centroids,
                }),
            };
            let s = dequantize(&q, &planes).unwrap();
            assert_eq!(s.sh_degree, 1);
            assert_eq!(s.sh_coefficients.len(), 3 * 4);
            let coefficient = |splat: usize, k: usize| s.sh_coefficients[splat * 4 + 1 + k];
            for k in 0..rest {
                let b = 20 + k;
                assert_eq!(
                    coefficient(0, k),
                    [q.sh_rest[b], q.sh_rest[b + 100], q.sh_rest[b + 200]]
                );
                assert_eq!(coefficient(2, k), [0.0; 3]);
                let expected = if version == Version::V2 {
                    [q.sh_rest[10 + k], q.sh_rest[110 + k], q.sh_rest[210 + k]]
                } else {
                    [0.0; 3]
                };
                assert_eq!(coefficient(1, k), expected);
            }
            assert_eq!(s.sh_coefficients[0], [q.sh0[128]; 3]);
        }
    }

    #[test]
    fn short_planes_are_rejected() {
        let q = quantization(Version::V2, 3);
        let rgba = [0u8; 12];
        let short = [0u8; 8];
        let planes = Planes {
            means_l: &rgba,
            means_u: &rgba,
            quats: &short,
            scales: &rgba,
            sh0: &rgba,
            sh: None,
        };
        assert!(matches!(
            dequantize(&q, &planes),
            Err(Error::PlaneSize {
                plane: "quats",
                texels: 2,
                count: 3
            })
        ));
    }

    fn encode_webp(data: &[u8], width: u32, height: u32, color: image_webp::ColorType) -> Vec<u8> {
        let mut out = Vec::new();
        image_webp::WebPEncoder::new(&mut out)
            .encode(data, width, height, color)
            .unwrap();
        out
    }

    #[test]
    fn webp_decodes_exact_rgba_and_rgb() {
        // RGB under A = 0 must survive (never premultiplied).
        let rgba: Vec<u8> = (0..6 * 4)
            .map(|i| (i * 37 % 256) as u8)
            .map(|v| v | 1)
            .collect();
        let mut rgba = rgba;
        rgba[3] = 0;
        let image = decode_webp(&encode_webp(&rgba, 3, 2, image_webp::ColorType::Rgba8)).unwrap();
        assert_eq!((image.width, image.height), (3, 2));
        assert_eq!(image.data, rgba);
        let rgb: Vec<u8> = (0..6 * 3).map(|i| (i * 53 % 256) as u8).collect();
        let image = decode_webp(&encode_webp(&rgb, 2, 3, image_webp::ColorType::Rgb8)).unwrap();
        let expected: Vec<u8> = rgb
            .chunks(3)
            .flat_map(|c| [c[0], c[1], c[2], 255])
            .collect();
        assert_eq!(image.data, expected);
        assert!(decode_webp(b"RIFF\0\0\0\0WEBPjunk").is_err());
    }

    #[test]
    fn decode_reads_planes_by_role() {
        let rgba = |texels: &[[u8; 4]]| {
            encode_webp(
                &texels.concat(),
                texels.len() as u32,
                1,
                image_webp::ColorType::Rgba8,
            )
        };
        let means = rgba(&[[0, 0, 0, 255], [255, 255, 255, 255]]);
        let quats = rgba(&[[128, 128, 128, 252], [128, 128, 128, 255]]);
        let scales = rgba(&[[1, 2, 3, 255], [4, 5, 6, 255]]);
        let sh0 = rgba(&[[0, 1, 2, 255], [3, 4, 5, 102]]);
        let webp = SogWebp {
            means_l: &means,
            means_u: &means,
            quats: &quats,
            scales: &scales,
            sh0: &sh0,
            sh_centroids: None,
            sh_labels: None,
        };
        let s = decode(&meta_json(3, ""), &webp).unwrap();
        assert_eq!(s.transforms.len(), 2);
        assert!(near(s.transforms[1][0], 1f64.exp_m1()));
        // v3 quats: (128,128,128) is identity whatever A says.
        assert_eq!(s.transforms[0][3..7], [1.0, 0.0, 0.0, 0.0]);
        let meta = Meta::parse(&meta_json(3, "")).unwrap();
        let q = meta.quantization().unwrap();
        assert_eq!(s.transforms[1][7..10], q.scales[4..7]);
        assert!((s.raw_opacities[1] - logit(0.25)).abs() < 1e-7);
        // Codebook entry 0 of scales was null: patched from entries 1 and 255.
        let cb = &meta.scales.codebook;
        let (one, last) = (cb[1].unwrap(), cb[255].unwrap());
        assert_eq!(q.scales[0], (one + (one - last) / 255.0) as f32);
        assert!(matches!(
            decode(
                &meta_json(3, ""),
                &SogWebp {
                    quats: &means[..20],
                    ..webp
                }
            ),
            Err(Error::Webp(_))
        ));
    }
}
