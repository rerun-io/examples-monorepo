//! Rust PLY parser for standard 3DGS Gaussian splat files.
//!
//! Mirrors the Python `Gaussians3D.from_ply()` in `gaussians3d.py` exactly:
//! - `x/y/z` → centers
//! - `rot_0/1/2/3` → quaternions (PLY stores wxyz, we convert to xyzw)
//! - `scale_0/1/2` → exp(log_scales), clamped ≥ 1e-6
//! - `opacity` → sigmoid(raw_opacity), clamped [0, 1]
//! - `f_dc_0/1/2` → SH DC → RGB via `SH_C0 * dc + 0.5`
//! - `f_rest_*` → higher-order SH (channel-major in PLY → interleaved)
//! - Fallback: `red/green/blue` or `r/g/b` vertex colors

use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

use glam::{Quat, Vec3};
use ply_rs::parser::Parser;
use ply_rs::ply::{DefaultElement, Property};

use crate::gsplat_core::constants::SH_C0;
use crate::gsplat_core::projection::normalize_quat_or_identity;
use crate::gsplat_core::sh::sh_degree_from_coeffs;
use crate::gsplat_core::types::{RenderGaussianCloud, RenderShCoefficients};

fn get_f32(element: &DefaultElement, name: &str) -> Option<f32> {
    match element.get(name)? {
        Property::Float(v) => Some(*v),
        Property::Double(v) => Some(*v as f32),
        Property::UChar(v) => Some(*v as f32),
        Property::UShort(v) => Some(*v as f32),
        Property::UInt(v) => Some(*v as f32),
        Property::Int(v) => Some(*v as f32),
        Property::Short(v) => Some(*v as f32),
        Property::Char(v) => Some(*v as f32),
        _ => None,
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Load a standard 3DGS PLY file into a [`RenderGaussianCloud`].
pub fn load_ply(path: &Path) -> anyhow::Result<RenderGaussianCloud> {
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(file);

    let vertex_parser = Parser::<DefaultElement>::new();
    let header = vertex_parser.read_header(&mut reader)?;

    // Find the vertex element.
    let vertex_element = header
        .elements
        .iter()
        .find(|(name, _)| *name == "vertex")
        .ok_or_else(|| anyhow::anyhow!("PLY file has no vertex element"))?;
    let vertex_count = vertex_element.1.count;

    // Collect property names for this element.
    let property_names: std::collections::HashSet<String> =
        vertex_element.1.properties.keys().cloned().collect();

    // Parse all vertices.
    let mut vertices: Vec<DefaultElement> = Vec::with_capacity(vertex_count);
    for (name, element) in &header.elements {
        if name == "vertex" {
            vertices = vertex_parser.read_payload_for_element(&mut reader, element, &header)?;
        } else {
            // Skip non-vertex elements.
            let _skip: Vec<DefaultElement> =
                vertex_parser.read_payload_for_element(&mut reader, element, &header)?;
        }
    }

    let n = vertices.len();
    let mut means = Vec::with_capacity(n);
    let mut quats = Vec::with_capacity(n);
    let mut scales = Vec::with_capacity(n);
    let mut opacities = Vec::with_capacity(n);
    let mut colors_dc = Vec::with_capacity(n);

    let has_sh_dc = property_names.contains("f_dc_0")
        && property_names.contains("f_dc_1")
        && property_names.contains("f_dc_2");

    // Count f_rest_* fields.
    let rest_count: usize = (0..)
        .take_while(|i| property_names.contains(&format!("f_rest_{i}")))
        .count();
    if rest_count != 0 && !rest_count.is_multiple_of(3) {
        anyhow::bail!(
            "Invalid PLY SH layout: found {rest_count} contiguous f_rest_* properties, \
             but the count must be divisible by 3 for RGB channel-major coefficients"
        );
    }
    let extra_coeffs_per_channel = rest_count / 3;

    for vertex in &vertices {
        // Centers.
        let x = get_f32(vertex, "x").unwrap_or(0.0);
        let y = get_f32(vertex, "y").unwrap_or(0.0);
        let z = get_f32(vertex, "z").unwrap_or(0.0);
        means.push(Vec3::new(x, y, z));

        // Quaternions: PLY stores wxyz, we need xyzw.
        let rot_0 = get_f32(vertex, "rot_0").unwrap_or(1.0); // w
        let rot_1 = get_f32(vertex, "rot_1").unwrap_or(0.0); // x
        let rot_2 = get_f32(vertex, "rot_2").unwrap_or(0.0); // y
        let rot_3 = get_f32(vertex, "rot_3").unwrap_or(0.0); // z
        quats.push(normalize_quat_or_identity(Quat::from_xyzw(
            rot_1, rot_2, rot_3, rot_0,
        )));

        // Scales: exponentiate from log-space, clamp >= 1e-6.
        let s0 = get_f32(vertex, "scale_0").unwrap_or(0.0).exp();
        let s1 = get_f32(vertex, "scale_1").unwrap_or(0.0).exp();
        let s2 = get_f32(vertex, "scale_2").unwrap_or(0.0).exp();
        scales.push(Vec3::new(s0.max(1e-6), s1.max(1e-6), s2.max(1e-6)));

        // Opacity: sigmoid activation.
        let raw_opacity = get_f32(vertex, "opacity").unwrap_or(0.0);
        opacities.push(sigmoid(raw_opacity).clamp(0.0, 1.0));

        // Colors: prefer SH DC, fallback to vertex colors.
        if has_sh_dc {
            let dc0 = get_f32(vertex, "f_dc_0").unwrap_or(0.0);
            let dc1 = get_f32(vertex, "f_dc_1").unwrap_or(0.0);
            let dc2 = get_f32(vertex, "f_dc_2").unwrap_or(0.0);
            let r = (SH_C0 * dc0 + 0.5).max(0.0);
            let g = (SH_C0 * dc1 + 0.5).max(0.0);
            let b = (SH_C0 * dc2 + 0.5).max(0.0);
            colors_dc.push([r, g, b]);
        } else if let (Some(r), Some(g), Some(b)) = (
            get_f32(vertex, "red").or_else(|| get_f32(vertex, "r")),
            get_f32(vertex, "green").or_else(|| get_f32(vertex, "g")),
            get_f32(vertex, "blue").or_else(|| get_f32(vertex, "b")),
        ) {
            // Normalize integer colors to [0, 1].
            let (r, g, b) = if r > 1.0 || g > 1.0 || b > 1.0 {
                (r / 255.0, g / 255.0, b / 255.0)
            } else {
                (r, g, b)
            };
            colors_dc.push([r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)]);
        } else {
            colors_dc.push([1.0, 1.0, 1.0]);
        }
    }

    // Build SH coefficients if available.
    let sh_coeffs = if has_sh_dc || rest_count > 0 {
        let raw_coeffs_per_channel = extra_coeffs_per_channel + 1;
        let supported_sizes: [usize; 5] = [1, 4, 9, 16, 25];
        let coeffs_per_channel = supported_sizes
            .iter()
            .copied()
            .find(|&s| s >= raw_coeffs_per_channel)
            .unwrap_or(raw_coeffs_per_channel);

        let mut flat = vec![0.0_f32; n * coeffs_per_channel * 3];
        for (i, vertex) in vertices.iter().enumerate() {
            let base = i * coeffs_per_channel * 3;
            // DC coefficient.
            if has_sh_dc {
                flat[base] = get_f32(vertex, "f_dc_0").unwrap_or(0.0);
                flat[base + 1] = get_f32(vertex, "f_dc_1").unwrap_or(0.0);
                flat[base + 2] = get_f32(vertex, "f_dc_2").unwrap_or(0.0);
            }
            // Higher-order SH: f_rest_* is channel-major in PLY.
            for c in 0..extra_coeffs_per_channel {
                let r_field = format!("f_rest_{}", c);
                let g_field = format!("f_rest_{}", extra_coeffs_per_channel + c);
                let b_field = format!("f_rest_{}", extra_coeffs_per_channel * 2 + c);
                let offset = base + (c + 1) * 3;
                flat[offset] = get_f32(vertex, &r_field).unwrap_or(0.0);
                flat[offset + 1] = get_f32(vertex, &g_field).unwrap_or(0.0);
                flat[offset + 2] = get_f32(vertex, &b_field).unwrap_or(0.0);
            }
        }

        if sh_degree_from_coeffs(coeffs_per_channel).is_some() {
            Some(RenderShCoefficients {
                coeffs_per_channel,
                coefficients: Arc::from(flat),
            })
        } else {
            None
        }
    } else {
        None
    };

    Ok(RenderGaussianCloud::from_raw(
        means, quats, scales, opacities, colors_dc, sh_coeffs,
    ))
}
