/// Triangle mesh of a TSDF zero level set.
///
/// Positions use world units. Triangle rows hold zero-based indices into
/// `vertices` and wind counter-clockwise seen from outside, the side the
/// cameras observed. Per-vertex rows of `colors` and `normals` follow
/// `vertices`.
///
/// # Example
///
/// ```
/// use tsdf::TsdfVolume;
///
/// let mesh = TsdfVolume::new(0.04, 0.12, false)?.extract_mesh()?;
/// assert!(mesh.vertices.is_empty() && mesh.normals.is_empty());
/// assert_eq!(mesh.colors, None);
/// # Ok::<(), tsdf::TsdfError>(())
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    /// World-space vertex positions.
    pub vertices: Vec<[f32; 3]>,
    /// Triangle vertex indices, counter-clockwise seen from outside.
    pub triangles: Vec<[u32; 3]>,
    /// Per-vertex RGB of the nearest voxel, for a color-fusing volume.
    pub colors: Option<Vec<[u8; 3]>>,
    /// Per-vertex unit normals: the TSDF gradient, pointing out of the
    /// surface toward the observing cameras.
    pub normals: Vec<[f32; 3]>,
}

/// Fraction of the way from sample `a` to sample `b` where the linear
/// interpolant crosses zero.
pub(crate) fn interpolation_amount(value_a: f32, value_b: f32) -> f32 {
    if value_a == 0.0 {
        0.0
    } else if value_b == 0.0 {
        1.0
    } else {
        value_a / (value_a - value_b)
    }
}
