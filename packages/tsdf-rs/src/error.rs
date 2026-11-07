use thiserror::Error;

/// Errors returned by TSDF integration and marching-cubes extraction.
///
/// Variants retain the input name and relevant dimensions so callers can
/// report actionable failures without parsing an error string.
///
/// # Example
///
/// ```
/// use tsdf_rs::{TsdfError, TsdfVolume};
///
/// let error = TsdfVolume::new(0.0, 0.12, false).err();
/// assert!(matches!(
///     error,
///     Some(TsdfError::InvalidParameter { name: "voxel_size", .. })
/// ));
/// ```
#[derive(Debug, Error, PartialEq)]
pub enum TsdfError {
    /// An input scalar is not valid for the requested operation.
    #[error("{name} {reason}; got {value}")]
    InvalidParameter {
        /// Name of the invalid input.
        name: &'static str,
        /// Required constraint.
        reason: &'static str,
        /// Supplied value.
        value: f32,
    },

    /// An input array has a different shape from the required shape.
    #[error("{name} shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// Name of the invalid input.
        name: &'static str,
        /// Required dimensions.
        expected: Vec<usize>,
        /// Supplied dimensions.
        got: Vec<usize>,
    },

    /// An input NumPy array is not stored in C-contiguous order.
    #[error("{name} must be C-contiguous")]
    NonContiguous {
        /// Name of the non-contiguous input.
        name: &'static str,
    },

    /// A camera matrix cannot be inverted.
    #[error("{name} must be finite and invertible")]
    SingularMatrix {
        /// Name of the singular matrix.
        name: &'static str,
    },

    /// Mesh extraction referenced a voxel that is not allocated.
    #[error("mesh edge references missing voxel {coordinate:?}")]
    MissingVoxel {
        /// Global integer voxel coordinate.
        coordinate: [i64; 3],
    },

    /// A marching-cubes table contains an invalid edge index.
    #[error("invalid marching-cubes edge index {edge}")]
    InvalidEdge {
        /// Invalid edge index.
        edge: usize,
    },

    /// A mesh contains more vertices than its 32-bit indices can represent.
    #[error("mesh has {vertex_count} vertices, exceeding the u32 index limit")]
    VertexIndexOverflow {
        /// Number of vertices produced before the overflow.
        vertex_count: usize,
    },

    /// Input dimensions overflow the addressable slice length.
    #[error("{name} dimensions {dimensions:?} overflow usize")]
    DimensionOverflow {
        /// Name of the input.
        name: &'static str,
        /// Dimensions whose product overflowed.
        dimensions: Vec<usize>,
    },
}
