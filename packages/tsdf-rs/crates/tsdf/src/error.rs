/// Errors returned by TSDF integration and mesh extraction.
///
/// Variants keep the input name and the numbers involved, so callers can
/// report a failure without parsing the message.
///
/// # Example
///
/// ```
/// use tsdf::{TsdfError, TsdfVolume};
///
/// let error = TsdfVolume::new(0.0, 0.12, false).err();
/// assert!(matches!(
///     error,
///     Some(TsdfError::InvalidParameter { name: "voxel_size", .. })
/// ));
/// ```
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum TsdfError {
    /// An input scalar is outside its valid range.
    #[error("{name} {reason}; got {value}")]
    InvalidParameter {
        /// Name of the invalid input.
        name: &'static str,
        /// Required constraint.
        reason: &'static str,
        /// Supplied value.
        value: f32,
    },

    /// An input slice length does not match the image size.
    #[error("{name} has {got} values; the image size needs {expected}")]
    ShapeMismatch {
        /// Name of the invalid input.
        name: &'static str,
        /// Required number of values.
        expected: usize,
        /// Supplied number of values.
        got: usize,
    },

    /// Input dimensions overflow the addressable slice length.
    #[error("{name} dimensions {dimensions:?} overflow usize")]
    DimensionOverflow {
        /// Name of the input.
        name: &'static str,
        /// Dimensions whose product overflowed.
        dimensions: Vec<usize>,
    },

    /// A camera matrix cannot be inverted.
    #[error("{name} must be finite and invertible")]
    SingularMatrix {
        /// Name of the singular matrix.
        name: &'static str,
    },

    /// A mesh has more vertices than its 32-bit indices can address.
    #[error("mesh has {vertex_count} vertices, exceeding the u32 index limit")]
    VertexIndexOverflow {
        /// Number of vertices produced before the overflow.
        vertex_count: usize,
    },
}
