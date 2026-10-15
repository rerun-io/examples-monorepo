//! Brush training observations converted to the native Rerun wire format.
mod conversion;
pub use conversion::{ConversionError, read_splats, retains, to_archetype};
