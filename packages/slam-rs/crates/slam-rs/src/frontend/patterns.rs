//! Application config identifiers for staged sampling patterns.
use kornia_staging_imgproc::optical_flow::patch_se2::{Pattern, Pattern51, Pattern52};

/// Associate an application config code with a geometric sampling pattern.
pub trait ConfiguredPattern: Pattern {
    /// Value stored as optical_flow_pattern in the SLAM configuration.
    const CODE: i32;
}
impl ConfiguredPattern for Pattern51 {
    const CODE: i32 = 51;
}
impl ConfiguredPattern for Pattern52 {
    const CODE: i32 = 52;
}
