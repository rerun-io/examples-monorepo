//! Model dispatch by a comptime tag (`MODEL_*`): each tag compiles to its own
//! shader with only that model's code, so a Brown8 caller's source is unchanged.
#![allow(missing_docs)]
use super::{
    brown8, double_sphere, equirect, eucm, kb4, MODEL_BROWN8, MODEL_DOUBLE_SPHERE, MODEL_EUCM,
    MODEL_KB4,
};
use cubecl::prelude::*;

/// Project with the camera tagged `model`; see each model's `project`.
#[cube]
pub fn project(
    params: &[f32],
    offset: usize,
    x: f32,
    y: f32,
    z: f32,
    out: &mut Array<f32>,
    #[comptime] model: u32,
) -> u32 {
    if comptime!(model == MODEL_BROWN8) {
        brown8::project(params, offset, x, y, z, out)
    } else if comptime!(model == MODEL_KB4) {
        kb4::project(params, offset, x, y, z, out, false)
    } else if comptime!(model == MODEL_DOUBLE_SPHERE) {
        double_sphere::project(params, offset, x, y, z, out, false)
    } else if comptime!(model == MODEL_EUCM) {
        eucm::project(params, offset, x, y, z, out, false)
    } else {
        // MODEL_EQUIRECT. A kernel cannot panic, so this arm does not check the tag: the host takes tags from
        // `DeviceCamera::model_id` and sizes every layout with `parameter_count`, which panics on any other tag.
        equirect::project(params, offset, x, y, z, out, false)
    }
}

/// Unproject with the camera tagged `model`; see each model's `unproject`.
#[cube]
pub fn unproject(
    params: &[f32],
    offset: usize,
    px: f32,
    py: f32,
    out: &mut Array<f32>,
    #[comptime] model: u32,
) -> u32 {
    if comptime!(model == MODEL_BROWN8) {
        brown8::unproject(params, offset, px, py, out)
    } else if comptime!(model == MODEL_KB4) {
        kb4::unproject(params, offset, px, py, out)
    } else if comptime!(model == MODEL_DOUBLE_SPHERE) {
        double_sphere::unproject(params, offset, px, py, out)
    } else if comptime!(model == MODEL_EUCM) {
        eucm::unproject(params, offset, px, py, out)
    } else {
        // MODEL_EQUIRECT. A kernel cannot panic, so this arm does not check the tag: the host takes tags from
        // `DeviceCamera::model_id` and sizes every layout with `parameter_count`, which panics on any other tag.
        equirect::unproject(params, offset, px, py, out)
    }
}

/// Project and write the row-major 2x3 point Jacobian into `out[2..8]`.
#[cube]
pub fn project_with_point_jacobian(
    params: &[f32],
    offset: usize,
    x: f32,
    y: f32,
    z: f32,
    out: &mut Array<f32>,
    #[comptime] model: u32,
) -> u32 {
    if comptime!(model == MODEL_BROWN8) {
        brown8::project_with_point_jacobian(params, offset, x, y, z, out)
    } else if comptime!(model == MODEL_KB4) {
        kb4::project(params, offset, x, y, z, out, true)
    } else if comptime!(model == MODEL_DOUBLE_SPHERE) {
        double_sphere::project(params, offset, x, y, z, out, true)
    } else if comptime!(model == MODEL_EUCM) {
        eucm::project(params, offset, x, y, z, out, true)
    } else {
        // MODEL_EQUIRECT. A kernel cannot panic, so this arm does not check the tag: the host takes tags from
        // `DeviceCamera::model_id` and sizes every layout with `parameter_count`, which panics on any other tag.
        equirect::project(params, offset, x, y, z, out, true)
    }
}
