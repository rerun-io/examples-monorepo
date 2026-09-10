//! Every CubeCL kernel the frontend runs, and the launchers that dispatch them.
//!
//! The bodies are ports of the CPU functions named on each one, not rewrites:
//! [`crate::pyramid::subsample`] is `subsample_kernel`, `patch::build_patch` is
//! `patch_build_kernel`, and `tracker::track_point` with
//! `tracker::track_point_at_level` is `klt_kernel`. Where the CPU port fixed an
//! arithmetic order because `f32` makes it load-bearing, this file fixes the
//! same one: **every reduction over pattern taps runs on unit 0 in ascending
//! tap order** (decision D21's "reductions by hand", decision D31's fixed
//! order), so a repeat run is bit-identical and the only differences from the
//! CPU are the ones shader contraction of `a * b + c` introduces.
//!
//! ## One cube per patch
//!
//! The per-patch kernels launch [`TAP_UNITS`] units per cube and give unit `t`
//! pattern tap `t`. Fifty-two taps sampled in parallel, then a serial
//! reduction on unit 0: the alternative, one unit per patch, leaves 400 units
//! to hide 200 dependent loads each and measured far worse in the design
//! sketch. Every `sync_cube` sits at a cube-uniform point — the validity flag
//! lives in shared memory and is read after a barrier, never inside a
//! per-tap branch — because a barrier some units skip hangs the cube.
//!
//! ## The `meta` array
//!
//! One `u32` buffer carries a pyramid's level geometry and the sampling
//! pattern, so no stage needs a seventh binding (§12.3: at most six buffers per
//! stage, which is also wgpu's floor).
//!
//! `u32` and not `f32`, because a level base is an **index**: it is added to a
//! pixel offset here, and above `2^24` an `f32` holds only every second
//! integer. A 4097x4097 frame puts level 2's base at 16,785,409, which `f32`
//! stores as 16,785,408, and every level-2 sample read one pixel early on both
//! lanes with nothing reporting it (the S25 review). WGSL has no `u16` or `u8`
//! but it does have `u32`, so this is the one exact carrier all three shader
//! compilers store natively. The pattern taps are the only genuinely
//! fractional values in the array, and they ride in it as their bit patterns
//! rather than in a binding of their own: a bitcast is one instruction
//! everywhere, and a seventh buffer is a portability question on every device
//! the portable lane runs on.
//!
//! ```text
//! meta[level * 4 + 0]  base offset of the level inside its buffer, in pixels
//! meta[level * 4 + 1]  width
//! meta[level * 4 + 2]  height
//! meta[level * 4 + 3]  0 = buffer `a`, 1 = buffer `b`  (the level's parity)
//! meta[levels * 4 + tap * 2 + 0]  pattern tap x, as `f32::to_bits`
//! meta[levels * 4 + tap * 2 + 1]  pattern tap y, as `f32::to_bits`
//! ```
//!
//! ## The `store` array
//!
//! One `f32` buffer per patch set, three sections, each with the patch index
//! fast-varying (§12.2). `C` is the capacity, `T` the pattern size, `L` the
//! level count; the offsets are [`data_offset`], [`jacobian_offset`] and
//! [`valid_offset`] on the host side.
//!
//! ```text
//! data     [(level * T + tap) * C + patch]                  from 0
//! h_inv_jt [((level * 3 + row) * T + tap) * C + patch]      from L*T*C
//! valid    [level * C + patch]                              from 4*L*T*C
//! ```
//!
//! ## The transform arrays
//!
//! `[m00, m01, m10, m11, tx, ty]` in six runs of `count`, as
//! [`crate::frontend::tracker::FlowTransforms`] holds them; a tracker result
//! adds a seventh run for the validity flag.
//!
//! ## One CubeCL trap this file works around everywhere
//!
//! An `if` expression used **directly** as the right-hand side of an array or
//! shared-memory store miscompiles. `inverse[r * 3 + c] = if r == c { 1.0 }
//! else { 0.0 }` left the array untouched, which turned every patch Hessian
//! inverse into zeros and dropped every keypoint, with nothing reported
//! anywhere. The same shape *returned* from a `#[cube]` function
//! (`reflect_high`) is fine. So every conditional value below is bound to a
//! local first and the local is stored, which is also why the two kernels that
//! do it allow `unused_assignments`: the initialiser exists to give the local a
//! type, not to be read.

mod cell_select;
mod fast;
mod layout;
mod patch;
mod pyramid;
mod sampling;
mod track;

pub(super) use cell_select::{CellSelectGeometry, launch_fast_cell_select};
pub(super) use fast::{
    MASK_BITS, RING_BIAS, launch_fast_localmax, launch_fast_mask, launch_fast_score,
};
pub(super) use layout::{MAX_CUBES_PER_DIM, PatchShape, PositionBases};
pub(super) use patch::launch_patch_build;
pub(super) use pyramid::launch_subsample;
pub(super) use sampling::{launch_copy_level0, launch_probe};
pub(super) use track::{launch_finish, launch_klt, launch_prepare_backward};
