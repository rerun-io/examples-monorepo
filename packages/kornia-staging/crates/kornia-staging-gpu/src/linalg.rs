//! The global BA's implicit PCG on the device (DESIGN §7, PR e1).
//!
//! [`GpuPcg`] is a [`PcgLane`] for `kornia_staging_slam::sqrt_ba::solve_global_with`:
//! each LM attempt's reduced camera system goes to the device in one upload
//! (blocks packed once into the buffer the runtime takes, split into pages
//! below the maximum binding size), and the whole PCG runs there. The host
//! waits on the device once every [`GBA_CG_CHECK_EVERY`] iterations, the same
//! constant the CPU lane uses for its residual replacement; the device stops
//! at exactly the iteration the CPU rules stop at. Each iteration is one
//! [`execute_exclusive`] section and each state read another, so frontend
//! work on the same device never waits behind more than one of them
//! (DESIGN §7.3).
//!
//! The block-Jacobi inverses come from the host, factored in f64 (DESIGN §7.3:
//! portable devices have no f64). The PCG scalars are compensated f32 sums in
//! one fixed order ([`fixed_order_dot`] states it), where the CPU lane sums in
//! f64; the S4 band (DESIGN §7.4) measures what is left.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use cubecl::prelude::*;
use kornia_staging_slam::sqrt_ba::{
    BlockJacobi, LaneSolve, PcgLane, PcgLaneError, PcgOptions, PcgSummary, PcgTermination,
    ReducedSystem, GBA_CG_CHECK_EVERY,
};

use crate::kernels::linalg::{self as kernels, PcgBuffers, PCG_UNITS};
use crate::kernels::Buffer;
use crate::runtime::{guarded, GpuError};
use crate::transfer::{binding_limits, execute_exclusive, read_buffers};

/// One page of blocks: their packed values and the metadata the kernels read.
#[derive(Debug, Default)]
struct Page {
    data: Vec<f32>,
    /// (page block, row in block) per row.
    row_meta: Vec<u32>,
    /// (page block, column in block) per column.
    col_meta: Vec<u32>,
    /// (data offset, rows, `t` offset, first slot, keyframes) per block.
    block_meta: Vec<u32>,
}

/// A [`ReducedSystem`] laid out for the device.
#[derive(Debug)]
struct Layout {
    pages: Vec<Page>,
    /// Keyframe of every (block, keyframe) slot, in block order.
    slot_kf: Vec<u32>,
    /// Pose-major CSR over the slots: `pose_slots[pose_ptr[k]..pose_ptr[k + 1]]`.
    pose_ptr: Vec<u32>,
    pose_slots: Vec<u32>,
    /// Rows over every block (`t`).
    rows: usize,
}

fn index(value: usize, what: &'static str) -> Result<u32, GpuError> {
    u32::try_from(value).map_err(|_| GpuError::PcgLayout { what })
}

impl Layout {
    /// Blocks in pages of at most `page_bytes`; every other buffer within
    /// `binding_bytes`.
    fn new(
        system: &ReducedSystem<'_, f32>,
        page_bytes: usize,
        binding_bytes: usize,
    ) -> Result<Self, GpuError> {
        let max_floats = page_bytes / size_of::<f32>();
        let mut pages: Vec<Page> = vec![Page::default()];
        let mut slot_kf: Vec<u32> = Vec::new();
        let mut rows = 0usize;
        for block in &system.blocks {
            let (m, k) = (block.rows, block.keyframes.len());
            let size = m * 6 * k;
            if size > max_floats {
                return Err(GpuError::PcgBlockTooLarge {
                    bytes: size * size_of::<f32>(),
                    limit: page_bytes,
                });
            }
            if pages
                .last()
                .is_some_and(|p| p.data.len() + size > max_floats)
            {
                pages.push(Page::default());
            }
            let page = pages
                .last_mut()
                .ok_or(GpuError::PcgLayout { what: "page" })?;
            let local = index(page.block_meta.len() / 5, "block")?;
            page.block_meta.extend([
                index(page.data.len(), "data offset")?,
                index(m, "rows")?,
                index(rows, "row offset")?,
                index(slot_kf.len(), "slot")?,
                index(k, "keyframes")?,
            ]);
            for r in 0..m {
                page.row_meta.extend([local, index(r, "row")?]);
            }
            for c in 0..6 * k {
                page.col_meta.extend([local, index(c, "column")?]);
                page.data.extend_from_slice(block.column(c));
            }
            for &kf in block.keyframes {
                slot_kf.push(index(kf, "keyframe")?);
            }
            rows += m;
        }
        pages.retain(|p| !p.data.is_empty());
        let poses = system.keyframes;
        let mut pose_ptr = vec![0u32; poses + 1];
        for &kf in &slot_kf {
            let kf = kf as usize;
            if kf >= poses {
                return Err(GpuError::PcgLayout { what: "keyframe" });
            }
            pose_ptr[kf + 1] += 1;
        }
        for k in 0..poses {
            pose_ptr[k + 1] += pose_ptr[k];
        }
        let mut cursor = pose_ptr.clone();
        let mut pose_slots = vec![0u32; slot_kf.len()];
        for (slot, &kf) in slot_kf.iter().enumerate() {
            pose_slots[cursor[kf as usize] as usize] = index(slot, "slot")?;
            cursor[kf as usize] += 1;
        }
        let largest = [rows, 6 * slot_kf.len(), 36 * poses]
            .into_iter()
            .max()
            .unwrap_or(0);
        if largest * size_of::<f32>() > binding_bytes {
            return Err(GpuError::PcgBlockTooLarge {
                bytes: largest * size_of::<f32>(),
                limit: binding_bytes,
            });
        }
        Ok(Self {
            pages,
            slot_kf,
            pose_ptr,
            pose_slots,
            rows,
        })
    }
}

/// One handle and its element count.
type Owned = (cubecl::server::Handle, usize);

fn view(owned: &Owned) -> Buffer<'_> {
    (&owned.0, owned.1)
}

/// Upload `values` without a second host copy: the vector becomes the bytes
/// the runtime takes. An empty vector uploads one zero, so every binding is
/// valid.
fn upload<R: Runtime, E: CubeElement + Default>(
    client: &ComputeClient<R>,
    mut values: Vec<E>,
) -> Owned {
    if values.is_empty() {
        values.push(E::default());
    }
    let count = values.len();
    (
        client.create(cubecl::bytes::Bytes::from_elems(values)),
        count,
    )
}

fn empty<R: Runtime>(client: &ComputeClient<R>, count: usize) -> Owned {
    let count = count.max(1);
    (client.empty(count * size_of::<f32>()), count)
}

/// One page on the device.
struct DevicePage {
    data: Owned,
    row_meta: Owned,
    col_meta: Owned,
    block_meta: Owned,
}

/// A reduced system on the device, with its scratch.
struct DeviceSystem {
    pages: Vec<DevicePage>,
    slot_kf: Owned,
    pose_ptr: Owned,
    pose_slots: Owned,
    t: Owned,
    contrib: Owned,
    lambda: f32,
    /// Bytes this system holds on the device.
    bytes: usize,
}

impl DeviceSystem {
    fn upload<R: Runtime>(client: &ComputeClient<R>, layout: Layout, lambda: f32) -> Self {
        let slots = layout.slot_kf.len();
        let pages: Vec<DevicePage> = layout
            .pages
            .into_iter()
            .map(|p| DevicePage {
                data: upload(client, p.data),
                row_meta: upload(client, p.row_meta),
                col_meta: upload(client, p.col_meta),
                block_meta: upload(client, p.block_meta),
            })
            .collect();
        let system = Self {
            pages,
            slot_kf: upload(client, layout.slot_kf),
            pose_ptr: upload(client, layout.pose_ptr),
            pose_slots: upload(client, layout.pose_slots),
            t: empty(client, layout.rows),
            contrib: empty(client, 6 * slots),
            lambda,
            bytes: 0,
        };
        let bytes = 4
            * (system
                .pages
                .iter()
                .map(|p| p.data.1 + p.row_meta.1 + p.col_meta.1 + p.block_meta.1)
                .sum::<usize>()
                + system.slot_kf.1
                + system.pose_ptr.1
                + system.pose_slots.1
                + system.t.1
                + system.contrib.1);
        Self { bytes, ..system }
    }

    /// `y = (J^T J + lambda I) x`; a no-op once `ustate` says stopped.
    fn matvec<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        x: Buffer<'_>,
        y: Buffer<'_>,
        ustate: Buffer<'_>,
    ) {
        for page in &self.pages {
            kernels::launch_block_gemv(
                client,
                view(&page.data),
                view(&page.row_meta),
                view(&page.col_meta),
                view(&page.block_meta),
                view(&self.slot_kf),
                x,
                view(&self.t),
                view(&self.contrib),
                ustate,
            );
        }
        kernels::launch_pose_reduce(
            client,
            view(&self.contrib),
            view(&self.pose_ptr),
            view(&self.pose_slots),
            x,
            y,
            ustate,
            self.lambda,
        );
    }
}

/// The largest storage binding the device takes, in bytes.
fn max_binding<R: Runtime>(client: &ComputeClient<R>) -> usize {
    binding_limits(client).1
}

fn read_f32<R: Runtime>(
    client: &ComputeClient<R>,
    owned: &Owned,
    what: &'static str,
) -> Result<Vec<f32>, GpuError> {
    let bytes = read_buffers(client, vec![owned.0.clone()], what)?;
    let values = f32::from_bytes(&bytes[0]);
    if values.len() < owned.1 {
        return Err(GpuError::ShortRead {
            what,
            actual: bytes[0].len(),
            expected: owned.1 * 4,
        });
    }
    Ok(values[..owned.1].to_vec())
}

#[cfg(feature = "test-support")]
/// `(J^T J + lambda I) x` on the device, the PCG's own matvec, with blocks
/// paged at `max_binding_bytes` (`None`: the device's limit).
///
/// # Errors
/// A block larger than a page, or a device failure.
pub fn device_matvec<R: Runtime>(
    client: &ComputeClient<R>,
    system: &ReducedSystem<'_, f32>,
    x: &[f32],
    max_binding_bytes: Option<usize>,
) -> Result<Vec<f32>, GpuError> {
    let limit = max_binding(client);
    let layout = Layout::new(system, max_binding_bytes.unwrap_or(limit).min(limit), limit)?;
    guarded(
        GpuError::DeviceLost {
            what: "the PCG matvec",
        },
        || {
            let device = DeviceSystem::upload(client, layout, system.lambda);
            let x = upload(client, x.to_vec());
            let y = empty(client, x.1);
            let ustate = upload(client, vec![0u32; 4]);
            device.matvec(client, view(&x), view(&y), view(&ustate));
            read_f32(client, &y, "the PCG matvec")
        },
    )
}

#[cfg(feature = "test-support")]
/// `a . b` in the order of the device PCG's sums: unit `u` of [`PCG_UNITS`]
/// adds the six entries of groups `u, u + 256, ...` in order into a Neumaier
/// (compensated) sum, then a fixed pairwise tree of TwoSums adds the units and
/// carries their errors. The host mirror of the device sum, for tests.
///
/// # Panics
/// If the lengths differ or are not a multiple of six.
pub fn fixed_order_dot(a: &[f32], b: &[f32]) -> f32 {
    assert!(
        a.len() == b.len() && a.len().is_multiple_of(6),
        "six-entry groups"
    );
    let (mut hi, mut lo) = ([0.0f32; PCG_UNITS], [0.0f32; PCG_UNITS]);
    for (k, (ga, gb)) in a
        .as_chunks::<6>()
        .0
        .iter()
        .zip(b.as_chunks::<6>().0)
        .enumerate()
    {
        let u = k % PCG_UNITS;
        for (x, y) in ga.iter().zip(gb) {
            let (sum, value) = (hi[u], x * y);
            let t = sum + value;
            lo[u] += if sum.abs() >= value.abs() {
                (sum - t) + value
            } else {
                (value - t) + sum
            };
            hi[u] = t;
        }
    }
    let mut stride = PCG_UNITS / 2;
    while stride > 0 {
        for u in 0..stride {
            let (a, b) = (hi[u], hi[u + stride]);
            let s = a + b;
            let bb = s - a;
            hi[u] = s;
            lo[u] += lo[u + stride] + ((a - (s - bb)) + (b - bb));
        }
        stride /= 2;
    }
    hi[0] + lo[0]
}

#[cfg(feature = "test-support")]
/// `a . b` on the device, by the kernel that sums the PCG's scalars.
///
/// # Errors
/// A device failure.
pub fn device_dot<R: Runtime>(
    client: &ComputeClient<R>,
    a: &[f32],
    b: &[f32],
) -> Result<f32, GpuError> {
    guarded(
        GpuError::DeviceLost {
            what: "the PCG dot",
        },
        || {
            let (a, b) = (upload(client, a.to_vec()), upload(client, b.to_vec()));
            let out = empty(client, 1);
            kernels::launch_dot(client, view(&a), view(&b), view(&out));
            Ok(read_f32(client, &out, "the PCG dot")?[0])
        },
    )
}

/// What the last [`GpuPcg`] solve cost.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DeviceSolveStats {
    /// Seconds to lay the system out and upload it.
    pub upload_s: f64,
    /// Bytes the solve held on the device.
    pub device_bytes: usize,
    /// Pages the blocks took.
    pub pages: usize,
    /// Host waits on the device.
    pub syncs: usize,
    /// Exclusive sections: one per iteration, one per state read.
    pub sections: usize,
    /// The longest exclusive section, seconds.
    pub longest_section_s: f64,
}

/// The device lane of the global BA's PCG.
pub struct GpuPcg<R: Runtime> {
    client: ComputeClient<R>,
    max_binding_bytes: usize,
    last: DeviceSolveStats,
    peak_device_bytes: usize,
    longest_section_s: f64,
    #[cfg(feature = "test-support")]
    residual: Vec<f32>,
}

impl<R: Runtime> GpuPcg<R> {
    /// A lane on `client`, paging at the device's maximum binding size.
    pub fn new(client: ComputeClient<R>) -> Self {
        let max_binding_bytes = max_binding(&client);
        Self {
            client,
            max_binding_bytes,
            last: DeviceSolveStats::default(),
            peak_device_bytes: 0,
            longest_section_s: 0.0,
            #[cfg(feature = "test-support")]
            residual: Vec::new(),
        }
    }

    /// Page the blocks at `bytes` instead, at most the device's limit (tests
    /// force small pages).
    #[must_use]
    pub fn with_max_binding_bytes(mut self, bytes: usize) -> Self {
        self.max_binding_bytes = bytes;
        self
    }

    /// The last solve's costs.
    pub fn last_solve(&self) -> DeviceSolveStats {
        self.last
    }

    /// The most bytes any solve bound on the device (the lane's own
    /// accounting, not the allocator's pool).
    pub fn peak_device_bytes(&self) -> usize {
        self.peak_device_bytes
    }

    /// The last solve's residual `r` as the device holds it (tests).
    #[cfg(feature = "test-support")]
    pub fn last_residual(&self) -> &[f32] {
        &self.residual
    }

    /// The longest exclusive section of any solve, seconds.
    pub fn longest_section_s(&self) -> f64 {
        self.longest_section_s
    }

    fn run(
        &mut self,
        system: &ReducedSystem<'_, f32>,
        preconditioner: &BlockJacobi<f32>,
        b: &[f32],
        x: &mut [f32],
        options: &PcgOptions,
    ) -> Result<LaneSolve, GpuError> {
        let n = b.len();
        if b.iter().all(|v| *v == 0.0) {
            x.fill(0.0);
            self.last = DeviceSolveStats::default();
            return Ok(LaneSolve {
                summary: PcgSummary {
                    iterations: 0,
                    termination: PcgTermination::ZeroRhs,
                },
                upload_s: 0.0,
                syncs: 0,
            });
        }
        let client = &self.client;
        let started = Instant::now();
        let limit = max_binding(client);
        let layout = Layout::new(system, self.max_binding_bytes.min(limit), limit)?;
        let pages = layout.pages.len();
        let minv: Vec<f32> = preconditioner
            .inverses()
            .iter()
            .flat_map(|m| (0..36).map(move |e| m[(e / 6, e % 6)]))
            .collect();
        let device = DeviceSystem::upload(client, layout, system.lambda);
        let minv = upload(client, minv);
        let b_dev = upload(client, b.to_vec());
        let x_dev = upload(client, vec![0.0f32; n]);
        let r = upload(client, b.to_vec());
        let (z, p, q) = (empty(client, n), empty(client, n), empty(client, n));
        let fstate = upload(client, vec![1.0f32, 0.0, 0.0, 0.0]);
        let ustate = upload(client, vec![0u32; 4]);
        // Wait for the copies here, outside any section: the upload is timed
        // whole and the first iteration's section does not carry it.
        cubecl::future::block_on(client.sync())
            .map_err(|error| crate::transfer::read_failed("the PCG upload", &error))?;
        let upload_s = started.elapsed().as_secs_f64();
        let device_bytes = device.bytes + 4 * (minv.1 + 2 * n + 4 * n + 8);
        let buffers = PcgBuffers {
            minv: view(&minv),
            b: view(&b_dev),
            x: view(&x_dev),
            r: view(&r),
            z: view(&z),
            p: view(&p),
            q: view(&q),
            fstate: view(&fstate),
            ustate: view(&ustate),
        };
        // A cap of 0 runs one iteration, as the CPU lane does.
        let max = u32::try_from(options.max_iterations.max(1))
            .map_err(|_| GpuError::PcgLayout { what: "iterations" })?;
        let eta = options.eta as f32;
        let mut syncs = 0usize;
        let mut longest = 0.0f64;
        let mut sections = 0usize;
        let mut it = 1u32;
        let state: Vec<u32> = loop {
            // One iteration per exclusive section (DESIGN 7.3): frontend work
            // waits behind one iteration at most.
            let section = Instant::now();
            execute_exclusive(client, "a PCG iteration", || {
                kernels::launch_pcg_precondition(client, &buffers, it);
                device.matvec(client, buffers.p, buffers.q, buffers.ustate);
                let replace = (it as usize).is_multiple_of(GBA_CG_CHECK_EVERY);
                let mode = if replace {
                    kernels::ADVANCE
                } else {
                    kernels::STEP
                };
                kernels::launch_pcg_update(client, &buffers, it, mode, max, eta);
                if replace {
                    device.matvec(client, buffers.x, buffers.q, buffers.ustate);
                    kernels::launch_pcg_update(client, &buffers, it, kernels::REPLACE, max, eta);
                }
                Ok::<_, GpuError>(())
            })?;
            longest = longest.max(section.elapsed().as_secs_f64());
            sections += 1;
            if !(it as usize).is_multiple_of(GBA_CG_CHECK_EVERY) && it < max {
                it += 1;
                continue;
            }
            // Wait for the device outside any section, then read the state in
            // a short section of its own.
            cubecl::future::block_on(client.sync())
                .map_err(|error| crate::transfer::read_failed("the PCG state", &error))?;
            let section = Instant::now();
            let state = execute_exclusive(client, "the PCG state", || {
                let bytes = read_buffers(client, vec![ustate.0.clone()], "the PCG state")?;
                Ok::<_, GpuError>(u32::from_bytes(&bytes[0]).to_vec())
            })?;
            longest = longest.max(section.elapsed().as_secs_f64());
            sections += 1;
            syncs += 1;
            if state.len() < 3 {
                return Err(GpuError::ShortRead {
                    what: "the PCG state",
                    actual: state.len() * 4,
                    expected: 12,
                });
            }
            if state[kernels::STOPPED] != 0 || it >= max {
                break state;
            }
            it += 1;
        };
        x.copy_from_slice(&read_f32(client, &x_dev, "the PCG solution")?);
        syncs += 1;
        #[cfg(feature = "test-support")]
        {
            self.residual = read_f32(client, &r, "the PCG residual")?;
        }
        let termination = match state[kernels::TERMINATION] {
            kernels::CONVERGED => PcgTermination::Converged,
            kernels::INDEFINITE => PcgTermination::Indefinite,
            kernels::NUMERICAL_FAILURE => PcgTermination::NumericalFailure,
            kernels::MAX_ITERATIONS => PcgTermination::MaxIterations,
            _ => {
                return Err(GpuError::PcgLayout {
                    what: "termination",
                });
            }
        };
        self.last = DeviceSolveStats {
            upload_s,
            device_bytes,
            pages,
            syncs,
            sections,
            longest_section_s: longest,
        };
        self.peak_device_bytes = self.peak_device_bytes.max(device_bytes);
        self.longest_section_s = self.longest_section_s.max(longest);
        Ok(LaneSolve {
            summary: PcgSummary {
                iterations: state[kernels::ITERATIONS] as usize,
                termination,
            },
            upload_s,
            syncs,
        })
    }
}

impl<R: Runtime> PcgLane<f32> for GpuPcg<R> {
    fn name(&self) -> &'static str {
        "gpu"
    }

    fn solve(
        &mut self,
        system: &ReducedSystem<'_, f32>,
        preconditioner: &BlockJacobi<f32>,
        b: &[f32],
        x: &mut [f32],
        options: &PcgOptions,
    ) -> Result<LaneSolve, PcgLaneError> {
        guarded(GpuError::DeviceLost { what: "the PCG" }, || {
            self.run(system, preconditioner, b, x, options)
        })
        .map_err(|error| PcgLaneError {
            lane: "gpu",
            reason: error.to_string(),
        })
    }
}

/// Probe systems: `POSES` keyframes and `LANDMARKS` blocks of 2-3 keyframes.
const PROBE_POSES: usize = 8;
const PROBE_LANDMARKS: usize = 200;
const PROBE_ITERATIONS: usize = 3;
/// Largest entry error of the probe against the host, relative to the largest entry.
const PROBE_TOLERANCE: f32 = 1e-4;

fn wobble(i: usize, k: usize) -> f32 {
    ((((i * 7919 + k * 104_729) % 9973) as f32) * 0.618_034).sin()
}

/// The probe's reduced system: block values and keyframes, row counts.
fn probe_system() -> (Vec<Vec<f32>>, Vec<Vec<usize>>) {
    let mut values = Vec::with_capacity(PROBE_LANDMARKS);
    let mut keyframes = Vec::with_capacity(PROBE_LANDMARKS);
    for l in 0..PROBE_LANDMARKS {
        let mut kfs: Vec<usize> = [l, 3 * l + 1, 5 * l + 2]
            .iter()
            .map(|v| v % PROBE_POSES)
            .collect();
        kfs.sort_unstable();
        kfs.dedup();
        let rows = 2 * kfs.len() + 3;
        values.push((0..rows * 6 * kfs.len()).map(|e| wobble(l, e)).collect());
        keyframes.push(kfs);
    }
    (values, keyframes)
}

/// GBA round trips ([`probe_pcg`]) run by this process so far, counted apart
/// from the device's capability probes, which have one owner
/// ([`crate::runtime::default_capabilities`]).
static PCG_ROUND_TRIPS: AtomicUsize = AtomicUsize::new(0);

/// How many GBA round trips this process has run.
pub fn pcg_round_trips_run() -> usize {
    PCG_ROUND_TRIPS.load(Ordering::SeqCst)
}

/// Make every later round trip's device result zeros, as a runtime that runs
/// no kernel would (tests of the fallback; one process each).
#[cfg(feature = "test-support")]
pub fn corrupt_round_trips() {
    CORRUPT.store(true, Ordering::SeqCst);
}

#[cfg(feature = "test-support")]
static CORRUPT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The round trip's verdict: every device entry within [`PROBE_TOLERANCE`] of
/// the host's largest entry, none NaN, and a host solution that is not zero.
///
/// # Errors
/// [`GpuError::PcgRoundTrip`] with the entries that disagree.
fn round_trip_agrees(host: &[f32], device: &[f32]) -> Result<(), GpuError> {
    let scale = host.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let wrong = host
        .iter()
        .zip(device)
        .filter(|(h, d)| {
            let error = (*h - *d).abs();
            error.is_nan() || error > PROBE_TOLERANCE * scale
        })
        .count()
        + host.len().abs_diff(device.len());
    if wrong == 0 && scale > 0.0 {
        Ok(())
    } else {
        Err(GpuError::PcgRoundTrip {
            wrong,
            count: host.len(),
        })
    }
}

/// The back end's GBA round trip on `client` (DESIGN §7.3): a small reduced
/// system (8 poses, 200 landmark blocks), three PCG iterations on the device
/// and on the host, every entry of the solution within 1e-4 of the host's
/// largest. Guards against a runtime that runs the kernels and returns zeros.
/// The solve holds its own exclusive sections; callers add none.
///
/// # Errors
/// [`GpuError::PcgRoundTrip`] when the device disagrees; a device failure.
pub fn probe_pcg<R: Runtime>(client: &ComputeClient<R>) -> Result<(), GpuError> {
    PCG_ROUND_TRIPS.fetch_add(1, Ordering::SeqCst);
    let (values, keyframes) = probe_system();
    let system = ReducedSystem {
        keyframes: PROBE_POSES,
        lambda: 1e-2f32,
        blocks: values
            .iter()
            .zip(&keyframes)
            .map(|(v, kfs)| {
                let rows = 2 * kfs.len() + 3;
                kornia_staging_slam::sqrt_ba::ReducedBlock {
                    keyframes: kfs,
                    values: v,
                    stride: rows,
                    offset: 0,
                    rows,
                }
            })
            .collect(),
    };
    let preconditioner = BlockJacobi::<f32>::from_blocks(&system.diagonal_blocks());
    let b: Vec<f32> = (0..6 * PROBE_POSES).map(|i| wobble(i, 7)).collect();
    let options = PcgOptions {
        max_iterations: PROBE_ITERATIONS,
        eta: f64::NEG_INFINITY,
    };
    let mut host = vec![0.0f32; b.len()];
    kornia_staging_slam::sqrt_ba::pcg(
        |v, out| system.matvec(v, out),
        &preconditioner,
        &b,
        &mut host,
        &options,
    );
    let mut device = vec![0.0f32; b.len()];
    GpuPcg::new(client.clone()).run(&system, &preconditioner, &b, &mut device, &options)?;
    #[cfg(feature = "test-support")]
    if CORRUPT.load(Ordering::SeqCst) {
        device.fill(0.0);
    }
    round_trip_agrees(&host, &device)
}

#[cfg(test)]
mod tests {
    use super::{round_trip_agrees, PROBE_TOLERANCE};

    #[test]
    fn the_round_trip_refuses_every_wrong_device_result() {
        let host: Vec<f32> = (0..48).map(|i| (i as f32 * 0.37).sin()).collect();
        let max = host.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(round_trip_agrees(&host, &host).is_ok());
        let wrong = |device: Vec<f32>| round_trip_agrees(&host, &device).unwrap_err().to_string();
        // Silent zeros: every entry but the host's own zero at i = 0.
        assert!(wrong(vec![0.0; 48]).contains("on 47 of 48"));
        let mut nan = host.clone();
        nan[5] = f32::NAN;
        assert!(wrong(nan).contains("on 1 of 48"));
        let mut off = host.clone();
        off[7] += 2.0 * PROBE_TOLERANCE * max;
        assert!(wrong(off).contains("on 1 of 48"));
        assert!(wrong(host[..40].to_vec()).contains("on 8 of 48"));
        // A zero host solution proves nothing.
        assert!(round_trip_agrees(&[0.0; 48], &[0.0; 48]).is_err());
    }
}
