//! The persistent keyframe map on the moving-ceiling rig (DESIGN a1).
//!
//! The invariant: the raw VIO is bit-identical with the back end off, Offline
//! or Realtime (PR a2), synchronous and deferred (D84) keyframes alike, because
//! the window never reads the map; and Realtime, with no loop query to drop
//! yet, builds the map Offline builds. The map budget is small here, so the over-budget batch cull
//! runs inside the comparison. And the map holds what the window handed it:
//! more keyframes than the window, the window keyframes still active, solved
//! poses, landmarks with two or more observations, and no image.

#![cfg(feature = "backend")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use slam_rs as api;
#[path = "common/deferred.rs"]
#[allow(dead_code)]
mod fixture;
use fixture::*;

// ── a counting allocator, so retention is measured, not asserted ──────────

thread_local! {
    /// Whether this thread is inside [`allocated_by`].
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    /// Bytes this thread asked for since the gate opened.
    static ALLOCATED_BYTES: Cell<usize> = const { Cell::new(0) };
}

/// Add `bytes` when this thread is measuring. `try_with`: a thread tearing down
/// its locals degrades to "not counting" instead of panicking inside `alloc`.
fn count(bytes: usize) {
    if COUNTING.try_with(Cell::get).unwrap_or(false) {
        let _ = ALLOCATED_BYTES.try_with(|slot| slot.set(slot.get() + bytes));
    }
}

struct Counting;

// SAFETY: every method forwards to `System` with the layout it was given. The
// only addition is a thread-local add on `Cell`s with `const` initialisers,
// which cannot allocate (see tests/frame_allocations.rs).
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Bytes `body` asks the allocator for on this thread.
fn allocated_by<T>(body: impl FnOnce() -> T) -> (T, usize) {
    ALLOCATED_BYTES.set(0);
    COUNTING.set(true);
    let value: T = body();
    COUNTING.set(false);
    (value, ALLOCATED_BYTES.get())
}

// ── the rig ───────────────────────────────────────────────────────────────

/// Map budget of the run: small, so the batch cull runs several times.
const MAX_KEYFRAMES: usize = 6;

fn pipeline_with_map(deferred: bool, mode: Option<api::config::BackendMode>) -> api::Vio<f32> {
    pipeline_with(deferred, mode, false)
}

fn pipeline_with(
    deferred: bool,
    mode: Option<api::config::BackendMode>,
    lag: bool,
) -> api::Vio<f32> {
    let mut config = config(deferred);
    config.port_frontend_lag = lag;
    config.backend_enabled = mode.is_some();
    config.backend_mode = mode.unwrap_or_default();
    // More keyframes and a short keyframe budget, so
    // keyframes leave the window and the map outgrows it.
    config.vio_new_kf_keypoints_thresh = 0.9;
    config.vio_max_kfs = 3;
    config.map_max_keyframes = MAX_KEYFRAMES;
    config.map_batch_cull_size = 2;
    api::Vio::new(
        config,
        calibration(),
        api::frontend::flow::FrontendOptions {
            threads: 2,
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn the_map_never_moves_a_vio_pose_while_it_culls() {
    let frames: Vec<[Vec<u8>; 4]> = (0..FRAMES)
        .map(|index| {
            let t_s = index as f64 * FRAME_NS as f64 / 1e9;
            std::array::from_fn(|camera| render(t_s, camera))
        })
        .collect();

    for deferred in [false, true] {
        let mut off = pipeline_with_map(deferred, None);
        let mut on = pipeline_with_map(deferred, Some(api::config::BackendMode::Offline));
        let mut realtime = pipeline_with_map(deferred, Some(api::config::BackendMode::Realtime));
        assert!(off.map_snapshot().is_none());
        let without = run(&frames, &mut off);
        let with = run(&frames, &mut on);
        let threaded = run(&frames, &mut realtime);
        assert_eq!(
            with.results, without.results,
            "the back end moved a pose (deferred = {deferred})"
        );
        assert_eq!(with.solves, without.solves);
        // Realtime moves no pose either, and with no loop query to drop yet
        // it builds the very map Offline builds.
        assert_eq!(
            threaded.results, without.results,
            "the Realtime back end moved a pose (deferred = {deferred})"
        );
        assert_eq!(threaded.solves, without.solves);

        on.finish_deferred_keyframe().unwrap();
        realtime.finish_deferred_keyframe().unwrap();
        realtime.flush_backend();
        assert_eq!(realtime.map_snapshot(), on.map_snapshot());
        // f1: without a loop, Realtime and Offline give the same final
        // trajectory.
        let raw = |results: &[api::VioResult]| -> Vec<(i64, [f64; 7])> {
            results
                .iter()
                .filter_map(|result| result.pose.map(|pose| (result.t_ns, pose.world_from_rig)))
                .collect()
        };
        let offline_map = on.map_snapshot().unwrap();
        let offline_final = offline_map.reanchor(&raw(&with.results)).unwrap();
        let realtime_final = realtime
            .map_snapshot()
            .unwrap()
            .reanchor(&raw(&threaded.results))
            .unwrap();
        assert_eq!(realtime_final, offline_final);
        assert_eq!(offline_final.len(), raw(&with.results).len());
        // Each map keyframe lands on its map pose.
        let raw_t_ns: Vec<i64> = raw(&with.results).iter().map(|&(t_ns, _)| t_ns).collect();
        for (t_ns, pose) in offline_map
            .keyframe_t_ns
            .iter()
            .zip(&offline_map.keyframe_poses)
        {
            let row = raw_t_ns.binary_search(t_ns).unwrap();
            let [x, y, z, ..] = offline_final[row];
            assert!((nalgebra::Vector3::new(x, y, z) - pose.translation).norm() < 1e-12);
        }
        let counters = realtime.backend_counters().unwrap();
        assert_eq!(counters, {
            let mut offline = on.backend_counters().unwrap();
            offline.max_pending_stamps = counters.max_pending_stamps;
            offline.step = counters.step;
            offline.query = counters.query;
            offline.publish = counters.publish;
            offline
        });
        // The flush posted the view Realtime readers get; Offline takes its views on demand.
        assert!(counters.publish.steps >= 1, "{counters:?}");
        assert_eq!(on.backend_counters().unwrap().publish.steps, 0);
        assert!(counters.max_pending_stamps >= 1, "{counters:?}");
        assert_eq!(on.estimator().missing_stamp_poses(), 0);
        let queries_run = on.backend_counters().unwrap().queries_run;
        on.read_map(|map_backend| check_map(&on, map_backend, deferred, queries_run))
            .unwrap();
    }
}

/// With `port.frontend_lag` (robocap-live turns it on) the estimator finishes
/// frameset t-1 beside the frontend of t, which already holds t's images:
/// loop closing must still describe each keyframe from its own images.
#[test]
fn a_lagged_pipeline_hands_loop_closing_its_keyframe_images() {
    let mut vio = pipeline_with(false, Some(api::config::BackendMode::Offline), true);
    let t0: i64 = 1_000_000_000;
    let mut imu_t: i64 = t0 - 100_000_000;
    for index in 0..FRAMES {
        let t_ns: i64 = t0 + index * FRAME_NS;
        while imu_t <= t_ns + 5_000_000 {
            let t_s = ((imu_t - t0) as f64 / 1e9).max(0.0);
            let accel = acceleration(t_s) + nalgebra::Vector3::new(0.0, 0.0, 9.81);
            vio.push_imu(imu_t, [0.0; 3], accel.into()).unwrap();
            imu_t += 5_000_000;
        }
        let frames: [Vec<u8>; 4] =
            std::array::from_fn(|camera| render((t_ns - t0) as f64 / 1e9, camera));
        let views: Vec<api::ImageView<'_>> = frames
            .iter()
            .map(|data| api::ImageView {
                width: WIDTH,
                height: HEIGHT,
                stride: WIDTH,
                data,
            })
            .collect();
        vio.track(t_ns, &views).unwrap();
    }
    vio.flush().unwrap();
    vio.read_map(|map_backend| {
        let loops = map_backend.loop_closing().unwrap();
        assert!(loops.counts().indexed > 3, "{:?}", loops.counts());
        assert_eq!(loops.counts().without_images, 0, "{:?}", loops.counts());
        assert_eq!(loops.held_images(), 0);
    })
    .unwrap();
}

/// What the map holds after the run, and what it costs.
fn check_map(
    on: &api::Vio<f32>,
    map_backend: &api::backend::MapBackend,
    deferred: bool,
    queries_run: u64,
) {
    let map = map_backend.map();
    let stats = map_backend.stats();
    let culls = map_backend.cull_counts();
    let window: Vec<i64> = on.estimator().kf_ids().collect();
    eprintln!("deferred = {deferred}: {stats:?}, {culls:?}, window {window:?}");
    // The batch cull ran inside the bit-identical comparison and held the
    // budget.
    assert!(culls.over_budget > 0, "{culls:?}");
    assert!(stats.keyframes <= MAX_KEYFRAMES, "{stats:?}");
    // The map is a superset of the window, and active = the window.
    assert!(stats.keyframes > window.len(), "{stats:?}");
    assert_eq!(
        map_backend
            .active_keyframes()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        window
    );
    // Window keyframes carry a solved pose: the one of the last keyframe
    // solve, which later framesets may refine a little.
    let snapshot = on.estimator().snapshot();
    for id in &window {
        let solved = snapshot
            .states
            .iter()
            .chain(snapshot.poses.iter())
            .find(|state| state.t_ns == *id)
            .map(|state| state.t_w_i.cast::<f64>())
            .unwrap();
        let stored = map.keyframe(*id).unwrap().pose;
        assert!((stored.translation - solved.translation).norm() < 1e-3);
    }
    // Loop closing (on by default with the back end) described every new
    // keyframe from the images its stamp carried, and kept none after its
    // query; the deferred solve's stamps carry them too.
    let loops = map_backend.loop_closing().unwrap();
    assert!(loops.counts().indexed > 0, "{:?}", loops.counts());
    assert_eq!(loops.counts().without_images, 0, "{:?}", loops.counts());
    assert_eq!(loops.counts().indexed, queries_run);
    assert_eq!(loops.held_images(), 0);
    assert!(stats.landmarks > 20, "{stats:?}");
    assert!(stats.observations >= 2 * stats.landmarks);
    assert!(
        map.landmarks()
            .values()
            .all(|landmark| landmark.obs.len() >= 2)
    );
    // The graph spans the map with one tree.
    assert_eq!(map_backend.graph().len(), stats.keyframes);
    assert_eq!(
        map_backend.graph().root(),
        map.keyframes().keys().next().copied()
    );
    assert!(
        map.keyframes()
            .values()
            .all(|keyframe| keyframe.keypoints.len() == 4)
    );

    // Retention (DESIGN §2.2), measured: a deep copy of the back end asks the
    // allocator for every buffer the back end holds (about 37 KB per
    // keyframe on this rig). One camera's image kept per keyframe, at
    // pyramid level 0 or 1 (WIDTH x HEIGHT or WIDTH x HEIGHT / 4 bytes), would
    // break the per-keyframe bound.
    // Loop closing's part (descriptors and bags of words, 32 B a keypoint
    // plus the trees; no image once its query ran) is measured on its own,
    // under the same per-keyframe bound.
    let (copy, total) = allocated_by(|| map_backend.clone());
    drop(copy);
    let (loops_copy, loops_held) = allocated_by(|| map_backend.loop_closing().cloned());
    drop(loops_copy);
    eprintln!(
        "deferred = {deferred}: loop closing holds {loops_held} B ({} B per keyframe)",
        loops_held / stats.keyframes
    );
    assert!(loops_held / stats.keyframes < WIDTH * HEIGHT / 4);
    let held: usize = total - loops_held;
    let per_keyframe: usize = held / stats.keyframes;
    let counted: usize = map.retained_bytes();
    eprintln!(
        "deferred = {deferred}: back end holds {held} B ({per_keyframe} B per keyframe); \
             retained_bytes counts {counted} B"
    );
    assert!(
        per_keyframe < WIDTH * HEIGHT / 4,
        "{per_keyframe} B per keyframe"
    );
    // `retained_bytes` (reported for long runs) is a lower bound within a
    // known factor of what the allocator holds.
    assert!(counted <= held, "{counted} > {held}");
    assert!(held <= 4 * counted, "{held} > 4 x {counted}");
}
