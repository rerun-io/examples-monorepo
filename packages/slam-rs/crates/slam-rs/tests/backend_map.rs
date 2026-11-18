//! The persistent keyframe map on the moving-ceiling rig (DESIGN a1).
//!
//! The invariant: the raw VIO is bit-identical with the back end on or off,
//! synchronous and deferred (D84) keyframes alike, because the window never
//! reads the map. The map budget is small here, so the over-budget batch cull
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

fn pipeline_with_map(deferred: bool, enabled: bool) -> api::Vio<f32> {
    let mut config = config(deferred);
    config.backend_enabled = enabled;
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
        let mut off = pipeline_with_map(deferred, false);
        let mut on = pipeline_with_map(deferred, true);
        assert!(off.map_backend().is_none());
        let without = run(&frames, &mut off);
        let with = run(&frames, &mut on);
        assert_eq!(
            with.results, without.results,
            "the back end moved a pose (deferred = {deferred})"
        );
        assert_eq!(with.solves, without.solves);

        on.finish_deferred_keyframe().unwrap();
        assert_eq!(on.estimator().missing_stamp_poses(), 0);
        let map_backend = on.map_backend().unwrap();
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
        let (copy, held) = allocated_by(|| map_backend.clone());
        drop(copy);
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
}
