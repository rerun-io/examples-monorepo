//! Which keyframes the map gives up (RootSLAM §III-A, `map_database.cpp:73-162,
//! 698-850`).
//!
//! Two rules. A keyframe leaving the window is culled at once when it is
//! redundant ([`KeyframeMap::redundancy_score`] at or above
//! `map_max_redundancy_score`). A map over `map_max_keyframes` culls the
//! `map_batch_cull_size` lowest-scored keyframes. Neither ever touches a
//! protected keyframe: the root and the window's keyframes. Loop closing needs
//! no protection: it detects and applies in one query, with no cull between
//! (B10: RootSLAM leaves an island unprotected between its request and the
//! response, then reads it with `.at()`).
//!
//! The score is RootSLAM's weighted sum, computed from the map as it is when
//! the batch runs, where RootSLAM freezes the coverage and landmark terms when
//! the keyframe leaves the window. The batch itself
//! ([`super::MapBackend`]'s over-budget cull) re-ranks after each cull, as
//! RootSLAM's `handleGraphScoreChange` does.

use crate::types::FrameId;

use super::covisibility::CovisibilityGraph;
use super::map::KeyframeMap;

/// RootSLAM's culling score of keyframe `id` (`computeKeyframeScore`): low is
/// replaceable. `0.4` graph (few strong edges), `0.2` loops, `0.1` distance
/// from the middle of the map's time span, `0.1` image coverage, `0.2`
/// landmarks seen.
pub fn keyframe_score(
    map: &KeyframeMap,
    graph: &CovisibilityGraph,
    id: FrameId,
    high_weight: u32,
    resolutions: &[[u32; 2]],
) -> f32 {
    let first: Option<FrameId> = map.keyframes().keys().next().copied();
    let last: Option<FrameId> = map.keyframes().keys().next_back().copied();
    let time_score: f32 = match (first, last) {
        (Some(first), Some(last)) if last > first => {
            let t_norm: f64 = (id - first) as f64 / (last - first) as f64;
            (2.0 * t_norm - 1.0).abs() as f32
        }
        _ => 0.0,
    };
    0.4 * graph.graph_score(id, high_weight)
        + 0.2 * graph.loop_score(id)
        + 0.1 * time_score
        + 0.1 * map.image_coverage(id, resolutions)
        + 0.2 * map.observed_landmarks_score(id)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::collections::BTreeMap;

    use super::*;
    use crate::lie::Se3;

    fn map_with(ids: &[FrameId]) -> KeyframeMap {
        let mut map = KeyframeMap::new();
        for &id in ids {
            map.upsert_keyframe(id, Se3::identity(), None);
        }
        map
    }

    #[test]
    fn the_score_weighs_graph_loops_time_coverage_and_landmarks() {
        let map = map_with(&[10, 20, 30, 40, 50]);
        let mut graph = CovisibilityGraph::new();
        graph.insert_keyframe(10, &BTreeMap::new());
        for id in [20, 30, 40, 50] {
            graph.insert_keyframe(id, &BTreeMap::from([(id - 10, 5)]));
        }
        graph.add_loop_closure(20, 50);
        // Every keyframe: graph score 1 (no edge reaches 100), no coverage, no
        // landmarks. Time: 10 and 50 score 1, 20 and 40 score 0.5, 30 scores 0.
        // Loops lift 20 and 50.
        let score = |id| keyframe_score(&map, &graph, id, 100, &[]);
        assert!((score(30) - 0.4).abs() < 1e-6);
        assert!((score(40) - 0.45).abs() < 1e-6);
        assert!((score(20) - 0.55).abs() < 1e-6);
        assert!((score(10) - 0.5).abs() < 1e-6);
    }
}
