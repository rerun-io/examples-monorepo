//! The covisibility graph and its spanning tree (RootSLAM §III-A,
//! `covisibility_graph.cpp`).
//!
//! A node per map keyframe. An edge's weight is the number of landmarks the two
//! keyframes share, counted when the newer one joins the map and lowered as
//! culling removes landmarks. Each keyframe's spanning-tree parent is its
//! strongest neighbour when it joins; loop-closure edges come from loop closing.
//!
//! Differences from RootSLAM, all deliberate:
//! - nodes are explicit, not "has an edge": a keyframe whose edges all reach
//!   zero stays a node with its tree links (RootSLAM re-inserts it and drops its
//!   children list);
//! - [`CovisibilityGraph::decrement_edge`] touches only an existing edge between
//!   existing nodes (B19: `decrementEdge` creates both nodes through
//!   `operator[]`);
//! - an edge counts shared *landmarks*; RootSLAM counts every observation of
//!   the other keyframe, so a landmark seen by two of its cameras counts twice;
//! - "high covisibility" is a threshold passed in at query time, not a set kept
//!   in step with one fixed value, so the threshold can follow the map;
//! - every order is by id (`BTreeMap`), so ties resolve to the smaller id.

use std::collections::{BTreeMap, BTreeSet};

use crate::types::FrameId;

/// The three kinds of pose-graph edge (`PoseGraphEdgeType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PoseGraphEdgeKind {
    /// Parent to child in the spanning tree.
    SpanningTree,
    /// A verified loop closure.
    LoopClosure,
    /// A covisibility edge at or above the threshold that is not a tree edge.
    HighCovisibility,
}

/// One edge of the pose graph a loop closure optimizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PoseGraphEdge {
    /// The parent for a tree edge, the smaller id otherwise.
    pub from: FrameId,
    /// The child for a tree edge, the larger id otherwise.
    pub to: FrameId,
    /// What kind of edge this is.
    pub kind: PoseGraphEdgeKind,
}

#[derive(Debug, Clone, Default, PartialEq)]
struct Node {
    parent: Option<FrameId>,
    children: BTreeSet<FrameId>,
    edges: BTreeMap<FrameId, u32>,
    loops: BTreeSet<FrameId>,
}

/// Weighted covisibility graph, spanning tree and loop edges over map keyframes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CovisibilityGraph {
    nodes: BTreeMap<FrameId, Node>,
    root: Option<FrameId>,
}

impl CovisibilityGraph {
    /// An empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// The first keyframe: the tree's root and the one node never removed.
    pub fn root(&self) -> Option<FrameId> {
        self.root
    }

    /// Whether `id` is a node.
    pub fn contains(&self, id: FrameId) -> bool {
        self.nodes.contains_key(&id)
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph has no node.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Every node, in id order.
    pub fn node_ids(&self) -> impl Iterator<Item = FrameId> + '_ {
        self.nodes.keys().copied()
    }

    /// The weight of the edge `a`-`b`, if there is one.
    pub fn weight(&self, a: FrameId, b: FrameId) -> Option<u32> {
        self.nodes.get(&a)?.edges.get(&b).copied()
    }

    /// `id`'s neighbours and edge weights, in id order.
    pub fn neighbours(&self, id: FrameId) -> impl Iterator<Item = (FrameId, u32)> + '_ {
        self.nodes
            .get(&id)
            .into_iter()
            .flat_map(|node| node.edges.iter().map(|(&other, &weight)| (other, weight)))
    }

    /// `id`'s spanning-tree parent; `None` for the root or an unknown id.
    pub fn parent(&self, id: FrameId) -> Option<FrameId> {
        self.nodes.get(&id)?.parent
    }

    /// `id`'s spanning-tree children, in id order.
    pub fn children(&self, id: FrameId) -> impl Iterator<Item = FrameId> + '_ {
        self.nodes
            .get(&id)
            .into_iter()
            .flat_map(|node| node.children.iter().copied())
    }

    /// The keyframes `id` closed a loop with, in id order.
    pub fn loop_closures(&self, id: FrameId) -> impl Iterator<Item = FrameId> + '_ {
        self.nodes
            .get(&id)
            .into_iter()
            .flat_map(|node| node.loops.iter().copied())
    }

    /// Add keyframe `id` with its shared-landmark counts to existing nodes
    /// (`updateCovisibilityGraph`). Counts naming a missing node or `id` itself
    /// are ignored. The first keyframe becomes the root. Every later one gets a
    /// parent: its strongest neighbour (the smaller id on a tie), or, sharing
    /// nothing, the newest older node through a zero-weight edge, so the tree
    /// stays one component. Returns `false`, changing nothing, if `id` is
    /// already a node.
    pub fn insert_keyframe(&mut self, id: FrameId, shared: &BTreeMap<FrameId, u32>) -> bool {
        if self.nodes.contains_key(&id) {
            return false;
        }
        let Some(_) = self.root else {
            self.root = Some(id);
            self.nodes.insert(id, Node::default());
            return true;
        };
        let mut edges: BTreeMap<FrameId, u32> = shared
            .iter()
            .filter(|(other, _)| **other != id && self.nodes.contains_key(other))
            .map(|(&other, &weight)| (other, weight))
            .collect();
        if edges.is_empty() {
            let previous: Option<FrameId> = self
                .nodes
                .range(..id)
                .next_back()
                .map(|(&other, _)| other)
                .or(self.root);
            if let Some(previous) = previous {
                edges.insert(previous, 0);
            }
        }
        let mut parent: Option<(FrameId, u32)> = None;
        for (&other, &weight) in &edges {
            if parent.is_none_or(|(_, best)| weight > best) {
                parent = Some((other, weight));
            }
        }
        for (&other, &weight) in &edges {
            if let Some(node) = self.nodes.get_mut(&other) {
                node.edges.insert(id, weight);
            }
        }
        let parent: Option<FrameId> = parent.map(|(other, _)| other);
        if let Some(parent) = parent.and_then(|parent| self.nodes.get_mut(&parent)) {
            parent.children.insert(id);
        }
        self.nodes.insert(
            id,
            Node {
                parent,
                edges,
                ..Node::default()
            },
        );
        true
    }

    /// Lower the edge `a`-`b` by `by`, removing it at zero. Does nothing if
    /// either node or the edge is missing (B19).
    pub fn decrement_edge(&mut self, a: FrameId, b: FrameId, by: u32) {
        let Some(weight) = self.weight(a, b) else {
            return;
        };
        let lowered: u32 = weight.saturating_sub(by);
        for (from, to) in [(a, b), (b, a)] {
            if let Some(node) = self.nodes.get_mut(&from) {
                if lowered == 0 {
                    node.edges.remove(&to);
                } else {
                    node.edges.insert(to, lowered);
                }
            }
        }
    }

    /// Record a loop closure between two existing nodes; `false` if either is
    /// missing or they are the same.
    pub fn add_loop_closure(&mut self, a: FrameId, b: FrameId) -> bool {
        if a == b || !self.contains(a) || !self.contains(b) {
            return false;
        }
        for (from, to) in [(a, b), (b, a)] {
            if let Some(node) = self.nodes.get_mut(&from) {
                node.loops.insert(to);
            }
        }
        true
    }

    /// Remove keyframe `id` (`removeNode`): its edges, its loop edges, and its
    /// place in the tree, where its children move to its parent. The root is
    /// never removed; `false` for the root or an unknown id.
    pub fn remove_keyframe(&mut self, id: FrameId) -> bool {
        if self.root == Some(id) {
            return false;
        }
        let Some(node) = self.nodes.remove(&id) else {
            return false;
        };
        for other in node.edges.keys().chain(node.loops.iter()) {
            if let Some(other) = self.nodes.get_mut(other) {
                other.edges.remove(&id);
                other.loops.remove(&id);
            }
        }
        if let Some(parent) = node.parent.and_then(|parent| self.nodes.get_mut(&parent)) {
            parent.children.remove(&id);
            parent.children.extend(node.children.iter().copied());
        }
        for child in &node.children {
            if let Some(child) = self.nodes.get_mut(child) {
                child.parent = node.parent;
            }
        }
        true
    }

    /// Edges at or above `threshold`, each once as `(smaller, larger)`, in order.
    pub fn high_covisibility_edges(&self, threshold: u32) -> Vec<(FrameId, FrameId)> {
        self.nodes
            .iter()
            .flat_map(|(&a, node)| {
                node.edges
                    .range(a + 1..)
                    .filter(move |(_, weight)| **weight >= threshold)
                    .map(move |(&b, _)| (a, b))
            })
            .collect()
    }

    /// The pose graph a loop closure optimizes (`getPoseGraphEdges`): the tree,
    /// the loop closures, and the high-covisibility edges that are not tree
    /// edges, in that order and each sorted.
    pub fn pose_graph_edges(&self, threshold: u32) -> Vec<PoseGraphEdge> {
        let mut edges: Vec<PoseGraphEdge> = self
            .nodes
            .iter()
            .filter_map(|(&to, node)| {
                node.parent.map(|from| PoseGraphEdge {
                    from,
                    to,
                    kind: PoseGraphEdgeKind::SpanningTree,
                })
            })
            .collect();
        for (&a, node) in &self.nodes {
            for &b in node.loops.range(a + 1..) {
                edges.push(PoseGraphEdge {
                    from: a,
                    to: b,
                    kind: PoseGraphEdgeKind::LoopClosure,
                });
            }
        }
        for (a, b) in self.high_covisibility_edges(threshold) {
            if self.parent(a) != Some(b) && self.parent(b) != Some(a) {
                edges.push(PoseGraphEdge {
                    from: a,
                    to: b,
                    kind: PoseGraphEdgeKind::HighCovisibility,
                });
            }
        }
        edges
    }

    /// How replaceable `id` is (`computeGraphScore`): 1 with no high edge, 0 with
    /// ten or more.
    pub fn graph_score(&self, id: FrameId, threshold: u32) -> f32 {
        let high: usize = self
            .neighbours(id)
            .filter(|(_, weight)| *weight >= threshold)
            .count();
        1.0 - (high as f32 / 10.0).min(1.0)
    }

    /// How much `id` holds the map together through loops: `n / (n + 1)`.
    pub fn loop_score(&self, id: FrameId) -> f32 {
        let loops: f32 = self.loop_closures(id).count() as f32;
        loops / (loops + 1.0)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use proptest::prelude::*;

    fn counts(pairs: &[(FrameId, u32)]) -> BTreeMap<FrameId, u32> {
        pairs.iter().copied().collect()
    }

    /// 1 root; 2 shares 50 with 1; 3 shares 10 with 1 and 60 with 2; 4 shares
    /// nothing; 5 shares 80 with 3 and 80 with 4 (a tie).
    fn hand_graph() -> CovisibilityGraph {
        let mut graph = CovisibilityGraph::new();
        assert!(graph.insert_keyframe(1, &counts(&[])));
        assert!(graph.insert_keyframe(2, &counts(&[(1, 50)])));
        assert!(graph.insert_keyframe(3, &counts(&[(1, 10), (2, 60)])));
        assert!(graph.insert_keyframe(4, &counts(&[(99, 5)])));
        assert!(graph.insert_keyframe(5, &counts(&[(3, 80), (4, 80)])));
        graph
    }

    #[test]
    fn a_keyframe_joins_under_its_strongest_neighbour() {
        let graph = hand_graph();
        assert_eq!(graph.root(), Some(1));
        assert_eq!(graph.parent(1), None);
        assert_eq!(graph.parent(2), Some(1));
        assert_eq!(graph.parent(3), Some(2));
        assert_eq!(graph.parent(5), Some(3), "a tie goes to the smaller id");
        assert_eq!(graph.weight(3, 2), Some(60));
        assert_eq!(graph.weight(2, 3), Some(60));
        assert_eq!(graph.weight(1, 3), Some(10));
        assert_eq!(graph.children(1).collect::<Vec<_>>(), vec![2]);
        assert_eq!(graph.children(3).collect::<Vec<_>>(), vec![4, 5]);
    }

    #[test]
    fn a_keyframe_sharing_nothing_hangs_from_the_previous_one_at_weight_zero() {
        let graph = hand_graph();
        assert_eq!(graph.parent(4), Some(3));
        assert_eq!(graph.weight(4, 3), Some(0));
        assert_eq!(graph.weight(4, 99), None, "a missing node gets no edge");
        assert!(!graph.contains(99));
    }

    #[test]
    fn inserting_twice_changes_nothing() {
        let mut graph = hand_graph();
        let before = graph.clone();
        assert!(!graph.insert_keyframe(3, &counts(&[(1, 500)])));
        assert_eq!(graph, before);
    }

    #[test]
    fn removing_a_keyframe_moves_its_children_to_its_parent() {
        let mut graph = hand_graph();
        assert!(graph.add_loop_closure(5, 1));
        assert!(graph.remove_keyframe(3));
        assert!(!graph.contains(3));
        assert_eq!(graph.parent(5), Some(2));
        assert_eq!(graph.parent(4), Some(2));
        assert_eq!(graph.children(2).collect::<Vec<_>>(), vec![4, 5]);
        assert_eq!(graph.weight(2, 3), None);
        assert_eq!(graph.neighbours(5).collect::<Vec<_>>(), vec![(4, 80)]);
        assert_eq!(graph.loop_closures(1).collect::<Vec<_>>(), vec![5]);
        assert!(!graph.remove_keyframe(1), "the root stays");
        assert!(!graph.remove_keyframe(3), "already gone");
    }

    #[test]
    fn decrementing_lowers_then_drops_an_edge_and_never_creates_nodes() {
        let mut graph = hand_graph();
        graph.decrement_edge(2, 3, 20);
        assert_eq!(graph.weight(3, 2), Some(40));
        graph.decrement_edge(3, 2, 100);
        assert_eq!(graph.weight(2, 3), None);
        assert!(graph.contains(3), "a node outlives its edges");
        let before = graph.clone();
        graph.decrement_edge(42, 43, 1);
        graph.decrement_edge(1, 42, 1);
        graph.decrement_edge(1, 5, 1);
        assert_eq!(graph, before, "B19: a missing edge or node is left alone");
    }

    #[test]
    fn the_pose_graph_is_the_tree_the_loops_and_the_strong_non_tree_edges() {
        let mut graph = hand_graph();
        assert!(graph.add_loop_closure(5, 1));
        assert!(!graph.add_loop_closure(5, 5));
        assert!(!graph.add_loop_closure(5, 77));
        let edges = graph.pose_graph_edges(60);
        let tree = |from, to| PoseGraphEdge {
            from,
            to,
            kind: PoseGraphEdgeKind::SpanningTree,
        };
        assert_eq!(
            edges,
            vec![
                tree(1, 2),
                tree(2, 3),
                tree(3, 4),
                tree(3, 5),
                PoseGraphEdge {
                    from: 1,
                    to: 5,
                    kind: PoseGraphEdgeKind::LoopClosure
                },
                PoseGraphEdge {
                    from: 4,
                    to: 5,
                    kind: PoseGraphEdgeKind::HighCovisibility
                },
            ]
        );
        assert_eq!(
            graph.high_covisibility_edges(60),
            vec![(2, 3), (3, 5), (4, 5)]
        );
    }

    #[test]
    fn scores_follow_strong_edges_and_loops() {
        let mut graph = hand_graph();
        assert_eq!(graph.graph_score(3, 60), 0.8);
        assert_eq!(graph.graph_score(3, 1000), 1.0);
        assert_eq!(graph.loop_score(5), 0.0);
        assert!(graph.add_loop_closure(5, 1));
        assert!(graph.add_loop_closure(5, 2));
        assert!((graph.loop_score(5) - 2.0 / 3.0).abs() < 1e-6);
    }

    /// One operation of the property test below.
    #[derive(Debug, Clone)]
    enum Op {
        Insert(Vec<(u8, u8)>),
        Remove(u8),
        Decrement(u8, u8, u8),
        Loop(u8, u8),
    }

    fn op() -> impl Strategy<Value = Op> {
        prop_oneof![
            3 => prop::collection::vec((0u8..40, 0u8..5), 0..6).prop_map(Op::Insert),
            2 => (0u8..40).prop_map(Op::Remove),
            2 => (0u8..40, 0u8..40, 0u8..4).prop_map(|(a, b, by)| Op::Decrement(a, b, by)),
            1 => (0u8..40, 0u8..40).prop_map(|(a, b)| Op::Loop(a, b)),
        ]
    }

    proptest! {
        /// Whatever the order of inserts, removals, decrements and loops: edges
        /// are symmetric and join nodes, the tree is one component rooted at the
        /// first keyframe, and parent and child links agree.
        #[test]
        fn the_graph_stays_one_consistent_tree(ops in prop::collection::vec(op(), 1..80)) {
            let mut graph = CovisibilityGraph::new();
            let mut next: FrameId = 0;
            for op in ops {
                match op {
                    Op::Insert(shared) => {
                        let shared: BTreeMap<FrameId, u32> = shared
                            .into_iter()
                            .map(|(other, weight)| (FrameId::from(other), u32::from(weight)))
                            .collect();
                        prop_assert!(graph.insert_keyframe(next, &shared));
                        next += 1;
                    }
                    Op::Remove(id) => {
                        let id = FrameId::from(id);
                        let expected = graph.contains(id) && graph.root() != Some(id);
                        prop_assert_eq!(graph.remove_keyframe(id), expected);
                        prop_assert_eq!(graph.contains(id), graph.root() == Some(id));
                    }
                    Op::Decrement(a, b, by) => {
                        graph.decrement_edge(a.into(), b.into(), by.into());
                    }
                    Op::Loop(a, b) => {
                        graph.add_loop_closure(a.into(), b.into());
                    }
                }
                let root = graph.root();
                prop_assert_eq!(root.is_some(), next > 0);
                for id in graph.node_ids() {
                    for (other, weight) in graph.neighbours(id) {
                        prop_assert!(graph.contains(other));
                        prop_assert_eq!(graph.weight(other, id), Some(weight));
                    }
                    for other in graph.loop_closures(id) {
                        prop_assert!(graph.loop_closures(other).any(|back| back == id));
                    }
                    for child in graph.children(id) {
                        prop_assert_eq!(graph.parent(child), Some(id));
                    }
                    // Walking up reaches the root in fewer steps than there are nodes.
                    let mut at = id;
                    let mut steps = 0;
                    while let Some(parent) = graph.parent(at) {
                        prop_assert!(graph.contains(parent));
                        prop_assert!(graph.children(parent).any(|child| child == at));
                        at = parent;
                        steps += 1;
                        prop_assert!(steps <= graph.len());
                    }
                    prop_assert_eq!(Some(at), root);
                }
            }
        }
    }
}
