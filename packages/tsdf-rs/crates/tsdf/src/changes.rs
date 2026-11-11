//! Which blocks' meshes changed since a consumer last took them.

use rustc_hash::FxHashSet;

use crate::mesh::Mesh;
use crate::volume::BlockKey;

/// Offsets of a block and its 26 neighbors.
const NEIGHBORHOOD: [[i64; 3]; 27] = {
    let mut offsets = [[0; 3]; 27];
    let mut index = 0;
    while index < 27 {
        let i = index as i64;
        offsets[index] = [i % 3 - 1, i / 3 % 3 - 1, i / 9 - 1];
        index += 1;
    }
    offsets
};

/// The volume's record of meshes not yet delivered.
#[derive(Debug, Default)]
pub(crate) struct ChangeTracker {
    /// Blocks whose mesh may have changed since the last delivery.
    dirty: FxHashSet<BlockKey>,
    /// Blocks whose last delivered mesh had triangles.
    pub(crate) surfaced: FxHashSet<BlockKey>,
}

impl ChangeTracker {
    /// Mark the meshes that updating `touched` blocks can change.
    ///
    /// A block's cells read its {0,1}^3 neighborhood and their normals one
    /// voxel further on both sides, so updating a block stales the meshes of
    /// itself and all 26 neighbors.
    pub(crate) fn mark_touched<'a>(&mut self, touched: impl IntoIterator<Item = &'a BlockKey>) {
        for key in touched {
            for offset in NEIGHBORHOOD {
                let neighbor = [
                    key[0].checked_add(offset[0]),
                    key[1].checked_add(offset[1]),
                    key[2].checked_add(offset[2]),
                ];
                if let [Some(x), Some(y), Some(z)] = neighbor {
                    self.dirty.insert([x, y, z]);
                }
            }
        }
    }

    /// The dirty blocks that `exists`, sorted.
    pub(crate) fn dirty_keys(&self, exists: impl Fn(&BlockKey) -> bool) -> Vec<BlockKey> {
        let mut keys: Vec<BlockKey> = self.dirty.iter().copied().filter(exists).collect();
        keys.sort_unstable();
        keys
    }

    /// Drop every record of `key`, a block the volume freed.
    pub(crate) fn forget(&mut self, key: &BlockKey) {
        self.dirty.remove(key);
        self.surfaced.remove(key);
    }

    /// Pair `meshes` with the [`Delivery`] that records them in this tracker.
    pub(crate) fn pending(&mut self, meshes: Vec<(BlockKey, Mesh)>) -> DirtyMeshes<'_> {
        let surfaced = meshes
            .iter()
            .map(|(key, mesh)| (*key, !mesh.triangles.is_empty()))
            .collect();
        DirtyMeshes {
            meshes,
            delivery: Delivery {
                tracker: self,
                surfaced,
            },
        }
    }
}

/// Per-block meshes from [`crate::TsdfVolume::dirty_block_meshes`] and the
/// receipt that marks them delivered.
#[derive(Debug)]
pub struct DirtyMeshes<'a> {
    /// `(block_key, mesh)` pairs in sorted key order.
    pub meshes: Vec<(BlockKey, Mesh)>,
    /// Confirm once the meshes reached their consumer.
    pub delivery: Delivery<'a>,
}

/// Records a [`DirtyMeshes`] extraction as delivered to its volume.
///
/// It borrows the volume mutably, so no frame can be integrated and no other
/// volume's meshes can be confirmed in between. Dropping it unconfirmed
/// keeps the meshes pending for the next extraction.
#[derive(Debug)]
#[must_use = "unconfirmed meshes stay pending; call `confirm` once delivered"]
pub struct Delivery<'a> {
    tracker: &'a mut ChangeTracker,
    /// Each delivered block and whether its mesh has triangles.
    surfaced: Vec<(BlockKey, bool)>,
}

impl Delivery<'_> {
    /// Remember which delivered blocks have surface and clear the dirty set.
    pub fn confirm(self) {
        for (key, has_surface) in self.surfaced {
            if has_surface {
                self.tracker.surfaced.insert(key);
            } else {
                self.tracker.surfaced.remove(&key);
            }
        }
        self.tracker.dirty.clear();
    }
}
