//! Which blocks' meshes changed since a consumer last took them.

use rustc_hash::{FxHashMap, FxHashSet};

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
    /// Content hash ([`Mesh::content_hash`]) of each block's last delivered
    /// mesh, for the blocks whose last delivered mesh had triangles.
    pub(crate) delivered: FxHashMap<BlockKey, u64>,
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
        self.delivered.remove(key);
    }

    /// Pair `meshes`, each with its content hash, with the [`Delivery`] that
    /// records them in this tracker.
    pub(crate) fn pending(&mut self, meshes: Vec<(BlockKey, Mesh, u64)>) -> DirtyMeshes<'_> {
        let (meshes, delivered) = meshes
            .into_iter()
            .map(|(key, mesh, hash)| {
                let surface_hash = (!mesh.triangles.is_empty()).then_some(hash);
                ((key, mesh), (key, surface_hash))
            })
            .unzip();
        DirtyMeshes {
            meshes,
            delivery: Delivery {
                tracker: self,
                delivered,
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
    /// Each delivered block and its mesh's content hash, `None` if the mesh
    /// has no triangles.
    delivered: Vec<(BlockKey, Option<u64>)>,
}

impl Delivery<'_> {
    /// Remember the delivered meshes and clear the dirty set.
    pub fn confirm(self) {
        for (key, surface_hash) in self.delivered {
            match surface_hash {
                Some(hash) => self.tracker.delivered.insert(key, hash),
                None => self.tracker.delivered.remove(&key),
            };
        }
        self.tracker.dirty.clear();
    }
}
