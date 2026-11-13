//! Which blocks' meshes changed since a consumer last took them.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::mesh::Mesh;
use crate::volume::{checked_offset, BlockKey, LocalBox, BLOCK_RESOLUTION_USIZE};

/// Offsets of a block and its 26 neighbors, x fastest.
pub(crate) const NEIGHBORHOOD: [[i64; 3]; 27] = {
    let mut offsets = [[0; 3]; 27];
    let mut index = 0;
    while index < 27 {
        let i = index as i64;
        offsets[index] = [i % 3 - 1, i / 3 % 3 - 1, i / 9 - 1];
        index += 1;
    }
    offsets
};

/// The blocks whose meshes may have changed since the last confirmed
/// [`Delivery`], and what that delivery held.
#[derive(Debug, Default)]
pub(crate) struct ChangeTracker {
    /// Blocks whose mesh may have changed since the last confirmed delivery.
    dirty: FxHashSet<BlockKey>,
    /// Content hash ([`Mesh::content_hash`]) of each block's last delivered
    /// mesh, for the blocks whose last delivered mesh had triangles.
    surface_hashes: FxHashMap<BlockKey, u64>,
}

/// A dirty block's new mesh, not yet confirmed as delivered.
pub(crate) struct Pending {
    pub(crate) key: BlockKey,
    pub(crate) mesh: Mesh,
    /// The mesh's content hash, `None` if it has no triangles.
    pub(crate) surface_hash: Option<u64>,
}

impl ChangeTracker {
    /// Content hash of the last delivered mesh of `key`, if it had triangles.
    pub(crate) fn last_surface_hash(&self, key: &BlockKey) -> Option<u64> {
        self.surface_hashes.get(key).copied()
    }

    /// Mark the meshes that removing or replacing `touched` blocks can change:
    /// their own and all 26 neighbors' (see [`ChangeTracker::mark_changed`]).
    pub(crate) fn mark_touched<'a>(&mut self, touched: impl IntoIterator<Item = &'a BlockKey>) {
        for key in touched {
            for offset in NEIGHBORHOOD {
                if let Some(neighbor) = checked_offset(*key, offset) {
                    self.dirty.insert(neighbor);
                }
            }
        }
    }

    /// Mark the meshes that a change to the `changed` voxels of block `key`
    /// can alter.
    ///
    /// The mesh of block `k` reads global voxels `16 * k - 1` to
    /// `16 * k + 17` on each axis (cell corners up to `16 * k + 16`, their
    /// normals one voxel further), so a change at local coordinate `c` of
    /// `key` reaches the previous block on that axis only if `c <= 1`, and
    /// the next one only if `c >= 15`.
    pub(crate) fn mark_changed(&mut self, key: BlockKey, changed: &LocalBox) {
        let last = BLOCK_RESOLUTION_USIZE - 1;
        for offset in NEIGHBORHOOD {
            let reaches = (0..3).all(|axis| match offset[axis] {
                -1 => changed.min[axis] <= 1,
                1 => changed.max[axis] >= last,
                _ => true,
            });
            if let Some(neighbor) = checked_offset(key, offset).filter(|_| reaches) {
                self.dirty.insert(neighbor);
            }
        }
    }

    /// Mark the meshes of `keys` themselves, as after a change that affects
    /// every block alike.
    pub(crate) fn mark_all<'a>(&mut self, keys: impl IntoIterator<Item = &'a BlockKey>) {
        self.dirty.extend(keys);
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
        self.surface_hashes.remove(key);
    }

    /// The `pending` meshes with the [`Delivery`] that records them in this
    /// tracker.
    pub(crate) fn pending(&mut self, pending: Vec<Pending>) -> DirtyMeshes<'_> {
        let (meshes, receipts) = pending
            .into_iter()
            .map(|pending| {
                (
                    (pending.key, pending.mesh),
                    (pending.key, pending.surface_hash),
                )
            })
            .unzip();
        DirtyMeshes {
            meshes,
            delivery: Delivery {
                tracker: self,
                receipts,
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
/// It borrows the volume mutably: until it is confirmed or dropped, this
/// volume can neither integrate a frame nor extract again, so what it
/// records matches the volume's state. Dropping it unconfirmed keeps the
/// meshes pending for the next extraction.
#[derive(Debug)]
#[must_use = "unconfirmed meshes stay pending; call `confirm` once delivered"]
pub struct Delivery<'a> {
    tracker: &'a mut ChangeTracker,
    /// Each extracted block and its mesh's content hash, `None` if the mesh
    /// has no triangles.
    receipts: Vec<(BlockKey, Option<u64>)>,
}

impl Delivery<'_> {
    /// Remember the delivered meshes and clear the dirty set.
    pub fn confirm(self) {
        for (key, surface_hash) in self.receipts {
            match surface_hash {
                Some(hash) => self.tracker.surface_hashes.insert(key, hash),
                None => self.tracker.surface_hashes.remove(&key),
            };
        }
        self.tracker.dirty.clear();
    }
}
