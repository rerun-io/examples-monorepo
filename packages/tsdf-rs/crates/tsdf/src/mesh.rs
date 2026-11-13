/// Triangle mesh of a TSDF zero level set.
///
/// Positions use world units. Triangle rows hold zero-based indices into
/// `vertices` and wind counter-clockwise seen from outside, the side the
/// cameras observed. Per-vertex rows of `colors` and `normals` follow
/// `vertices`.
///
/// # Example
///
/// ```
/// use tsdf::TsdfVolume;
///
/// let mesh = TsdfVolume::new(0.04, 0.12, false)?.extract_mesh()?;
/// assert!(mesh.vertices.is_empty() && mesh.normals.is_empty());
/// assert_eq!(mesh.colors, None);
/// # Ok::<(), tsdf::TsdfError>(())
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    /// World-space vertex positions.
    pub vertices: Vec<[f32; 3]>,
    /// Triangle vertex indices, counter-clockwise seen from outside.
    pub triangles: Vec<[u32; 3]>,
    /// Per-vertex RGB of the nearest voxel, for a color-fusing volume.
    pub colors: Option<Vec<[u8; 3]>>,
    /// Per-vertex unit normals: the TSDF gradient, pointing out of the
    /// surface toward the observing cameras.
    pub normals: Vec<[f32; 3]>,
}

impl Mesh {
    /// 64-bit hash of the vertex, triangle, color and normal bytes: equal
    /// for byte-identical meshes, and different otherwise up to a hash
    /// collision. The same on every target, wasm32 included.
    pub(crate) fn content_hash(&self) -> u64 {
        let mut hasher = WordHasher::default();
        hasher.write_word(self.vertices.len() as u64);
        hasher.write_word(self.triangles.len() as u64);
        hasher.write_floats(self.vertices.as_flattened());
        hasher.write_floats(self.normals.as_flattened());
        hasher.write_u32s(self.triangles.as_flattened());
        if let Some(colors) = &self.colors {
            hasher.write_bytes(colors.as_flattened());
        }
        hasher.0
    }
}

/// FxHash's rotate-xor-multiply step on 64-bit words. `rustc_hash::FxHasher`
/// hashes `usize` words, so its hashes are 32-bit on wasm32.
#[derive(Default)]
struct WordHasher(u64);

impl WordHasher {
    fn write_word(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x517c_c1b7_2722_0a95);
    }

    /// Two values per word.
    fn write_u32s(&mut self, values: &[u32]) {
        let mut pairs = values.chunks_exact(2);
        for pair in &mut pairs {
            self.write_word(u64::from(pair[0]) | (u64::from(pair[1]) << 32));
        }
        if let [last] = pairs.remainder() {
            self.write_word(u64::from(*last));
        }
    }

    /// The bit patterns of `values`, two per word.
    fn write_floats(&mut self, values: &[f32]) {
        let mut pairs = values.chunks_exact(2);
        for pair in &mut pairs {
            self.write_word(u64::from(pair[0].to_bits()) | (u64::from(pair[1].to_bits()) << 32));
        }
        if let [last] = pairs.remainder() {
            self.write_word(u64::from(last.to_bits()));
        }
    }

    /// Eight bytes per word, the last zero-padded.
    fn write_bytes(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.write_word(u64::from_le_bytes(word));
        }
    }
}

/// Fraction of the way from sample `a` to sample `b` where the linear
/// interpolant crosses zero.
pub(crate) fn interpolation_amount(value_a: f32, value_b: f32) -> f32 {
    if value_a == 0.0 {
        0.0
    } else if value_b == 0.0 {
        1.0
    } else {
        value_a / (value_a - value_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle() -> Mesh {
        Mesh {
            vertices: vec![[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
            triangles: vec![[0, 2, 1]],
            colors: Some(vec![[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
            normals: vec![[0.0, 0.0, -1.0]; 3],
        }
    }

    #[test]
    fn the_content_hash_is_pinned_and_sees_every_bit() {
        // A fixed value: the hash must not depend on the target's word size.
        assert_eq!(triangle().content_hash(), 4_450_279_870_364_916_991);
        let mut flipped = triangle();
        flipped.normals[2][2] = (-1.0_f32).next_up();
        assert_ne!(flipped.content_hash(), triangle().content_hash());
        let mut recolored = triangle();
        recolored.colors = Some(vec![[255, 0, 0], [0, 255, 0], [0, 0, 254]]);
        assert_ne!(recolored.content_hash(), triangle().content_hash());
    }
}
