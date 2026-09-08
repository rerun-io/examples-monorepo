//! `tbb::parallel_deterministic_reduce`'s association, reproduced exactly.
//!
//! basalt reduces at four sites in `src/linearization/linearization_abs_qr.cpp`
//! — `:262` (the linearization error), `:307` (the model cost change), `:354`
//! (the Jacobian column norms) and `:550` (the dense `H` and `b`) — and every
//! one of them builds `tbb::blocked_range<size_t>(0, n)` with the two-argument
//! constructor, so **grainsize 1**.
//!
//! `parallel_deterministic_reduce` uses `simple_partitioner`, which keeps
//! splitting while the range reports `is_divisible()` (`grainsize < size()`),
//! and `blocked_range`'s splitting constructor cuts at
//! `begin + (end - begin) / 2`, giving the original the left half and the new
//! range the right. Bodies join as `left.join(right)`. The result is a balanced
//! binary tree whose shape depends only on `n` — that is exactly what makes the
//! algorithm deterministic, and it is **not** a left fold:
//!
//! ```text
//! n = 4:  (x0 + x1) + (x2 + x3)
//! n = 5:  (x0 + x1) + (x2 + (x3 + x4))
//! ```
//!
//! In `f32` with `[2²⁴, 1, 1, 1]` the tree returns `16777218` and a left fold
//! `16777216`, because `2²⁴ + 1` is not representable. Decision D31 said these
//! four sites needed a fixed order; it did not say *which*, and a left fold is
//! the wrong one. The rule here is pinned bit for bit against the fork's own
//! TBB by `tests/fixtures/linearize/tbb_reduce_oracle.json`
//! (`tools/tbb_reduce_probe.cpp` on `slam-rs-reference`): 120 cases, 90 of them
//! inputs where the tree and a left fold disagree, and the probe records that
//! the answer does not move between one thread and the machine's default.
//!
//! A `rayon` version has to reproduce the same tree — `par_chunks` with a
//! sequential merge does not, and neither does `reduce` over unspecified
//! splits. The shape here (recurse left into the caller's accumulator, right
//! into a scratch buffer one level down, then join) is what a `join`-based
//! parallel recursion would have anyway.

use crate::lie::LieScalar;

/// One leaf of the tree: `body(blocked_range(i, i + 1), acc)`.
///
/// C++'s body loops over the whole subrange; with grainsize 1 a subrange is one
/// index, so the loop body *is* the leaf. It takes the accumulator because
/// basalt's `backSubstitute` mutates it in place (`:302`).
pub(crate) type LeafResult<E> = Result<(), E>;

/// What [`deterministic_reduce`] can build a tree over.
///
/// C++'s reduction body carries the three operations TBB needs: the split
/// constructor that starts a right-hand subtree from the identity, the identity
/// itself, and `join` (`linearization_abs_qr.cpp:513-542`). Naming them on the
/// accumulator instead of passing three closures per call is what keeps the
/// recursion below to one shape for both accumulator types — and the shape is
/// the contract the `tbb_reduce_oracle` fixture pins.
pub(crate) trait Reducible {
    /// A fresh identity of this accumulator's own shape, for the scratch buffer
    /// one recursion level down.
    fn identity_like(&self) -> Self;
    /// Back to the identity, so one scratch buffer serves every subtree at its
    /// depth.
    fn reset(&mut self);
    /// `my_value = my_reduction(my_value, rhs.my_value)`: left, then right.
    fn join(&mut self, right: &Self);
}

/// Reduce `0..n` in `tbb::parallel_deterministic_reduce`'s order, accumulating
/// through `&mut T` so nothing is cloned and the matrix sites can reuse one
/// buffer per recursion level.
///
/// `out` must already hold the identity, and `leaf` folds one index into an
/// accumulator; the identity, the reset and the join come off [`Reducible`], in
/// that order, as `lambda_reduce_body::join` does.
///
/// Scratch buffers are kept between calls in `scratch`, one per recursion
/// depth, so a reduction over `n` blocks allocates `ceil(log2 n)` accumulators
/// rather than `n` of them. C++ allocates one full `total_size²` matrix per
/// task (`linearization_abs_qr.cpp:527-535`), which the architecture dossier
/// already flags as wasteful; the association is what has to match, not the
/// allocation count.
pub(crate) fn deterministic_reduce<T: Reducible, E>(
    n: usize,
    out: &mut T,
    scratch: &mut Vec<Option<T>>,
    leaf: &mut dyn FnMut(usize, &mut T) -> LeafResult<E>,
) -> LeafResult<E> {
    if n == 0 {
        // An empty `blocked_range` never runs a body, so the identity survives.
        return Ok(());
    }
    reduce_range(0, n, 0, out, scratch, leaf)
}

fn reduce_range<T: Reducible, E>(
    lo: usize,
    hi: usize,
    depth: usize,
    out: &mut T,
    scratch: &mut Vec<Option<T>>,
    leaf: &mut dyn FnMut(usize, &mut T) -> LeafResult<E>,
) -> LeafResult<E> {
    // `blocked_range::is_divisible()` is `grainsize < size()`, and the
    // grainsize at all four of basalt's sites is the default 1.
    if hi - lo == 1 {
        return leaf(lo, out);
    }

    // `blocked_range::do_split`: `middle = begin + (end - begin) / 2`, the
    // original keeps `[begin, middle)` and the new range takes `[middle, end)`.
    let mid: usize = lo + (hi - lo) / 2;

    // The left subtree accumulates into the caller's buffer, which already
    // holds the identity — that is C++'s original body.
    reduce_range(lo, mid, depth + 1, out, scratch, leaf)?;

    // The right subtree is C++'s split-constructed body, which starts from the
    // identity. Take the buffer for this depth so the recursion below can use
    // the deeper ones.
    while scratch.len() <= depth {
        scratch.push(None);
    }
    let mut buffer: T = match scratch.get_mut(depth).and_then(Option::take) {
        Some(buffer) => buffer,
        None => out.identity_like(),
    };
    buffer.reset();
    let result: LeafResult<E> = reduce_range(mid, hi, depth + 1, &mut buffer, scratch, leaf);
    if result.is_ok() {
        out.join(&buffer);
    }
    if let Some(slot) = scratch.get_mut(depth) {
        *slot = Some(buffer);
    }
    result
}

/// The same tree over plain scalars, which is what two of the four sites need.
///
/// `leaf(i, acc)` returns the accumulator after folding index `i` into it, so a
/// site that subtracts (`backSubstitute` does, `:302`) reads naturally.
pub(crate) fn deterministic_reduce_scalar<S: LieScalar, E>(
    n: usize,
    leaf: &mut dyn FnMut(usize, S) -> Result<S, E>,
) -> Result<S, E> {
    let mut out: Sum<S> = Sum(S::zero());
    let mut scratch: Vec<Option<Sum<S>>> = Vec::new();
    deterministic_reduce(
        n,
        &mut out,
        &mut scratch,
        &mut |index: usize, acc: &mut Sum<S>| {
            acc.0 = leaf(index, acc.0)?;
            Ok(())
        },
    )?;
    Ok(out.0)
}

/// A scalar accumulator. [`Reducible`] sits on this rather than on `LieScalar`,
/// so the two numeric widths keep no knowledge of the reduction tree.
struct Sum<S: LieScalar>(S);

impl<S: LieScalar> Reducible for Sum<S> {
    fn identity_like(&self) -> Self {
        Self(S::zero())
    }
    fn reset(&mut self) {
        self.0 = S::zero();
    }
    fn join(&mut self, right: &Self) {
        self.0 += right.0;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    impl Reducible for String {
        fn identity_like(&self) -> Self {
            Self::new()
        }
        fn reset(&mut self) {
            self.clear();
        }
        fn join(&mut self, right: &Self) {
            *self = format!("({self}+{right})");
        }
    }

    /// The tree of `n = 4` is `(x0 + x1) + (x2 + x3)`, and of `n = 5`
    /// `(x0 + x1) + (x2 + (x3 + x4))`. Checked structurally, by recording the
    /// order the joins happen in rather than by comparing a sum.
    #[test]
    fn the_split_rule_is_tbbs() {
        for (n, want) in [
            (1usize, "0"),
            (2, "(0+1)"),
            (3, "(0+(1+2))"),
            (4, "((0+1)+(2+3))"),
            (5, "((0+1)+(2+(3+4)))"),
            (6, "((0+(1+2))+(3+(4+5)))"),
            (7, "((0+(1+2))+((3+4)+(5+6)))"),
            (8, "(((0+1)+(2+3))+((4+5)+(6+7)))"),
        ] {
            let mut out: String = String::new();
            let mut scratch: Vec<Option<String>> = Vec::new();
            deterministic_reduce::<String, ()>(
                n,
                &mut out,
                &mut scratch,
                &mut |index: usize, acc: &mut String| {
                    assert!(acc.is_empty(), "a leaf ran on a non-identity accumulator");
                    *acc = index.to_string();
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(out, want, "n = {n}");
        }
    }

    /// An empty range never runs a body, so the identity survives (`:1` of
    /// `blocked_range`'s contract, and what C++ returns for a window with no
    /// landmarks).
    #[test]
    fn an_empty_range_is_the_identity() {
        let sum: f64 =
            deterministic_reduce_scalar::<f64, ()>(0, &mut |_, acc| Ok(acc + 1.0)).unwrap();
        assert_eq!(sum, 0.0);
    }

    /// A leaf that fails stops the reduction and returns its error, and every
    /// scratch buffer goes back where it came from.
    #[test]
    fn a_failing_leaf_propagates() {
        let result: Result<f64, &'static str> =
            deterministic_reduce_scalar::<f64, &'static str>(8, &mut |index, acc| {
                if index == 5 { Err("no") } else { Ok(acc + 1.0) }
            });
        assert_eq!(result, Err("no"));
    }
}
