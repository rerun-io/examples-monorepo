//! Explicit frontend parallelism (D31).
//! Use fixed chunks, sequential accumulation within chunks and a configured
//! thread count instead of the ambient Rayon pool. Each loop is a pure function
//! of its index; tests require identical output at one and four threads.

use std::sync::Arc;

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuildError, ThreadPoolBuilder};

/// The most workers a pool may be built for.
///
/// `ThreadPoolBuilder::num_threads` takes the request literally: rayon spawns OS
/// threads until the kernel refuses, so a `threads` of 100,000 arriving over the
/// Python boundary wedged the machine for minutes instead of returning an error.
/// A thousand is already more hardware threads than any host this port runs on,
/// and about 8 GB of thread stacks.
pub const MAX_THREADS: usize = 1024;

/// Chunks per worker in [`WorkPool::for_each_warp`].
const CHUNKS_PER_WORKER: usize = 8;
/// The fewest warps one chunk carries.
const MIN_CHUNK: usize = 8;

/// A fixed-width worker pool, or the sequential path when one thread was asked for.
///
/// Cloning shares the workers: the tracker and the patch stores it makes run on
/// the same pool.
#[derive(Debug, Clone)]
pub struct WorkPool {
    threads: usize,
    /// `None` at `threads == 1`: the sequential path runs on the caller's thread
    /// and never touches rayon at all.
    pool: Option<Arc<ThreadPool>>,
}

impl WorkPool {
    /// A pool of exactly `threads` workers; `threads == 0` is read as one.
    ///
    /// The frontend refuses zero before it gets here
    /// ([`FrontendError::NoThreads`](crate::frontend::flow::FrontendError::NoThreads)),
    /// so the clamp is the second line under that rule rather than the rule.
    ///
    /// # Errors
    ///
    /// [`ThreadPoolBuildError`] when the operating system refuses the threads.
    pub fn new(threads: usize) -> Result<Self, ThreadPoolBuildError> {
        let threads: usize = threads.max(1);
        let pool: Option<Arc<ThreadPool>> = if threads == 1 {
            None
        } else {
            Some(Arc::new(
                ThreadPoolBuilder::new().num_threads(threads).build()?,
            ))
        };
        Ok(Self { threads, pool })
    }

    /// Workers this pool runs on.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Run `f` with these workers as rayon's current pool, so the parallel
    /// iterators inside it use exactly them; `None`, without running `f`, on
    /// the sequential path, where the caller runs its own loop instead.
    pub fn install<R: Send>(&self, f: impl FnOnce() -> R + Send) -> Option<R> {
        self.pool.as_ref().map(|pool| pool.install(f))
    }

    /// Run a fallible, independent operation on each indexed slice entry.
    /// The one-worker path never enters Rayon.
    pub(crate) fn try_for_each_mut<T: Send, E: Send>(
        &self,
        items: &mut [T],
        body: impl Fn(usize, &mut T) -> Result<(), E> + Sync + Send,
    ) -> Result<(), E> {
        if let Some(pool) = &self.pool {
            pool.install(|| {
                items
                    .par_iter_mut()
                    .enumerate()
                    .try_for_each(|(index, item)| body(index, item))
            })
        } else {
            items
                .iter_mut()
                .enumerate()
                .try_for_each(|(index, item)| body(index, item))
        }
    }

    /// Apply `body` to every index of the shortest input, writing the warp it
    /// returns into six flat coefficient arrays and its flag into `valid`.
    ///
    /// `body` is a pure function of the index — it owns no mutable state — which
    /// is what makes the split irrelevant to the result and what lets the six
    /// arrays stay structure-of-arrays with the patch index fast-varying
    /// (`cubecl-portability.md` §12.2). The seven disjoint borrows are written
    /// out here because a closure cannot produce them and rayon requires them.
    pub fn for_each_warp(
        &self,
        coefficients: [&mut [f32]; 6],
        valid: &mut [bool],
        body: impl Fn(usize) -> ([f32; 6], bool) + Sync + Send,
    ) {
        let [m00, m01, m10, m11, tx, ty] = coefficients;
        let len: usize = [
            m00.len(),
            m01.len(),
            m10.len(),
            m11.len(),
            tx.len(),
            ty.len(),
        ]
        .into_iter()
        .chain(std::iter::once(valid.len()))
        .min()
        .unwrap_or(0);

        let Some(pool) = self.pool.as_ref() else {
            for index in 0..len {
                let (warp, flag) = body(index);
                m00[index] = warp[0];
                m01[index] = warp[1];
                m10[index] = warp[2];
                m11[index] = warp[3];
                tx[index] = warp[4];
                ty[index] = warp[5];
                valid[index] = flag;
            }
            return;
        };

        // A fixed split for a given (len, threads): eight chunks per worker of
        // at least eight warps, so a worker that drew the expensive points does
        // not hold the others up. Which worker runs a chunk is rayon's choice
        // and cannot matter: every index is a pure function of itself.
        let chunk: usize = len
            .div_ceil(self.threads * CHUNKS_PER_WORKER)
            .max(MIN_CHUNK);
        pool.install(|| {
            m00[..len]
                .par_chunks_mut(chunk)
                .zip(m01[..len].par_chunks_mut(chunk))
                .zip(m10[..len].par_chunks_mut(chunk))
                .zip(m11[..len].par_chunks_mut(chunk))
                .zip(tx[..len].par_chunks_mut(chunk))
                .zip(ty[..len].par_chunks_mut(chunk))
                .zip(valid[..len].par_chunks_mut(chunk))
                .enumerate()
                .for_each(|(block, ((((((m00, m01), m10), m11), tx), ty), valid))| {
                    let base: usize = block * chunk;
                    for offset in 0..valid.len() {
                        let (warp, flag) = body(base + offset);
                        m00[offset] = warp[0];
                        m01[offset] = warp[1];
                        m10[offset] = warp[2];
                        m11[offset] = warp[3];
                        tx[offset] = warp[4];
                        ty[offset] = warp[5];
                        valid[offset] = flag;
                    }
                });
        });
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn buffers(len: usize) -> ([Vec<f32>; 6], Vec<bool>) {
        (std::array::from_fn(|_| vec![0.0; len]), vec![false; len])
    }

    #[test]
    fn one_thread_takes_the_sequential_path() {
        let pool: WorkPool = WorkPool::new(1).unwrap();
        assert_eq!(pool.threads(), 1);
        assert!(pool.pool.is_none());
    }

    #[test]
    fn zero_threads_is_read_as_one() {
        assert_eq!(WorkPool::new(0).unwrap().threads(), 1);
    }

    #[test]
    fn every_index_is_visited_exactly_once_however_wide_the_pool() {
        for threads in [1, 2, 3, 8] {
            let pool: WorkPool = WorkPool::new(threads).unwrap();
            let (mut coefficients, mut valid) = buffers(37);
            let [m00, m01, m10, m11, tx, ty] = &mut coefficients;
            pool.for_each_warp([m00, m01, m10, m11, tx, ty], &mut valid, |index| {
                ([index as f32; 6], index % 3 == 0)
            });
            for slot in &coefficients {
                for (index, value) in slot.iter().enumerate() {
                    assert_eq!(*value, index as f32, "at {threads} threads");
                }
            }
            for (index, flag) in valid.iter().enumerate() {
                assert_eq!(*flag, index % 3 == 0);
            }
        }
    }

    /// The split must not change a value, whatever the width.
    #[test]
    fn every_width_produces_the_same_arrays() {
        let body = |index: usize| {
            let value: f32 = (index as f32 * 0.37).sin();
            (
                [value, value * 2.0, value * 3.0, value, -value, value],
                !index.is_multiple_of(5),
            )
        };
        let mut reference: Option<([Vec<f32>; 6], Vec<bool>)> = None;
        for threads in [1, 2, 5, 16] {
            let pool: WorkPool = WorkPool::new(threads).unwrap();
            let (mut coefficients, mut valid) = buffers(101);
            let [m00, m01, m10, m11, tx, ty] = &mut coefficients;
            pool.for_each_warp([m00, m01, m10, m11, tx, ty], &mut valid, body);
            match &reference {
                None => reference = Some((coefficients, valid)),
                Some(expected) => assert_eq!(&(coefficients, valid), expected),
            }
        }
    }

    #[test]
    fn an_empty_range_is_a_no_op() {
        let pool: WorkPool = WorkPool::new(4).unwrap();
        let (mut coefficients, mut valid) = buffers(0);
        let [m00, m01, m10, m11, tx, ty] = &mut coefficients;
        pool.for_each_warp([m00, m01, m10, m11, tx, ty], &mut valid, |_| unreachable!());
    }

    /// A short flag array bounds the pass rather than panicking.
    #[test]
    fn the_shortest_input_bounds_the_pass() {
        let pool: WorkPool = WorkPool::new(2).unwrap();
        let (mut coefficients, _) = buffers(10);
        let mut valid: Vec<bool> = vec![false; 4];
        let [m00, m01, m10, m11, tx, ty] = &mut coefficients;
        pool.for_each_warp([m00, m01, m10, m11, tx, ty], &mut valid, |index| {
            assert!(index < 4);
            ([1.0; 6], true)
        });
        assert!(valid.iter().all(|flag| *flag));
        assert_eq!(coefficients[0][4], 0.0);
    }
}
