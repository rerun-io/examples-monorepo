//! The frontend's one and only source of parallelism.
//!
//! basalt runs the per-keypoint tracking loop through `tbb::parallel_for` over a
//! `blocked_range` (`frame_to_frame_optical_flow.h:368-369`) and the per-camera
//! pyramid build the same way (`:224-228`, `:245-249`). Decision D31 fixes how
//! that lands in Rust: a **fixed chunk size**, per-chunk sequential accumulation,
//! and an **explicit thread count** from the config — never the ambient rayon
//! global pool, whose width depends on the machine and on whatever else in the
//! process touched rayon first.
//!
//! Every loop routed through here writes only to its own index, so the chunking
//! cannot change a result; the fixed chunk size is belt and braces, and the tests
//! run `threads = 1` against `threads = 4` and require identical output.

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuildError, ThreadPoolBuilder};

/// A fixed-width worker pool, or the sequential path when one thread was asked for.
#[derive(Debug)]
pub struct WorkPool {
    threads: usize,
    /// `None` at `threads == 1`: the sequential path runs on the caller's thread
    /// and never touches rayon at all.
    pool: Option<ThreadPool>,
}

impl WorkPool {
    /// A pool of exactly `threads` workers; `threads == 0` is read as one.
    ///
    /// # Errors
    ///
    /// [`ThreadPoolBuildError`] when the operating system refuses the threads.
    pub fn new(threads: usize) -> Result<Self, ThreadPoolBuildError> {
        let threads: usize = threads.max(1);
        let pool: Option<ThreadPool> = if threads == 1 {
            None
        } else {
            Some(ThreadPoolBuilder::new().num_threads(threads).build()?)
        };
        Ok(Self { threads, pool })
    }

    /// Workers this pool runs on.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Apply `body` to every index of `0..len`, in fixed-size chunks.
    ///
    /// `body` receives the index and must write only to slot `index` of whatever
    /// it owns, which is what makes the split irrelevant to the result. The
    /// caller supplies the two output slices the tracking phases write, because
    /// rayon needs disjoint mutable borrows and closures cannot produce them.
    pub fn for_each_indexed<A: Send, B: Send>(
        &self,
        first: &mut [A],
        second: &mut [B],
        body: impl Fn(usize, &mut A, &mut B) + Sync + Send,
    ) {
        let len: usize = first.len().min(second.len());
        let first: &mut [A] = &mut first[..len];
        let second: &mut [B] = &mut second[..len];

        let Some(pool) = self.pool.as_ref() else {
            for (index, (a, b)) in first.iter_mut().zip(second.iter_mut()).enumerate() {
                body(index, a, b);
            }
            return;
        };

        // One chunk per worker, rounded up: a fixed split for a given
        // (len, threads), not whatever rayon's work stealing would pick.
        let chunk: usize = len.div_ceil(self.threads).max(1);
        pool.install(|| {
            first
                .par_chunks_mut(chunk)
                .zip(second.par_chunks_mut(chunk))
                .enumerate()
                .for_each(|(block, (first_block, second_block))| {
                    let base: usize = block * chunk;
                    for (offset, (a, b)) in first_block
                        .iter_mut()
                        .zip(second_block.iter_mut())
                        .enumerate()
                    {
                        body(base + offset, a, b);
                    }
                });
        });
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

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
            let mut squares: Vec<usize> = vec![0; 37];
            let mut seen: Vec<u8> = vec![0; 37];
            pool.for_each_indexed(&mut squares, &mut seen, |index, square, mark| {
                *square = index * index;
                *mark += 1;
            });
            assert!(seen.iter().all(|count| *count == 1));
            for (index, square) in squares.iter().enumerate() {
                assert_eq!(*square, index * index);
            }
        }
    }

    #[test]
    fn an_empty_range_is_a_no_op() {
        let pool: WorkPool = WorkPool::new(4).unwrap();
        let mut nothing: Vec<usize> = Vec::new();
        let mut also_nothing: Vec<usize> = Vec::new();
        pool.for_each_indexed(&mut nothing, &mut also_nothing, |_, _, _| unreachable!());
    }
}
