//! Explicit frontend parallelism.
//! Use fixed chunks, sequential accumulation within chunks and a configured
//! thread count instead of the ambient Rayon pool. Each loop is a pure function
//! of its index; tests require identical output at one and four threads.

use std::sync::Arc;

use rayon::{ThreadPool, ThreadPoolBuildError, ThreadPoolBuilder};

/// The most workers a pool may be built for.
///
/// `ThreadPoolBuilder::num_threads` takes the request literally: rayon spawns OS
/// threads until the kernel refuses, so a `threads` of 100,000 arriving over the
/// Python boundary wedged the machine for minutes instead of returning an error.
/// A thousand is already more hardware threads than any host this port runs on,
/// and about 8 GB of thread stacks.
pub const MAX_THREADS: usize = 1024;

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
    /// A pool of exactly `threads` workers, bounded by [`MAX_THREADS`].
    ///
    /// # Arguments
    /// * `threads` - Worker count in `1..=MAX_THREADS`; one runs on the calling thread.
    ///
    /// # Errors
    ///
    /// Rejects zero or excessive workers, or an operating-system thread creation failure.
    pub fn new(threads: usize) -> Result<Self, WorkPoolError> {
        validate_threads(threads)?;
        let pool: Option<Arc<ThreadPool>> = if threads == 1 {
            None
        } else {
            Some(Arc::new(
                ThreadPoolBuilder::new().num_threads(threads).build()?,
            ))
        };
        Ok(Self { threads, pool })
    }

    /// Share this caller-owned Rayon pool with staged image operations.
    pub fn rayon_pool(&self) -> Option<Arc<ThreadPool>> {
        self.pool.clone()
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
}

/// Validate a requested worker budget without starting any threads.
/// # Errors
/// Rejects zero or a width above MAX_THREADS.
pub fn validate_threads(threads: usize) -> Result<(), WorkPoolError> {
    if !(1..=MAX_THREADS).contains(&threads) {
        return Err(WorkPoolError::InvalidThreads(threads));
    }
    Ok(())
}

/// Worker-pool construction failure.
#[derive(Debug, thiserror::Error)]
pub enum WorkPoolError {
    /// Requested width is outside `1..=MAX_THREADS`.
    #[error("worker count {0} must be in 1..={MAX_THREADS}")]
    InvalidThreads(usize),
    /// The operating system could not create the workers.
    #[error(transparent)]
    Build(#[from] ThreadPoolBuildError),
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
    fn invalid_thread_budgets_are_rejected_before_spawning() {
        assert!(WorkPool::new(0).is_err());
        assert!(WorkPool::new(MAX_THREADS + 1).is_err());
    }
}
