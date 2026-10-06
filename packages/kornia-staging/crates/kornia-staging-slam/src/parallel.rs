//! Ordered failure reporting for optional caller-owned pools.
use rayon::{prelude::*, ThreadPool};

// Parallel lanes all finish before the first error in input order is returned.
// Serial execution stops at that same first error without allocating a vector.
pub(crate) fn try_for_each_mut<T: Send, E: Send>(
    pool: Option<&ThreadPool>,
    items: &mut [T],
    body: impl Fn(usize, &mut T) -> Result<(), E> + Sync + Send,
) -> Result<(), E> {
    if let Some(pool) = pool {
        let results = pool.install(|| {
            items
                .par_iter_mut()
                .enumerate()
                .map(|(index, item)| body(index, item))
                .collect::<Vec<_>>()
        });
        for result in results {
            result?;
        }
        Ok(())
    } else {
        items
            .iter_mut()
            .enumerate()
            .try_for_each(|(index, item)| body(index, item))
    }
}
