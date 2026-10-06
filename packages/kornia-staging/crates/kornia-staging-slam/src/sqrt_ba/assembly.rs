//! Checked row layout and explicit, deterministic block execution.
use kornia_staging_algebra::Scalar;
use rayon::prelude::*;

/// Fold one logical job per block, independent of worker count. Parallel results
/// are collected by index before the serial fold. The serial path allocates no
/// result vector and stops immediately on the first error.
/// # Arguments
/// `body` returns a cost and validity flag. `pool` is explicit, including the
/// serial choice `None`; blocks retain their input order.
/// # Errors
/// Returns the first error in block order. Parallel jobs all finish before this
/// check, whereas the serial path does not run jobs after an error.
pub fn linearize_blocks<S: Scalar, B: Send, E: Send>(
    blocks: &mut [B],
    pool: Option<&rayon::ThreadPool>,
    body: impl Fn(usize, &mut B) -> Result<(S, bool), E> + Send + Sync,
) -> Result<(S, bool), E> {
    let mut error = S::zero();
    let mut valid = true;
    if let Some(results) = pool.map(|pool| {
        pool.install(|| {
            blocks
                .par_iter_mut()
                .enumerate()
                .map(|(i, block)| body(i, block))
                .collect::<Vec<_>>()
        })
    }) {
        for result in results {
            let (cost, ok) = result?;
            error += cost;
            valid &= ok;
        }
    } else {
        for (i, block) in blocks.iter_mut().enumerate() {
            let (cost, ok) = body(i, block)?;
            error += cost;
            valid &= ok;
        }
    }
    Ok((error, valid))
}
/// Eliminate independent blocks with the same indexed execution and first-error
/// contract as [`linearize_blocks`]. No arithmetic is reduced in parallel.
/// # Errors
/// Returns the first block error in input order.
pub fn eliminate_blocks<B: Send, E: Send>(
    blocks: &mut [B],
    pool: Option<&rayon::ThreadPool>,
    body: impl Fn(&mut B) -> Result<(), E> + Send + Sync,
) -> Result<(), E> {
    crate::parallel::try_for_each_mut(pool, blocks, |_, block| body(block))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn indexed_folds_and_first_errors_do_not_depend_on_workers() {
        let mut expected = None;
        for width in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(width)
                .build()
                .unwrap();
            let mut blocks = [1e16f64, 1.0, -1e16, 3.0];
            let actual = linearize_blocks(&mut blocks, Some(&pool), |i, value| {
                Ok::<_, usize>((*value, i != 2))
            })
            .unwrap();
            assert_eq!(actual, (3.0, false));
            assert_eq!(
                *expected.get_or_insert(actual.0.to_bits()),
                actual.0.to_bits()
            );
            let result = eliminate_blocks(&mut blocks, Some(&pool), |value| {
                *value += 1.0;
                Err(*value as i64)
            });
            assert_eq!(result, Err(10000000000000000));
            assert_eq!(blocks[1], 2.0); // all parallel jobs ran before first-error selection
        }
        let mut blocks = [0, 0, 0];
        let result = eliminate_blocks(&mut blocks, None::<&rayon::ThreadPool>, |n| {
            *n += 1;
            Err(7)
        });
        assert_eq!(result, Err(7));
        assert_eq!(blocks, [1, 0, 0]);
    }
}
