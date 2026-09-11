//! Fixed-order reductions. Sequential execution is a left fold in input order.
//! Thread count cannot change the association or the result.

use crate::lie::LieScalar;

pub(crate) fn deterministic_reduce<T, E>(
    n: usize,
    out: &mut T,
    leaf: &mut dyn FnMut(usize, &mut T) -> Result<(), E>,
) -> Result<(), E> {
    for i in 0..n {
        leaf(i, out)?;
    }
    Ok(())
}

pub(crate) fn deterministic_reduce_scalar<S: LieScalar, E>(
    n: usize,
    leaf: &mut dyn FnMut(usize, S) -> Result<S, E>,
) -> Result<S, E> {
    let mut sum = S::zero();
    deterministic_reduce(n, &mut sum, &mut |i, sum| {
        *sum = leaf(i, *sum)?;
        Ok(())
    })?;
    Ok(sum)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn fixed_left_fold_across_thread_counts() {
        // S34 natural arithmetic defines the sequential path as a left fold.
        let values = [16_777_216.0f32, 1.0, 1.0, 1.0];
        for threads in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for _ in 0..3 {
                let sum = pool.install(|| {
                    deterministic_reduce_scalar::<f32, ()>(4, &mut |i, acc| Ok(acc + values[i]))
                        .unwrap()
                });
                assert_eq!(sum.to_bits(), 16_777_216.0f32.to_bits());
            }
        }
    }

    #[test]
    fn empty_and_failing_reductions() {
        assert_eq!(
            deterministic_reduce_scalar::<f64, ()>(0, &mut |_, _| panic!("empty")),
            Ok(0.0)
        );
        let mut visited = Vec::new();
        let result = deterministic_reduce_scalar::<f64, &str>(8, &mut |i, acc| {
            visited.push(i);
            if i == 3 { Err("stop") } else { Ok(acc + 1.0) }
        });
        assert_eq!(result, Err("stop"));
        assert_eq!(visited, [0, 1, 2, 3]);
    }
}
