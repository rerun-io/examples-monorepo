use super::*;

pub(super) fn radial<S: Scalar>(theta: S, k: &[S]) -> (S, S) {
    let t2 = theta * theta;
    let mut value = S::zero();
    let mut slope = S::zero();
    for (i, &ki) in k.iter().enumerate().rev() {
        value = (value + ki) * t2;
        slope = (slope + c::<S>((2 * i + 3) as f64) * ki) * t2;
    }
    (theta * (S::one() + value), S::one() + slope)
}

// Isolate all real roots by the derivative's roots. Between consecutive critical
// points a polynomial is monotone, so bisection cannot skip a narrow negative lobe.
fn roots(coefficients: &[f64], low: f64, high: f64) -> Vec<f64> {
    let mut n = coefficients.len();
    while n > 1 && coefficients[n - 1] == 0.0 {
        n -= 1;
    }
    if n <= 1 {
        return Vec::new();
    }
    let eval = |x: f64| coefficients[..n].iter().rev().fold(0.0, |v, k| v * x + k);
    let derivative: Vec<_> = (1..n).map(|i| coefficients[i] * i as f64).collect();
    let mut knots = vec![low];
    knots.extend(roots(&derivative, low, high));
    knots.push(high);
    let mut result = Vec::new();
    for pair in knots.windows(2) {
        let (mut a, mut b) = (pair[0], pair[1]);
        let fa = eval(a);
        let fb = eval(b);
        if fa == 0.0 {
            result.push(a);
        }
        if fa.is_sign_positive() != fb.is_sign_positive() {
            for _ in 0..64 {
                let m = (a + b) * 0.5;
                if eval(m).is_sign_positive() == fa.is_sign_positive() {
                    a = m;
                } else {
                    b = m;
                }
            }
            result.push((a + b) * 0.5);
        }
    }
    result
}

pub(super) fn peak<S: Scalar>(k: &[S]) -> S {
    let mut coefficients = vec![1.0];
    coefficients.extend(
        k.iter()
            .enumerate()
            .map(|(i, v)| (2 * i + 3) as f64 * v.to_f64()),
    );
    let pi = std::f64::consts::PI;
    let mut candidates = roots(&coefficients, 0.0, pi * pi);
    candidates.sort_by(f64::total_cmp);
    candidates.dedup();
    candidates.push(pi * pi);
    let eval = |x: f64| coefficients.iter().rev().fold(0.0, |v, k| v * x + k);
    // A touching zero is stationary but does not end the increasing branch.
    let first = candidates
        .windows(2)
        .find(|pair| pair[0] > 0.0 && eval((pair[0] + pair[1]) * 0.5) < 0.0)
        .map(|pair| pair[0]);
    c(first.map(f64::sqrt).unwrap_or(pi))
}

pub(super) fn invert<S: Scalar>(radius: S, k: &[S], max: S, limit: S) -> Result<S, UnprojectError> {
    if !radius.is_finite() {
        return Err(UnprojectError::NonFinite);
    }
    let endpoint_tolerance = super::inverse_epsilon::<S>() * (S::one() + limit.abs());
    if radius > limit + endpoint_tolerance || radius < S::zero() {
        return Err(UnprojectError::OutsideDomain);
    }
    if radius == S::zero() {
        return Ok(S::zero());
    }
    if (radius - limit).abs() <= endpoint_tolerance {
        return Ok(max);
    }
    let (mut low, mut high) = (S::zero(), max);
    let mut theta = radius.min(max);
    for _ in 0..80 {
        let (value, slope) = radial(theta, k);
        let residual = value - radius;
        if residual.abs() <= super::inverse_epsilon::<S>() * (S::one() + radius) {
            return Ok(theta);
        }
        if residual > S::zero() {
            high = theta;
        } else {
            low = theta;
        }
        let newton = theta - residual / slope;
        theta = if slope > S::zero() && newton > low && newton < high {
            newton
        } else {
            c::<S>(0.5) * (low + high)
        };
    }
    Err(UnprojectError::NoConvergence)
}

// Account for the inverse solve's residual tolerance near a stationary branch endpoint.
const INVERSE_SLOPE_TOLERANCE_FACTOR: f64 = 8.0;
pub(super) fn slope_is_singular<S: Scalar>(theta: S, terms: &[S]) -> bool {
    radial(theta, terms).1.abs()
        <= super::inverse_epsilon::<S>() * c(INVERSE_SLOPE_TOLERANCE_FACTOR)
}
