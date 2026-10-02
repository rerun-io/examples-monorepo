//! The smallest circle enclosing a set of 2D points (OpenCV's `minEnclosingCircle`, which handtrack's
//! `labels.circles.enclosing_circles` calls), written kornia-style for upstreaming to kornia-imgproc's contour features.
//!
//! Welzl's algorithm in its iterative move-to-front-free form: deterministic (input order), O(n) expected and O(n^3) worst
//! case, which is nothing for a hand's 21 keypoints. Double precision; OpenCV computes in float32 and pads the radius by
//! 1e-4, so the two agree to about 1e-4 pixels.
#![deny(missing_docs)]

/// A circle: centre `(cx, cy)` and radius, in the points' units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Circle {
    /// Centre x.
    pub cx: f64,
    /// Centre y.
    pub cy: f64,
    /// Radius (0 for a single point).
    pub radius: f64,
}

impl Circle {
    /// `[cx, cy, radius]`.
    pub fn to_array(self) -> [f64; 3] {
        [self.cx, self.cy, self.radius]
    }

    fn contains(&self, p: [f64; 2]) -> bool {
        let d = ((p[0] - self.cx).powi(2) + (p[1] - self.cy).powi(2)).sqrt();
        d <= self.radius * (1.0 + 1e-12) + 1e-9
    }

    fn diameter(a: [f64; 2], b: [f64; 2]) -> Self {
        let radius = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt() / 2.0;
        Self { cx: (a[0] + b[0]) / 2.0, cy: (a[1] + b[1]) / 2.0, radius }
    }

    /// The circle through three points; for (nearly) collinear points, the circle on the farthest pair.
    fn through(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> Self {
        let (bx, by) = (b[0] - a[0], b[1] - a[1]);
        let (cx, cy) = (c[0] - a[0], c[1] - a[1]);
        let d = 2.0 * (bx * cy - by * cx);
        let scale = (bx * bx + by * by).max(cx * cx + cy * cy);
        if d.abs() <= 1e-12 * scale.max(f64::MIN_POSITIVE) {
            let pairs = [Self::diameter(a, b), Self::diameter(a, c), Self::diameter(b, c)];
            return pairs.into_iter().fold(pairs[0], |best, circle| if circle.radius > best.radius { circle } else { best });
        }
        let b2 = bx * bx + by * by;
        let c2 = cx * cx + cy * cy;
        let ux = (cy * b2 - by * c2) / d;
        let uy = (bx * c2 - cx * b2) / d;
        Self { cx: a[0] + ux, cy: a[1] + uy, radius: (ux * ux + uy * uy).sqrt() }
    }
}

/// The smallest circle enclosing `points`.
///
/// # Arguments
///
/// * `points` - The points, `[x, y]`. Non-finite points must be filtered out by the caller.
///
/// # Returns
///
/// The circle, or `None` when `points` is empty.
///
/// # Example
///
/// ```
/// use robocap_live::hands::circles::min_enclosing_circle;
/// let circle = min_enclosing_circle(&[[0.0, 0.0], [2.0, 0.0], [1.0, 0.5]]).unwrap();
/// assert!((circle.cx - 1.0).abs() < 1e-12 && (circle.radius - 1.0).abs() < 1e-12);
/// ```
pub fn min_enclosing_circle(points: &[[f64; 2]]) -> Option<Circle> {
    let first = *points.first()?;
    let mut circle = Circle { cx: first[0], cy: first[1], radius: 0.0 };
    for i in 1..points.len() {
        if circle.contains(points[i]) {
            continue;
        }
        circle = Circle { cx: points[i][0], cy: points[i][1], radius: 0.0 };
        for j in 0..i {
            if circle.contains(points[j]) {
                continue;
            }
            circle = Circle::diameter(points[i], points[j]);
            for k in 0..j {
                if !circle.contains(points[k]) {
                    circle = Circle::through(points[i], points[j], points[k]);
                }
            }
        }
    }
    Some(circle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_single_points() {
        assert!(min_enclosing_circle(&[]).is_none());
        assert_eq!(min_enclosing_circle(&[[3.0, 4.0]]), Some(Circle { cx: 3.0, cy: 4.0, radius: 0.0 }));
    }

    #[test]
    fn a_triangle_with_an_obtuse_angle_uses_its_longest_side() -> Result<(), String> {
        let circle = min_enclosing_circle(&[[0.0, 0.0], [10.0, 0.0], [5.0, 1.0]]).ok_or("no circle")?;
        assert!((circle.cx - 5.0).abs() < 1e-12 && circle.cy.abs() < 1e-12 && (circle.radius - 5.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn collinear_points() -> Result<(), String> {
        let circle = min_enclosing_circle(&[[0.0, 0.0], [1.0, 1.0], [3.0, 3.0], [2.0, 2.0]]).ok_or("no circle")?;
        assert!((circle.cx - 1.5).abs() < 1e-12 && (circle.radius - 1.5 * 2f64.sqrt()).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn every_point_is_inside_and_three_touch_a_random_cloud() -> Result<(), String> {
        let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..200 {
            let points: Vec<[f64; 2]> = (0..21).map(|_| [400.0 * next(), 300.0 * next()]).collect();
            let circle = min_enclosing_circle(&points).ok_or("no circle")?;
            let distances: Vec<f64> = points.iter().map(|p| ((p[0] - circle.cx).powi(2) + (p[1] - circle.cy).powi(2)).sqrt()).collect();
            assert!(distances.iter().all(|d| *d <= circle.radius + 1e-6));
            assert!(distances.iter().filter(|d| (**d - circle.radius).abs() < 1e-6).count() >= 2);
        }
        Ok(())
    }
}
