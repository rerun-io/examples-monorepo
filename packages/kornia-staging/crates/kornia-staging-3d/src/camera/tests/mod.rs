use super::*;
use approx::assert_relative_eq;

#[derive(serde::Deserialize)]
struct ProjectionFixture {
    intrinsics: [f64; 4],
    distortion: Vec<f64>,
    points: Vec<[f64; 3]>,
    pixels: Vec<[f64; 2]>,
}

// Public derivative and round-trip seams from slam-rs/tests/camera_jacobians.rs.
fn sweep<
    S: Scalar,
    M: CameraModel<S, IntrinsicJacobian = [[S; N]; 2], UnprojectIntrinsicJacobian = [[S; N]; 3]>,
    const N: usize,
>(
    params: [S; N],
    build: impl Fn([S; N]) -> M,
    step: f64,
    tolerance: f64,
) {
    let camera = build(params);
    let h = c::<S>(step);
    for x in -3..=3 {
        for y in -3..=3 {
            for z in [0.5, 1.0, 4.0] {
                let point = [c::<S>(x as f64 * 0.3), c::<S>(y as f64 * 0.3), c::<S>(z)];
                let Ok(pixel) = camera.project(point) else {
                    continue;
                };
                let mut jp = [[S::zero(); 3]; 2];
                let mut ji = [[S::zero(); N]; 2];
                camera.project_unchecked_with_jacobians(point, Some(&mut jp), Some(&mut ji));
                for i in 0..3 {
                    let mut a = point;
                    let mut b = point;
                    a[i] += h;
                    b[i] -= h;
                    let pa = camera.project_unchecked(a);
                    let pb = camera.project_unchecked(b);
                    for r in 0..2 {
                        let numerical = (pa[r] - pb[r]) / (h + h);
                        let actual = jp[r][i];
                        assert!(
                            (actual - numerical).abs().to_f64()
                                < tolerance * (1.0 + actual.abs().to_f64()),
                            "JP {r},{i}: {} vs {}",
                            actual.to_f64(),
                            numerical.to_f64()
                        );
                    }
                }
                for i in 0..N {
                    let mut a = params;
                    let mut b = params;
                    a[i] += h;
                    b[i] -= h;
                    let pa = build(a).project_unchecked(point);
                    let pb = build(b).project_unchecked(point);
                    for r in 0..2 {
                        let numerical = (pa[r] - pb[r]) / (h + h);
                        let actual = ji[r][i];
                        assert!(
                            (actual - numerical).abs().to_f64()
                                < tolerance * (1.0 + actual.abs().to_f64()),
                            "JI {r},{i}"
                        );
                    }
                }
                let ray = camera.unproject(pixel).unwrap();
                let norm = (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt();
                for i in 0..3 {
                    assert!((ray[i] - point[i] / norm).abs().to_f64() < tolerance);
                }
                let mut up = [[S::zero(); 2]; 3];
                let mut ui = [[S::zero(); N]; 3];
                camera
                    .unproject_with_jacobians(pixel, Some(&mut up), Some(&mut ui))
                    .unwrap();
                for i in 0..2 {
                    let mut a = pixel;
                    let mut b = pixel;
                    a[i] += h;
                    b[i] -= h;
                    let pa = camera.unproject(a).unwrap();
                    let pb = camera.unproject(b).unwrap();
                    for r in 0..3 {
                        let numerical = (pa[r] - pb[r]) / (h + h);
                        assert!((up[r][i] - numerical).abs().to_f64() < tolerance);
                    }
                }
                for i in 0..N {
                    let mut a = params;
                    let mut b = params;
                    a[i] += h;
                    b[i] -= h;
                    let pa = build(a).unproject(pixel).unwrap();
                    let pb = build(b).unproject(pixel).unwrap();
                    for r in 0..3 {
                        let numerical = (pa[r] - pb[r]) / (h + h);
                        assert!(
                            (ui[r][i] - numerical).abs().to_f64()
                                < tolerance * (1.0 + ui[r][i].abs().to_f64()),
                            "JU {r},{i}: {} vs {}, point {:?}, step {step}",
                            ui[r][i].to_f64(),
                            numerical.to_f64(),
                            point.map(|v| v.to_f64())
                        );
                    }
                }
            }
        }
    }
}

fn right_front() -> KannalaBrandt4<f64> {
    KannalaBrandt4::new([
        630.6917724609375,
        628.777587890625,
        946.6721801757812,
        539.53125,
        0.07725944370031357,
        -0.06258341670036316,
        0.08006518334150314,
        -0.02879575826227665,
    ])
    .expect("valid camera calibration")
}

mod brown;
mod fisheye624;
#[cfg(feature = "serde")]
mod formats;
mod kb4;
mod pinhole;
mod regressions;

#[test]
fn inverse_stops_backtracking_when_the_candidate_cannot_move() {
    let calls = std::cell::Cell::new(0usize);
    let result = super::newton2([1.0f32, 0.0], |_, jacobian| {
        calls.set(calls.get() + 1);
        *jacobian = [[1e20, 0.0], [0.0, 1e10]];
        [1.001, 0.0]
    });
    assert_eq!(result, Err(super::UnprojectError::NoConvergence));
    assert!(
        calls.get() <= 2,
        "{} identical candidates evaluated",
        calls.get()
    );
}

#[cfg(feature = "serde")]
mod basalt;
