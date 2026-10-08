//! Basalt derivative grids formerly owned by the SLAM camera adapter.
#![allow(clippy::excessive_precision)]
use super::*;
use nalgebra::{SMatrix, SVector};
use serde::{de::DeserializeOwned, Deserialize};

trait Precision: Scalar + DeserializeOwned {
    const STEP: f64;
    const TOLERANCE: f64;
}
impl Precision for f64 {
    const STEP: f64 = 1e-8;
    const TOLERANCE: f64 = 1e-3;
}
impl Precision for f32 {
    const STEP: f64 = 1e-2;
    const TOLERANCE: f64 = 1e-2;
}
fn derivative<S: Precision, const R: usize, const C: usize>(
    actual: SMatrix<S, R, C>,
    f: impl Fn(SVector<S, C>) -> SVector<S, R>,
) {
    let h = S::from_literal(S::STEP);
    let tolerance = S::from_literal(S::TOLERANCE);
    let mut numeric = SMatrix::<S, R, C>::zeros();
    for column in 0..C {
        let mut delta = SVector::zeros();
        delta[column] = h;
        numeric.set_column(column, &((f(delta) - f(-delta)) / (h + h)));
    }
    assert!(actual.iter().chain(numeric.iter()).all(|x| x.is_finite()));
    let small = |m: &SMatrix<S, R, C>| m.iter().all(|x| x.abs() <= tolerance);
    assert!(
        if small(&actual) && small(&numeric) {
            small(&(actual - numeric))
        } else {
            (actual - numeric).norm() <= tolerance * actual.norm().min(numeric.norm())
        },
        "derivative mismatch: analytic {actual}, numeric {numeric}"
    );
}
// N is the staged intrinsic extent; K is the Basalt subset (12 of Brown's 18).
fn grid<S: Precision, M, const N: usize, const K: usize>(
    params: [S; N],
    build: impl Fn([S; N]) -> M,
    domain: impl Fn([S; 2], bool) -> bool,
    parameters: bool,
    inverse: bool,
) where
    M: CameraModel<S, IntrinsicJacobian = [[S; N]; 2], UnprojectIntrinsicJacobian = [[S; N]; 3]>,
{
    let camera = build(params);
    for x in -10..=10 {
        for y in -10..=10 {
            for z in -1..=5 {
                let point = [x, y, z].map(|v| S::from_literal(f64::from(v)));
                let mut jp = [[S::zero(); 3]; 2];
                let mut ji = [[S::zero(); N]; 2];
                let Ok(pixel) = camera.project_with_jacobians(point, Some(&mut jp), Some(&mut ji))
                else {
                    continue;
                };
                if !domain(pixel, false) {
                    continue;
                }
                derivative(
                    SMatrix::<S, 2, 3>::from_row_slice(jp.as_flattened()),
                    |delta| {
                        SVector::from(
                            camera.project_unchecked(std::array::from_fn(|i| point[i] + delta[i])),
                        )
                    },
                );
                if parameters {
                    derivative(SMatrix::<S, 2, K>::from_fn(|r, c| ji[r][c]), |delta| {
                        SVector::from(
                            build(std::array::from_fn(|i| {
                                params[i] + if i < K { delta[i] } else { S::zero() }
                            }))
                            .project_unchecked(point),
                        )
                    });
                }
                if !inverse || z < 0 || !domain(pixel, true) {
                    continue;
                }
                let mut up = [[S::zero(); 2]; 3];
                let mut ui = [[S::zero(); N]; 3];
                camera
                    .unproject_with_jacobians(pixel, Some(&mut up), Some(&mut ui))
                    .unwrap();
                derivative(
                    SMatrix::<S, 3, 2>::from_row_slice(up.as_flattened()),
                    |delta| {
                        SVector::from(
                            camera
                                .unproject(std::array::from_fn(|i| pixel[i] + delta[i]))
                                .unwrap(),
                        )
                    },
                );
                derivative(SMatrix::<S, 3, K>::from_fn(|r, c| ui[r][c]), |delta| {
                    SVector::from(
                        build(std::array::from_fn(|i| {
                            params[i] + if i < K { delta[i] } else { S::zero() }
                        }))
                        .unproject(pixel)
                        .unwrap(),
                    )
                });
            }
        }
    }
}
fn synthetic<S: Precision>() {
    for params in [
        [
            460.76484651566468,
            459.4051018049483,
            365.8937161309615,
            249.33499869752445,
        ],
        [
            191.14799816648748,
            191.13150946585135,
            254.95857715233118,
            256.8815466235898,
        ],
    ] {
        grid::<S, _, 4, 4>(
            params.map(S::from_literal),
            |p| Pinhole::new(p).unwrap(),
            |_, _| true,
            true,
            true,
        );
    }
    grid::<S, _, 8, 8>(
        [
            379.045,
            379.008,
            505.512,
            509.969,
            0.00693023,
            -0.0013828,
            -0.000272596,
            -0.000452646,
        ]
        .map(S::from_literal),
        |p| KannalaBrandt4::new(p).unwrap(),
        |_, _| true,
        true,
        S::STEP < 1e-3,
    );
    let mut brown = [S::zero(); 18];
    brown[..12].copy_from_slice(
        &[
            269.0600776672363,
            269.1679859161377,
            324.3333053588867,
            245.22674560546875,
            0.6257319450378418,
            0.46612036228179932,
            -0.00018502399325370789,
            -4.2882973502855748e-5,
            0.0041795829311013222,
            0.89431935548782349,
            0.54253977537155151,
            0.0662121474742889,
        ]
        .map(S::from_literal),
    );
    grid::<S, _, 18, 12>(
        brown,
        |p| BrownConrady::new(p, Some(S::from_literal(2.5927503282280915))).unwrap(),
        |_, _| true,
        true,
        false,
    );
}
#[test]
fn synthetic_derivatives_f64() {
    synthetic::<f64>();
}
#[test]
fn synthetic_derivatives_f32() {
    synthetic::<f32>();
}

#[derive(Deserialize)]
struct Fixture<S> {
    intrinsics: Vec<crate::camera::formats::BasaltCamera<S>>,
    resolution: Vec<[u32; 2]>,
}
#[derive(Deserialize)]
struct Document<S> {
    value0: Fixture<S>,
}
fn shipped<S: Precision>() {
    for (text, radius) in [
        (
            include_str!("../../../../../fixtures/cameras/basalt/msdmi_calib.json"),
            472.0,
        ),
        (
            include_str!("../../../../../fixtures/cameras/basalt/msdmg_calib.json"),
            340.0,
        ),
        (
            include_str!("../../../../../fixtures/cameras/basalt/robocap_calib.json"),
            388.0,
        ),
    ] {
        let fixture = serde_json::from_str::<Document<S>>(text).unwrap().value0;
        for (model, resolution) in fixture.intrinsics.into_iter().zip(fixture.resolution) {
            let camera = CameraModelKind::try_from(&model).unwrap();
            let brown = matches!(camera, CameraModelKind::BrownConrady(_));
            let domain = |pixel: [S; 2], inverse: bool| {
                let [u, v] = pixel.map(Scalar::to_f64);
                let [w, h] = resolution.map(f64::from);
                let limit = radius * if brown && inverse { 0.5 } else { 1.0 };
                u >= 0.0
                    && v >= 0.0
                    && u < w
                    && v < h
                    && ((u - w / 2.0).powi(2) + (v - h / 2.0).powi(2)).sqrt() <= limit
            };
            match camera {
                CameraModelKind::Kb4(camera) => grid::<S, _, 8, 8>(
                    camera.params(),
                    |p| KannalaBrandt4::new(p).unwrap(),
                    domain,
                    true,
                    S::STEP < 1e-3,
                ),
                CameraModelKind::BrownConrady(camera) => grid::<S, _, 18, 12>(
                    camera.params(),
                    |p| BrownConrady::new(p, camera.valid_radius()).unwrap(),
                    domain,
                    S::STEP < 1e-3,
                    S::STEP < 1e-3,
                ),
                _ => panic!("unexpected shipped camera"),
            }
        }
    }
}
#[test]
fn shipped_derivatives_f64() {
    shipped::<f64>();
}
#[test]
fn shipped_derivatives_f32() {
    shipped::<f32>();
}
