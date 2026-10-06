use criterion::{criterion_group, criterion_main, Criterion};
use handfit::model::Step;
use handfit::residual::project_camera;
use handfit::scale::{calibrate_scale, CalibrationBlock, CalibrationConfig};
use handfit::{Model, Pose, View};
use nalgebra::{Matrix3, SMatrix, SVector, Vector2, Vector3};
use std::hint::black_box;

fn stereo_problem() -> (Model, Vec<CalibrationBlock>) {
    let model = Model {
        axes: SMatrix::from_fn(|i, k| if k == i % 3 { 1.0 } else { 0.0 }),
        pivots: SMatrix::from_fn(|i, k| ((i * 3 + k) as f64).cos() * 20.0),
        rest: SMatrix::from_fn(|i, k| ((i * 3 + k) as f64).sin() * 50.0),
        weights: SMatrix::from_fn(|_, k| if k == 0 { 1.0 } else { 0.0 }),
        limits: SMatrix::from_fn(|_, k| if k == 0 { -1.0 } else { 1.0 }),
    };
    let mut scaled = model.clone();
    scaled.pivots *= 1.2;
    scaled.rest *= 1.2;
    let hands = (0..4)
        .map(|frame| {
            let pose = Pose {
                rotation: Matrix3::identity(),
                translation: Vector3::new(0.02 * frame as f64, -0.01, 0.6),
                angles: SVector::repeat(0.1),
            };
            let points = scaled.landmarks(&pose, 1.0, &Step::zeros(), None);
            let views = [0.0, -0.15]
                .into_iter()
                .map(|offset| {
                    let mut view = View {
                        rotation: Matrix3::identity(),
                        translation: Vector3::new(offset, 0.0, 0.0),
                        camera: handfit::residual::camera_model(&Vector2::repeat(500.0), &Vector2::new(320.0, 240.0), None).unwrap(),
                        pixels: SMatrix::zeros(),
                        distances: SVector::zeros(),
                        weights: SVector::repeat(1.0),
                    };
                    let camera =
                        view.camera;
                    for i in 0..21 {
                        let p = points.fixed_rows::<3>(3 * i) + view.translation;
                        let pixel = project_camera(&camera, &p, None);
                        view.pixels.row_mut(i).copy_from(&pixel.transpose());
                        view.distances[i] = p.norm() * 1000.0;
                    }
                    view
                })
                .collect();
            let mut initial = pose;
            initial.translation += Vector3::new(0.004, -0.002, 0.01);
            CalibrationBlock {
                mirror: 1.0,
                views,
                initial,
            }
        })
        .collect();
    (model, hands)
}

fn benchmark(c: &mut Criterion) {
    let (model, hands) = stereo_problem();
    let config = CalibrationConfig::default();
    c.bench_function("calibrate_scale_four_stereo_poses", |b| {
        b.iter(|| {
            calibrate_scale(black_box(&model), black_box(&hands), black_box(&config)).unwrap()
        })
    });
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
