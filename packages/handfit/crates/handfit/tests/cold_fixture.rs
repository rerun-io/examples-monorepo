use handfit::{cold::wrist_hypotheses, Config, Model, View, Views};
use nalgebra::{Matrix3, SMatrix, SVector, Vector2, Vector3};

#[test]
fn golden_wrist_hypotheses_match_torch_export_in_order() {
    let values: Vec<f64> = include_str!("fixtures/cold.txt")
        .lines()
        .filter(|s| !s.starts_with('#'))
        .map(|s| s.parse().unwrap())
        .collect();
    let mut offset = 0;
    let mut take = |n: usize| {
        let slice = &values[offset..offset + n];
        offset += n;
        slice
    };
    let phi = take(1)[0];
    let mirror = take(1)[0];
    let count = take(1)[0] as usize;
    let model = Model {
        axes: SMatrix::from_row_slice(take(60)),
        pivots: SMatrix::from_row_slice(take(60)),
        rest: SMatrix::from_row_slice(take(63)),
        weights: SMatrix::from_row_slice(take(63)),
        limits: SMatrix::from_row_slice(take(40)),
    };
    let views: Vec<View> = (0..count)
        .map(|_| View {
            rotation: Matrix3::from_row_slice(take(9)),
            translation: Vector3::from_row_slice(take(3)),
            focal: Vector2::from_row_slice(take(2)),
            principal: Vector2::from_row_slice(take(2)),
            distortion: Some(SVector::from_row_slice(take(8))),
            pixels: SMatrix::from_row_slice(take(42)),
            weights: SVector::from_row_slice(take(21)),
            distances: SVector::from_row_slice(take(21)),
        })
        .collect();
    let rotations = take(26 * 9);
    let translations = take(26 * 3);
    let usable = take(26);
    let (poses, valid) = wrist_hypotheses(
        &model,
        &Config {
            phi,
            ..Config::default()
        },
        mirror,
        Views::new(&views).unwrap(),
    );
    assert_eq!(poses.len(), 26);
    for i in 0..26 {
        assert_eq!(valid[i], usable[i] != 0.0);
        // The absent view has no well-defined Kabsch orientation; its mask is the contract.
        if valid[i] {
            let rotation = Matrix3::from_row_slice(&rotations[9 * i..9 * i + 9]);
            let translation = Vector3::from_row_slice(&translations[3 * i..3 * i + 3]);
            assert!((poses[i].rotation - rotation).norm() < 2e-5, "rotation {i}");
            assert!(
                (poses[i].translation - translation).norm() < 2e-6,
                "translation {i}"
            );
            assert!((poses[i].rotation.determinant() - 1.0).abs() < 1e-6);
        }
    }
}
