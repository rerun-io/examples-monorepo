use gsplat_train::retains;

#[test]
fn retention_keeps_first_stride_and_final_once() {
    let steps: Vec<_> = (1..=7000)
        .filter(|&step| retains(step, 7000, 50, 1000))
        .collect();
    assert_eq!(steps, [50, 1000, 2000, 3000, 4000, 5000, 6000, 7000]);
    assert!(retains(37, 37, 50, 1000));
    assert!(!retains(0, 7000, 50, 1000));
}
