#![allow(clippy::unwrap_used)]
use slam_rs::frontend::parallel::WorkPool;
use slam_rs::pyramid::{CpuPyramidBuilder, PyramidBuilder};
#[test]
fn each_builder_entry_changes_the_host_generation() {
    let image = slam_rs::image::zeros(32, 32).unwrap();
    let mut builder = CpuPyramidBuilder::new();
    let mut pyramid = builder.allocate(32, 32, 2).unwrap();
    builder.build(0, &image, &mut pyramid).unwrap();
    let first = pyramid.generation();
    assert_eq!(first, pyramid.clone().generation());
    builder.build(0, &image, &mut pyramid).unwrap();
    let second = pyramid.generation();
    assert_ne!(first, second);
    builder
        .build_frames(
            &[image],
            std::slice::from_mut(&mut pyramid),
            &WorkPool::new(4).unwrap(),
        )
        .unwrap();
    assert_ne!(second, pyramid.generation());
}
