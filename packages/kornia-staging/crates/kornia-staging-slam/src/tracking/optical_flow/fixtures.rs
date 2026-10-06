use kornia_image::Image;
use kornia_staging_imgproc::pyramid::PyramidPlanU16;
pub(super) fn pyramid_of(image: &Image<u16, 1>, levels: usize) -> PyramidPlanU16 {
    let mut pyramid = PyramidPlanU16::new(image.size(), levels).unwrap();
    pyramid.run(image).unwrap();
    pyramid
}
pub(super) fn pool(threads: usize) -> Option<std::sync::Arc<rayon::ThreadPool>> {
    (threads > 1).then(|| {
        std::sync::Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap(),
        )
    })
}
