# Compute Gaussian viewer

`gsplat-rust-renderer` is a Rerun 0.38.1 viewer that draws native
`GaussianSplats3D` through `gsplat-core`. Open a recording or connect the SDK;
no visualizer override is required. Stock Rerun draws the same recording natively.
Explicit native and compute selections are preserved.

Python tools leave selection unset. `--compute --render-mode mip` is an explicit
custom-viewer choice. Stock Rerun does not implement `ComputeGaussianSplats3D`,
so that explicit override cannot draw splats there. Stock 0.38.1 silently skips
unknown visualizers; our explicit-choice Python helper warns before logging. The render-mode property is
custom-viewer-only; stock ignores it on a native instruction.

Picking, hover, and selection outlines for compute splats are not implemented.
Instances are separate full renders, and different splat entities are not jointly
sorted. Rerun 0.38.1 supplies the previous frame's eye to visualizers, so moving
splats can lag native content by one frame. Perspective cameras are supported.

See [architecture](../../docs/architecture.md) for selection, cache, depth,
framing and device details. Pixel tests are in `tests/test_viewer_pixels.py`.
