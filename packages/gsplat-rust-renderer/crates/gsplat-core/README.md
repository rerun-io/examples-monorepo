# gsplat-core

A raw-wgpu forward renderer hand-ported from Brush `1388f74c`. The library
has no Brush, Burn, or Rerun dependency. Brush is a test oracle only.

Create a wgpu device with `Features::SUBGROUP`, standard WebGPU compute
limits, and storage buffer limits large enough for the scene. Pass
`Capabilities::from_device(&device)?` to `Renderer::new(&device, &queue, caps)`.
A stock Rerun 0.38.1 device needs a custom eframe device descriptor to enable
these features and limits. Unsupported devices return a capability error.

`upload(&Splats)` accepts raw f32 parameters: 40-byte transforms (mean xyz,
quaternion wxyz, log-scale xyz), raw opacity, and tightly packed 12-byte RGB
SH coefficients. Shape errors fail at upload. Numerical invalidity follows
Brush's per-splat GPU culling policy. SH degrees zero through four are supported.
Cameras use right/down/forward coordinates and camera-to-world poses.
`center_uv` is the principal point divided by image dimensions.
Pinhole, KannalaBrandt4, rational RadialTangential8, and ThinPrismFisheye
models share the projection stages. Fisheye culling stops at the lens's first
fold; radial Jacobians use Brush's per-axis and radial clamp bounds.
`specialize_camera` selects fixed WGSL overrides instead of the uniform lens
switch, using the same shader source. Both paths accept every camera model.

`render(&mut encoder, &camera, &options, target)` writes to a caller-owned
float RGBA buffer, packed RGBA8 buffer, or rgba8unorm storage texture.
Float RGB is unclipped; alpha is accumulated coverage even with a nonblack
background. Packed output truncates after scaling to 255, as Brush does.
Storage textures use the format's unorm conversion.

Exact-count rendering submits projection and waits for an eight-byte readback,
then encodes depth sorting, visible projection, recursive scan, intersection
mapping/sorting, tile ranges, and rasterization into the supplied encoder.
It grows intersection storage before encoding, without truncation. Submit the
encoder before the next call on the same Renderer: its scratch and count buffers
are reused. Use separate Renderer instances for independent views or queues.
Only the count readback synchronizes with the CPU; output pixels stay on the GPU.

Tests run through the root Pixi environment:

```sh
pixi run -e gsplat-rust-renderer-dev --frozen cargo test \
  --manifest-path packages/gsplat-rust-renderer/Cargo.toml \
  -p gsplat-core --test primitives -- --ignored --test-threads=1
```

`tests/render.rs` checks analytic pixels. `tests/parity.rs` requires
`GSPLAT_TEST_PLY`, accepts `GSPLAT_TEST_CAMERAS` for a NeRF split, and otherwise
uses an orbit. Optional `GSPLAT_TEST_SIZE`, `GSPLAT_PARITY_VIEWS`, and
`GSPLAT_TEST_OUTPUT` control the viewport, view count, and saved pixel evidence.
Its direct float comparison is diagnostic; the shared benchmark's RGB8 export
plus Brush evaluator is the cross-renderer reporting convention.

`RenderOptions::render_mode` selects the default 0.3 covariance blur or the
0.1 mip blur with determinant opacity compensation. Supply the mode parsed
from PLY metadata at the loading boundary. An optional per-splat `min_scale`
adds the 3D scale floor and mass compensation. `splat_scale` is a positive
uniform multiplier applied before that floor; uploaded parameters stay intact.
