# gsplat-core

Raw-wgpu Gaussian splat rendering with shared scene uploads and independent views.
Pinhole, KB4, RT8 and thin-prism cameras use Brush-compatible f32 parameters.

Request `wgpu::Features::SUBGROUP`, standard WebGPU compute limits, and storage
binding limits large enough for the scene. Unsupported devices return a capability
error. Stock Rerun 0.38.1 needs a custom eframe device descriptor for these limits.

```rust,ignore
let renderer = Renderer::new(&device, &queue)?;
let scene = renderer.upload(&splats)?;
let mut view = renderer.create_view(&scene, 1_048_576)?;
renderer.render(&mut encoder, &mut view, &camera, &options, target)?;
queue.submit([encoder.finish()]);
if let Some(stats) = view.poll_feedback()? {
    // Submit the view again when stats.needs_rerender is true.
}
```

Several views can share `scene` and encode before one submission. Each view keeps
three uniform/feedback slots; poll completed feedback before reuse or resizing.
Outputs are unclipped f32 RGBA, packed RGBA8, or a single-mip rgba8unorm texture.

Native WGSL is validated by naga in the unit lane. A future web build needs wgpu's
WebGPU SUBGROUP mapping (at least Brush fork commit `4db81837f`) and must prepend
`enable subgroups;` on the WebGPU backend only: naga 30 rejects that directive.
No web path is built here. The viewer must request the same compute feature tier.

Run `pixi run -e gsplat-rust-renderer-dev --frozen gate` for unit/GPU contracts and
`tests-golden` for the nine shared-benchmark float parity cases (assets required).
