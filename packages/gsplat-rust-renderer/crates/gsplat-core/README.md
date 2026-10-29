# gsplat-core

Request subgroup support and the adapter's storage limits before creating the core.
See [architecture](../../docs/architecture.md) for formats and capability limits.

```rust,ignore
let renderer = Renderer::new(&device)?;
let scene = renderer.upload(&splats)?;
let mut view = renderer.create_view(&scene, 1_048_576)?;
renderer.render(&queue, &mut encoder, &mut view, &camera, &options, target)?;
queue.submit([encoder.finish()]);
if let Some(stats) = view.poll_feedback()? {
    // Encode and submit again when stats.needs_rerender is true.
}
```

Poll completed feedback before reusing a view. Scenes can be shared by several views.

Hexels (honeycomb-surfels/hexels) share views, sorting and targets with splats:
`renderer.upload_hexels(&hexels)` takes activated rows in the hexel PLY order. 3D hexels
need a pinhole camera; `Target::Surface` adds expected and median depth and world normals.
`HexelVariant::Image` renders 2D fits with upstream's top-10 normalized blend.
