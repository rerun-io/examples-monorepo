# gsplat-core

Request subgroup support and the adapter's storage limits before creating the core.
See [architecture](../../docs/architecture.md) for formats and capability limits.

```rust,ignore
let renderer = Renderer::new(&device, &queue)?;
let scene = renderer.upload(&splats)?;
let mut view = renderer.create_view(&scene, 1_048_576)?;
renderer.render(&mut encoder, &mut view, &camera, &options, target)?;
queue.submit([encoder.finish()]);
if let Some(stats) = view.poll_feedback()? {
    // Encode and submit again when stats.needs_rerender is true.
}
```

Poll completed feedback before reusing a view. Scenes can be shared by several views.
