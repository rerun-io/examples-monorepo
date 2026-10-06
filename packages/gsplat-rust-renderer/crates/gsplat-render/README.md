# gsplat-render

The standalone CLI and benchmark share this crate's `CameraSpec`, file adapters,
raw upload helper, and persistent renderer. It uses `gsplat-core` and Brush's
pinned PLY/COLMAP readers. It has no Rerun dependency.

```sh
pixi run -e gsplat-rust-renderer-dev --frozen gsplat-rust-renderer-render \
  --ply scene.ply --camera transforms_test.json --frame 0 \
  --width 800 --height 800 --background 1,1,1 --output frame.png
```

`--output-dir` renders every frame while preserving relative image paths.
`--camera` also accepts a strict `CameraSpec` JSON array or a COLMAP sparse
directory containing `cameras.bin`/`images.bin` or their text counterparts.
The four core lens models and COLMAP's simpler radial variants are preserved;
the unsupported COLMAP FOV model returns an error.

`--render-mode auto` uses PLY metadata; `default` and `mip` override it.
`--splat-scale` is a positive multiplier. Background is applied by the core
before a symmetric clipped, rounded RGB8 export. Rendered alpha is coverage;
it is not applied a second time during export.

`--benchmark --num-frames N` reports render plus GPU completion without pixel
readback. Use `gsplat-bench speed` for the full idle-GPU, repeated protocol.
