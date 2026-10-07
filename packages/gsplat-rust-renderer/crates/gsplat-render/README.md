# gsplat-render

From the package directory in an activated Pixi shell:

```bash
target/release/gsplat-render --ply scene.ply --camera transforms_test.json \
  --frame 0 --output render.png --background 0,0,0
target/release/gsplat-render --ply scene.ply --camera transforms_test.json \
  --output-dir renders --width 800 --height 800 --render-mode mip
```

Cameras may be NeRF transforms, a CameraSpec JSON array, or a COLMAP model directory.
Use `--splat-scale`, `--min-scale`, and `--initial-capacity` to set render controls.
