# gsplat

From the package directory in an activated Pixi shell:

```bash
target/release/gsplat render --ply scene.ply --camera transforms_test.json \
  --frame 0 --output render.png --background 0,0,0
target/release/gsplat render --ply scene.ply --camera transforms_test.json \
  --output-dir renders --width 800 --height 800 --render-mode mip
target/release/gsplat eval --render renders --gt ground-truth \
  --convention brush --lpips --out metrics.json
target/release/gsplat score --ply export_7000.ply --dataset SCENE --out score.json
target/release/gsplat parity --impl ours --oracle brush \
  --ply scene.ply --path transforms_test.json --holdout-every 8 --out parity.json
target/release/gsplat parity --impl ours --archetype recording.rrd \
  --ply scene.ply --path cameras.json --save-images evidence --out quantization.json
```

Camera files accept NeRF transforms, CameraSpec JSON arrays, or COLMAP model
directories. `--res native` retains input
dimensions. `--render-mode auto|default|mip`, `--splat-scale`, `--min-scale`, and
`--initial-capacity` control compute/Brush rendering.

An archetype recording must contain one complete splat row; omit its path to
load the PLY through Rerun. Parity scores unclipped RGBA and white composition;
EXR evidence preserves those values, while PNGs are previews.

`score` evaluates unclipped float renders against the NeRF test split for
training quality. `eval` pairs directories by identical relative PNG paths;
the ground-truth directory must contain matching RGB images without depth sidecars.
`--convention published` selects the white-background checkpoint convention,
and `brush` is the default. PNG scoring includes export clipping and rounding.

See [architecture](../../docs/architecture.md) for the timing
protocol, precision boundaries, and report provenance.
