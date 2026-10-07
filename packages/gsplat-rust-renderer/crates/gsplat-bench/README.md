# gsplat-bench

```bash
target/release/gsplat-bench speed --impl ours,brush,native \
  --ply scene.ply --path orbit:300 --res 1920x1080 --out speed.json
target/release/gsplat-bench parity --impl ours --oracle brush \
  --ply scene.ply --path transforms_test.json --holdout-every 8 --out parity.json
target/release/gsplat-bench parity --impl ours --archetype recording.rrd \
  --ply scene.ply --path cameras.json --save-images evidence --out quantization.json
```

Paths accept `orbit:N`, NeRF JSON, CameraSpec JSON arrays, and COLMAP directories.
`--res native` retains input dimensions. `--render-mode auto|default|mip`,
`--splat-scale`, and `--min-scale` control compute/Brush rendering.
An archetype recording must contain one complete splat row; omit its path to
load the PLY through Rerun. See [architecture](../../docs/architecture.md) for
measurement boundaries and repeat statistics.

`gsplat-bench score --ply export_7000.ply --dataset SCENE --out score.json` scores
unclipped float renders against the test split for training quality.
