# gsplat-bench

```sh
pixi run -e gsplat-rust-renderer-dev gsplat-bench-build
cd packages/gsplat-rust-renderer
target/release/gsplat-bench speed --impl ours,brush,native \
  --ply scene.ply --path orbit:300 --res 1920x1080 --out speed.json
target/release/gsplat-bench parity --impl ours --oracle brush \
  --ply scene.ply --path transforms_test.json --out parity.json
```

Paths are `orbit:N` or a NeRF JSON, CameraSpec JSON array, or COLMAP directory.
`--holdout-every 8` selects every eighth camera after sorting filenames.
`--res native` retains input size. `--render-mode auto|default|mip`,
`--splat-scale` and `--min-scale` control the compute and Brush paths.

Speed waits for ten quiet host samples, warms up for at least five seconds,
and measures at least ten seconds and two complete orbits per repeat.
Frame counts can differ to meet the minimum duration; every camera receives
equal weight through complete orbits. Equal counts would either shorten fast
runs below ten seconds or greatly extend the CPU-bound native runs.
Backends run serially with a rotated order. Reports retain all repeats,
median of repeat medians, median of repeat p95s, pooled statistics, host samples,
and a stability flag. Rerun the measurement if the repeat medians vary by more
than five percent. GPU pixels remain resident during timing; GPU completion is
included. Lane 1 stage timestamps and sparse camera checks run outside timing.
Camera checks read the same target format and render settings used for timing.
Native is the Rerun 0.38.1 renderer path with per-frame CPU sort and full splat
upload; these timings do not measure the complete viewer.

Parity scores unclipped premultiplied float RGB, alpha and white-composited RGB.
`--save-images DIR` writes float EXRs, PNG previews and amplified differences.
`parity --impl ours --archetype [recording.rrd]` measures native archetype input
precision against the unquantized compute path; without a recording it loads the
PLY through Rerun. A recording must contain one complete splat row.
