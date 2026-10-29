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
target/release/gsplat speed --impl ours,brush,native \
  --ply scene.ply --path orbit:300 --res 1920x1080 --out speed.json
target/release/gsplat parity --impl ours --oracle brush \
  --ply scene.ply --path transforms_test.json --holdout-every 8 --out parity.json
target/release/gsplat parity --impl ours --archetype recording.rrd \
  --ply scene.ply --path cameras.json --save-images evidence --out quantization.json
# Build with cargo build --locked -p gsplat-cli --features probe:
target/debug/gsplat probe recording.rrd --out frames.json --window-size 1920x1080
```

Hexel PLYs (`comment hexels_format 1`; variants v1, v2, v3 and 2D fits) render through
the same `render` and `speed --impl ours` paths (the Brush and native renderers reject them):

```bash
target/release/gsplat render --ply model.ply --camera cameras.json \
  --output-dir renders/test --background 0,0,0 --png16 --surface surface.rrd
target/release/gsplat render --ply fit-2d.ply --output fit.png --png16 --fps fit-fps.json
```

Without `--width/--height`, `render` keeps the camera file's size. `--png16` writes 16-bit
PNGs. `--surface` adds a recording with, per view, the camera and RGB, median and expected
depth, alpha and world normals under `world/camera/<PLY stem>/` (`--surface-name` renames
it). 2D hexel fits take no camera file. `--fps fps.json` then
times the rendered views as upstream Hexels does: 3D scenes with the paper protocol (2 warm-up
passes, 4 trials of 10 passes, median FPS; frames submitted back to back with up to 3 in flight,
the GPU drained whenever all 3 are pending and at the end of each trial, no readback), 2D fits
like its 2D decode timing (10 warm-up renders, 5 trials of 20, each render waited on).

Camera files accept NeRF transforms, Hexels `cameras.json` or `contract.json`
(`geometry.test`, principal point at the image centre as upstream), CameraSpec JSON
arrays, or COLMAP model directories. Speed and parity also accept `orbit:N`; `--res native` retains input
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

The probe needs one 3D view and writes its camera path and a PNG beside the report.
Use `--storage-binding-limit N` to exercise capacity fallback and `--require-motion`
to reject held views. See [architecture](../../docs/architecture.md) for the timing
protocol, precision boundaries, and report provenance.

On Metal, `speed` reports `lane1: null` with `lane1_unavailable_reason`: per-stage GPU timestamps are unreliable. Frame timings remain available; Vulkan and DX12 retain stage timings.
