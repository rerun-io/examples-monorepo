# gsplat-bench

Build with `pixi run -e gsplat-rust-renderer-dev --frozen gsplat-bench-build`.
Run the binary in `target/release/` from this package directory:

```sh
taskset -c 8-15,24-31 target/release/gsplat-bench speed \
  --impl brush,ours,ours-old,native --ply scene.ply --path orbit:300 \
  --res 1920x1080 --out speed.json
target/release/gsplat-bench parity --impl ours --oracle brush \
  --ply scene.ply --path test-views:transforms_test.json --out parity.json
```

Choose a single CCD's CPU list from `lscpu -e`; the example is for the 5090 host.
Never use cores 6-7 there. The process and its GPU helper threads inherit affinity.
On Linux, timing requires NVIDIA telemetry and waits for ten quiet one-second samples:
GPU below 5%, one-minute host load below 25% of logical CPUs. After 90 minutes per case
(one deadline shared by all backends, repeats and retries),
CCD1 runs may proceed with `loaded_host: true` and their measured load recorded. Each repeat warms
for a full path and five seconds, then measures whole passes for at least two
passes and ten seconds. Implementation order rotates each repeat. Outlying
repeats get at most two retries; all attempts remain in JSON. Headline statistics
are median of repeat medians and median of repeat p95s; pooled values are secondary.
On macOS, sysctl supplies load/core counts and ioreg supplies GPU utilization.
Admission requires ten samples with GPU utilization at most 20% and one-minute
load below 4; missing telemetry blocks admission. Affinity is unpinned and
`desktop_baseline_gpu_pct` records the measured desktop GPU baseline.
Archive builds may set `GSPLAT_SOURCE_SHA`; absent Git and override, provenance is
`unknown`. Checkout builds otherwise record `git describe --always --dirty`.

CameraSpec uses row-major world-from-camera, OpenCV +x right/+y down/+z forward.
Paths: `orbit:N`, `held`, `test-views:FILE`, `colmap:SPARSE_DIR` (binary COLMAP),
`colmap-test:SPARSE_DIR` (filename-sorted every-eighth-image holdout), or
`specs:FILE` (strict CameraSpec JSON array). `--res native` retains input size.
Orbit framing accepts `--center x y z`, `--radius x y`, `--elevation z`, and `--orbit-up x y z` (default +Z).

Lane 2 waits for each frame's GPU completion without pixel transfer. Separate
lane-1 diagnostics use device timestamps for Brush, eight stages of ours, and six
stages of ours-old. Add `--profile-orbit` to replay the exact camera sequence from
each selected wall-time repeat for ours and Brush, after separate admission and
warmup. The GPU headline takes each frame's stage sum before taking medians.
Core timestamps bracket actual compute passes. Brush's `TimingMethod::Device`
spans its first through last pass, including the mid-frame host count-readback
gap; it does not measure pure active-kernel time.

Each profile frame also records queue submits and blocking device polls from a
separate normal render with API event counting enabled. Logging is disabled for
wall and GPU timing. `readback_to_submit_ms` measures the host interval from the
last mapped count-buffer access to the next submit; it excludes the submit call
and preceding readback latency. It is a diagnostic with logging overhead. Async
count and completion waits need not call blocking `Device::poll`, so the poll
count alone is not a synchronization count.
Parity scores in-memory RGBA floats: RGB on black, alpha, and RGB over white.
Old/native targets are intrinsically byte formats; Brush Float remains unclipped.
Worst-five EXRs preserve scored values; PNGs are previews, never metric inputs.
`--save-images DIR` retains every scored pair. Unsupported old/native lenses fail.

`parity --impl ours-archetype` measures native archetype input loss. It reads the
PLY through Rerun's `GaussianSplats3D` (RGBA8 and f16 SH), then uses the viewer's
conversion and the shared compute renderer. Add `--archetype-rrd recording.rrd`
to read the actual Python tool output (one complete native splat row) instead.
Its default oracle is `ours` with
the original f32 PLY; `--oracle brush` is also available. Reports include mean
and minimum RGB PSNR. For the staged datasets, use
`--path test-views:transforms_test.json` for all 200 Lego views and
`--path colmap-test:sparse/0 --res 1297x840` for Garden's images_4 test split.
This implementation is available only for parity, not speed runs.

`pixi run -e gsplat-rust-renderer-dev --frozen gate` includes Rust and GPU checks.
`tests-integration` reports missing external fixtures as skips; set
`GSPLAT_TEST_PLY`, `GSPLAT_TEST_CAMERAS`, `GSPLAT_TEST_GT`, and `GSPLAT_TEST_COLMAP`
to use staged Lego and garden assets. JSON reports carry build and lock provenance.
