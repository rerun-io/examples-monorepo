# Fixed renderer camera paths

Each JSON file contains 300 pinhole `CameraSpec` records for one complete orbit.
The image size is 1920x1080. Poses are row-major camera-to-world transforms in
OpenCV right/down/forward coordinates; focal lengths and principal points are
in pixels. `--res 3840x2160` scales the intrinsics by two and leaves poses unchanged.

Use the same file for every implementation and revision being compared. From the
package directory in an activated environment:

```bash
target/release/gsplat speed --impl ours,brush,native --ply LEGO_PLY \
  --path tests/fixtures/benchmark/lego-orbit.json --res 1920x1080 --out lego.json
target/release/gsplat speed --impl ours,brush,native --ply GARDEN_PLY \
  --path tests/fixtures/benchmark/garden-orbit.json --res 3840x2160 --out garden.json
```

The default generated orbit uses the loaded scene bounds and can frame the scene
differently. These explicit paths make the timing workload reproducible. Each
repeat covers whole cycles after warmup; JSON reports retain cameras, frame
counts, timing conditions, source revision, and dependency fingerprint.
