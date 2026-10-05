# Native replay

Fixture inputs remain available through `replay --clip DIR --frames FILE`.
Build with `--features catalog` to read RoboCap and MSD G2 segments directly:

```sh
cargo build --offline --release -p slam-rs-cli --features catalog
slam-rs replay --catalog "$SLAM_RS_CATALOG_URL" \
  --segment msd-g2__MGO_others__MGO07_mapping_easy --max-framesets 1596 \
  --config resolved-config.json --out poses.csv --summary summary.json
```

Add `gpu-wgpu` for `--lane gpu`. The catalog mode needs no Python or local
fixture. `--config -` reads the resolved config from stdin. All four cameras,
calibration and IMU come from the named segment. The dataset is its `robocap__`
or `msd-g2__` prefix; other rigs are rejected.

The optional catalog build uses Rerun 0.38.1 protocol/chunk crates without
DataFusion and static OpenH264. AV1 uses dav1d 1.2.1 (`libdav1d.so.6`). Provide
its headers through `DAV1D_INCLUDE_DIR` when building. At runtime, put the
library on the loader path or set `SLAM_RS_DAV1D_LIB` to its filename. The Linux slam-rs environments declare dav1d headers. From the workspace root,
`pixi run -e robocap-cross slam-rs-cli-catalog-build` builds the ARM binary with
those headers, the aarch64 C++ compiler, and static libstdc++.

RoboCap selects `left_front,right_front,left,right`, takes raw decoded Y, and
uses the same integer area /3 reducer as robocap-live. This intentionally differs
from the Python fixture's combined color conversion and resize. G2 expands
limited-range Y to full-range gray8 as the Python feed does. The loader checks
codec, resolution, camera matching and IMU timing. It undoes the catalog's
applied time shift for both streams, then uses `Calibration::from_catalog_parts`.

Replay starts at the segment's beginning. `--max-framesets` limits complete
framesets, with at most one 64-chunk fetch batch of encoded video over-read.
Decoding starts at the first keyframe; interior seeking is not supported.
All selected gray8 images are in RAM before the existing replay clock starts.
Budget roughly `framesets * sum(camera width * height)` bytes plus encoded
input and IMU: a 1,596-frame G2 clip uses about 1.83 GiB of gray8. `load_s`
includes catalog fetch, calibration, association, IMU pairing and decoding;
`setup_s`, replay timing, CSV and summary fields retain their existing meanings.

The ignored read-only catalog contract test compares timestamps, IMU and every
pixel against a Python fixture. It requires exact G2 pixels and reports the
RoboCap luma differences:

```sh
SLAM_RS_CATALOG_URL=rerun+http://host:51235 \
SLAM_RS_TEST_FIXTURE=/path/to/fixture-starting-at-frame-zero \
SLAM_RS_TEST_OUTPUT=/tmp/catalog-pixels \
cargo test --offline --release -p slam-rs-cli --features catalog \
  catalog_inputs_match_python_fixture -- --ignored --nocapture
```

It requires `frames.u8` or `frames.u8.xz` and `xz` for the latter. No catalog
registration or server mutation is performed.
