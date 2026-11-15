# Provenance of the Caspar adapter patches

All four patches apply, in this order, to colmap/colmap `be5e29168d4aff238409d60424812df66aac919f` (tag 4.2.0).
`SHA256SUMS` hashes all 274 files they touch (verified by the recipe's Step 0 before the CUDA build).

| Patch | Kind | From |
|---|---|---|
| `add-opencv-fisheye-caspar-adapter.patch` | hand-written | fork `caspar-opencv-fisheye` @ c03b1f38 (pablovela5620/pyCuSFM, `packages/pycolmap-caspar-fisheye`, cut by `tools/caspar_fisheye_patch.sh regen`) |
| `add-opencv-fisheye-generated-f32.patch` | Symforce output | same branch: 224 fisheye kernels + 4 aggregates |
| `add-full-opencv-caspar-adapter.patch` | hand-written | branch `caspar-full-opencv` @ 06871fdc + fc4a8e6c (setter names) on top of c03b1f38 |
| `add-full-opencv-generated-f32.patch` | Symforce output | branch `caspar-full-opencv` @ 0cc0355f: 42 FULL_OPENCV kernels + the 4 aggregates |

## FULL_OPENCV (2026-11-15)

Design: upstream COLMAP PR #4657 (fixed-calibration THIN_PRISM_FISHEYE). FULL_OPENCV's 12 parameters do not fit a
tunable merged or split calib in Caspar's 48 KB shared-memory budget, so only two factors are generated, with the
calibration as constant sequential data:

- `full_opencv_split_fixed_focal_and_extra_fixed_principal_point` (pose + point tunable)
- `full_opencv_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point` (point tunable)

`focal_and_extra = [fx, fy, k1, k2, p1, p2, k3, k4, k5, k6]`, `principal_point = [cx, cy]`; the projection mirrors
`FullOpenCVCameraModel` in `src/colmap/sensor/models.h` (rational radial term plus tangential p1, p2). Any other variant
(refined intrinsics, refined `sensor_from_rig`, fixed points) fails with `LOG(FATAL_THROW)` instead of silently dropping
observations. The factors are registered after every existing one, so the solver's constructor only appends arguments.

Generator (run from the root of the branch checkout; the output must sit inside it so clang-format finds `.clang-format`):

```bash
cp <pyCuSFM>/packages/pycolmap-caspar-fisheye/generator/pixi.{toml,lock} .   # symforce==0.12.0 (PyPI), clang-format 21.1.2
pixi run --locked gen "$PWD/.pixi/gen/f32" f32                               # 31 s on pablo-dl-server
```

Regenerating changes more than the FULL_OPENCV files: the generator's SumStore padding-thread fix
(`_add_sum_write_template`, see upstream PR #4611) reaches the stock PINHOLE / SIMPLE_RADIAL kernels too, and symforce
0.12.0 writes a different `CMakeLists.txt`. As for the fisheye patch, only the new files (`kernel_full_opencv_*`,
`kernel_FullOpenCVPose_*`) and the four aggregates (`solver.h`, `solver.cc`, `caspar_mappings.h`, `caspar_mappings.cu`)
are taken; the vendored stock kernels and `CMakeLists.txt` (which globs `*.cu`) stay as upstream ships them.

Cut and checked:

```bash
git diff --full-index c03b1f38 fc4a8e6c -- <the 4 hand-written files>   # -> add-full-opencv-caspar-adapter.patch (after the `# ` header)
git diff --full-index 06871fdc 0cc0355f   # -> add-full-opencv-generated-f32.patch
git diff --name-only be5e2916 fc4a8e6c | grep -v '^\.gitignore$' | grep -v generated/f64/ | xargs sha256sum > SHA256SUMS
# fresh checkout of be5e2916: git apply all four patches; sha256sum -c SHA256SUMS -> 274/274
```

The branch `caspar-full-opencv` lives in a local clone of the fisheye fork (`/tmp/bring-in/colsfm/colmap-fo` on
pablo-dl-server, not pushed); the patches plus this file are the record. Drop both FULL_OPENCV patches when upstream
CASPAR supports FULL_OPENCV.
