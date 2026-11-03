# SelfCap SOMA fitting

Fit a shared SOMA body shape and a full-rate upper-body pose to catalog
observations. Inputs are triangulated RTMW or Sapiens2 keypoints, calibrated
cameras, and Quest hand positions and orientations. The fit uses triangulated
ears and soft Quest head rotation. Reports contain training residuals.

## Run

Use `selfcap-run` for the complete conversion and fitting pipeline.
For existing calibrated catalog segments, run from the repository root:

```bash
pixi run -e selfcap-soma --frozen soma-batch \
  --catalog-url rerun+http://127.0.0.1:51235 --dataset selfcap-new \
  --segment selfcap__2025-12-03__d31bf1da__ep001 --output /local/soma
pixi run -e selfcap-soma --frozen soma-xlo \
  --soma-dir /local/soma --dataset selfcap-new --output-dir /local/soma-xlo \
  --segment selfcap__2025-12-03__d31bf1da__ep001
```

`--segments-file` also accepts one segment id per line. Batch writes local files;
`selfcap-run` registers them with `dataforge-swap-layer`. It uses one directory
per dataset and segment, with a lock and resumable checkpoints. `--force` refits.

## Outputs

The `soma` layer holds full-rate upper-body meshes, normals, joints, ears,
camera overlays, the crop vertex ids, and a `soma_quality` property record.
The `soma_xlo` layer keeps the same motion with a smaller mesh for previews.
JSON reports hold calibration, selected shape frames, hand orientation sources,
residuals, jitter, stage times, and checkpoint state. Files publish atomically.

## Assets and checks

`assets.py` pins the SOMA-X model revision and required files. `--assets` supplies
a local copy. Without it, the pinned assets download on first use.
`pixi run -e selfcap-soma-dev --frozen tests` runs the unit tests.
GPU and asset checks are in `tests-integration`; reference checks use `tests-golden`.
