# Catalog-Native MVAPI Prediction Layer Spec

## Goal

Build a new catalog-native MVAPI prediction-layer pipeline inside `packages/mv-api`.
The existing MVAPI full exo/ego pipeline remains untouched and acts as the oracle baseline for comparison.

The new tool mounts the full SimpleCV ExoEgo Forge Rerun catalog, selects one deterministic Assembly101 segment by default, loads exo frames directly through `rerun.experimental.dataloader`, runs the existing MVAPI tracker, writes a prediction `.rrd`, and registers that `.rrd` back into the catalog as a new layer.

## Dependency And Environment Policy

- Pin `simplecv` to latest pushed `origin/main` rev:
  `178479b53d14b9dd4a79b212dd2c5a3be52b4de8`.
- Use a separate Pixi environment from the existing oracle path:
  - `mv-api-catalog`
  - `mv-api-catalog-dev`
- The catalog environments must use the SimpleCV-compatible prerelease Rerun wheel lane:
  - `rerun-sdk[datafusion] == 0.33.0a1+dev`
  - `find-links = https://build.rerun.io/commit/5f732f2/wheels/`
  - `prerelease-mode = allow`
- Keep the existing `mv-api` / `mv-api-dev` environments stable for the current oracle pipeline.
- Use Pixi only. Do not use `pip` or `uv`.

## Package Layout

Add the new pipeline inside the existing `mv-api` package:

```text
packages/mv-api/src/mv_api/api/catalog_prediction_layer.py
packages/mv-api/tools/apps/catalog_prediction_layer.py
packages/mv-api/tests/test_catalog_prediction_layer.py
```

This is a new orchestration path. It may reuse shared MVAPI tracker/model code, but it must not retrofit the current `FullExoEgoApp` pipeline.

## Catalog Behavior

- Mount the full ExoEgo Forge catalog by default through latest SimpleCV catalog APIs.
- Keep full-catalog registration (`datasets=()`), but pre-optimize only the target dataset by default (`optimize_datasets=("assembly101",)`) so validation is not blocked by unrelated datasets.
- Default RRD root:
  `/mnt/8tb/data/exoego-forge-catalog`
- Default catalog port:
  `9988`
- Support connecting to an already-running catalog URL as an override.
- Default inference target is one Assembly101 segment, not all Assembly101 segments.
- The default segment is deterministic:
  select row `id=120` after sorting `assembly101` catalog rows by `sequence_key`.
  Row 0 currently exposes `VideoStream:sample` in the schema but has null sample values; row 120 is the first validated default with non-null AV1 `VideoStream:sample` values for all exo cameras.
- Expose a segment selector override, such as row id or sequence key.
- Preserve the selected catalog segment's exact `recording_id`.
  The prediction `.rrd` must use this `recording_id` so Rerun registers it as a layer on the same segment, not as a new segment.

## Dataloader Behavior

- Load frames directly through `rerun.experimental.dataloader`.
- Do not use SimpleCV's `RRDExoEgoConfig` frame loader for this new path.
- One PyTorch dataloader sample represents one `video_time` timestamp for one Assembly101 segment.
- Each sample contains all selected exo camera images, keyed by camera name.
- Use `torch.utils.data.DataLoader` with:
  - `batch_size=1`
  - `num_workers=0`
- Use `fill_latest_at=True`.
- Drive sampling from a deterministic canonical exo timeline:
  the exo stream with the shortest duration, matching the current SimpleCV/MVAPI latest-at behavior.
- Default `max_frames=10` for smoke-run safety.
- Provide an explicit way to run the full selected segment.

## Schema Discovery And Validation

- Discover exo cameras from the catalog schema.
- Use all discovered calibrated exo cameras, sorted by camera name.
- Do not expose camera subset controls in v1.
- Require every selected exo camera to have:
  - `/world/exo/{cam}/pinhole/video:VideoStream:sample`
  - `/world/exo/{cam}/pinhole:Pinhole:image_from_camera`
  - `/world/exo/{cam}/pinhole:Pinhole:camera_xyz`
  - `/world/exo/{cam}/pinhole:Pinhole:resolution`
  - `/world/exo/{cam}:Transform3D:translation`
  - `/world/exo/{cam}:Transform3D:mat3x3`
- Fail loudly if a selected segment only has `AssetVideo:blob`.
- Fail loudly if calibration is missing for any selected exo camera.
- Do not estimate calibration in v1.
- Do not process ego cameras in v1.

## Inference Behavior

- Reuse:
  - `MultiviewBodyTracker`
  - `MultiviewBodyTrackerConfig`
  - `MVHistory`
- Create new catalog-specific orchestration around those primitives.
- Use every discovered calibrated exo camera for 2D prediction and 3D triangulation.
- Keep the current MVAPI upper-body filtering:
  shoulders/arms plus face plus hands.
- Mask the rest of the COCO133 keypoints consistently with the current oracle path.
- Keep the v1 contract single-person.
- No training.
- No ego reprojection.
- No multi-person output paths.

## Tensor Conversion

The dataloader returns image tensors as `UInt8[torch.Tensor, "3 h w"]` in RGB channel order.
Convert each sample to the tracker contract:

```python
bgr: UInt8[np.ndarray, "h w 3"]
```

Conversion rule:

1. Move channels from CHW to HWC.
2. Convert RGB to BGR.
3. Ensure `np.uint8`.
4. Keep one `bgr_list` entry per sorted exo camera.

## Output Layer

Default layer name:

```text
mvapi_coco133_upper_body_v1
```

Default output root:

```text
packages/mv-api/artifacts/catalog_layers
```

Default output file shape:

```text
packages/mv-api/artifacts/catalog_layers/assembly101/<sequence_key>/mvapi_coco133_upper_body_v1.rrd
```

The output `.rrd` must:

- Use `application_id="assembly101_mvapi_coco133"` or another stable MVAPI catalog application id.
- Use the selected source segment's exact `recording_id`.
- Log on the same `video_time` timeline used for inference.
- Register back into the mounted `assembly101` dataset immediately by default.
- Fail on duplicate layer by default.
- Provide an explicit debug option to write the `.rrd` without registering it.

## Viewer Screenshot Validation

The implementation is not complete until the prediction layer is validated in a real Rerun Viewer session.
Use the `rerun-viewer-validation` skill workflow and native Viewer screenshots.

Validation requirements:

- Create a validation run directory under:
  `packages/mv-api/artifacts/rerun-viewer-validation/<timestamp>/`
- Launch or connect to the catalog/viewer through the Pixi catalog environment.
- Open the selected Assembly101 segment with the newly registered prediction layer enabled.
- Capture screenshots from an actual Rerun Viewer, not only from `.rrd` metadata or logs.
- Capture one screenshot for each exo 2D camera view showing the image and `/world/exo/{cam}/pinhole/pred/mvapi/coco133_uv` overlay.
- Use deterministic viewport sizing, preferably `1920x1080`.
- Use Rerun's native screenshot support, following the pattern from:
  `https://github.com/rerun-io/rerun/blob/main/docs/snippets/all/howto/screenshot.py`
- A one-shot native screenshot is acceptable for direct `.rrd` validation when it can show the required view:
  `rerun --window-size 1920x1080 --screenshot-to <path> <rrd-or-url>`
- For interactive catalog/segment validation, use `rerun.experimental.ViewerClient(...).save_screenshot(...)` after the Viewer is running and settled.
- Write `notes.md` next to the screenshots with:
  - commands used
  - catalog URL or segment URL
  - output `.rrd`
  - layer name
  - screenshot filenames
  - pass/fail summary for each exo camera

The validation pass condition is: every discovered calibrated exo camera has a 2D screenshot where the image is visible and the MVAPI COCO133 prediction overlay is visible in the correct camera view.

## Logged Entities

Use SimpleCV custom confidence types:

- `Points2DWithConfidence`
- `Points3DWithConfidence`
- `confidence_scores_to_rgb`

Reuse MVAPI's existing annotation context helper:

```python
mv_api.api.full_exoego_pipeline.set_annotation_context(...)
```

Log static annotation context into the prediction `.rrd` so it remains useful when opened alone.

Log both 2D and 3D predictions:

```text
/world/pred/mvapi/coco133_xyz
/world/exo/{cam}/pinhole/pred/mvapi/coco133_uv
```

Use per-frame `rr.log(...)` in v1.
Do not optimize to `rr.send_columns(...)` until the catalog-native path is proven.

## CLI And Pixi Tasks

Add a new thin CLI:

```text
packages/mv-api/tools/apps/catalog_prediction_layer.py
```

Add a new Pixi task under the catalog environment, for example:

```bash
pixi run -e mv-api-catalog --frozen mv-api-catalog-prediction-layer
```

Recommended CLI options:

```text
--rrd-root /mnt/8tb/data/exoego-forge-catalog
--catalog-url rerun+http://127.0.0.1:9988
--catalog-port 9988
--assembly101-row-id 0
--sequence-key <optional sequence key override>
--max-frames 10
--output-root packages/mv-api/artifacts/catalog_layers
--layer-name mvapi_coco133_upper_body_v1
--no-register-layer
```

## Tests To Write First

Use TDD for implementation.
Start with tests that do not require the real catalog server or model weights.

Required tests:

- Default segment selection is deterministic from sorted Assembly101 rows.
- Schema discovery finds only exo `VideoStream:sample` fields.
- `AssetVideo:blob`-only exo streams fail loudly.
- Missing `Pinhole` calibration fails loudly.
- Missing `Transform3D` calibration fails loudly.
- RGB CHW torch tensor converts to BGR HWC NumPy correctly.
- Output path construction preserves dataset, sequence key, and layer name.
- Prediction recording uses the source segment `recording_id`.
- Duplicate layer policy is fail-by-default.
- CLI and Pixi task are wired to the new tool, not the existing full exo/ego oracle.
- Viewer validation command/path construction creates one expected screenshot target per discovered exo camera.

Full end-to-end model execution can remain a manual smoke test because it requires the full local catalog and model assets.
The manual smoke test must include the Rerun Viewer screenshot validation described above.

## Goal Prompt

```text
Implement the v1 catalog-native MVAPI prediction-layer pipeline in /home/pablo/0Dev/work/rerun-projects/examples-monorepo.

Use TDD. Do not modify the existing MVAPI full exo/ego oracle pipeline except for reusable imports if necessary. Create a new pipeline/tool inside packages/mv-api that mounts the full SimpleCV ExoEgo Forge Rerun catalog, selects one deterministic Assembly101 segment by default, loads exo frames directly with rerun.experimental.dataloader, runs the existing MultiviewBodyTracker/MVHistory on all discovered calibrated exo cameras, and registers a new prediction layer back into the catalog.

Implement the decisions from packages/mv-api/CATALOG_PREDICTION_LAYER_SPEC.md:
- pin simplecv to git rev 178479b53d14b9dd4a79b212dd2c5a3be52b4de8
- create separate mv-api-catalog / mv-api-catalog-dev Pixi envs using the SimpleCV-compatible prerelease rerun-sdk[datafusion] wheel lane
- full catalog mount by default, but inference only one Assembly101 segment by default
- require VideoStream:sample and catalog calibration; fail loudly otherwise
- DataLoader batch_size=1, num_workers=0, fill_latest_at=True
- all calibrated exo cameras, no camera subset controls
- no ego processing, no training, no calibration estimation, no multi-person support
- reuse MultiviewBodyTracker and current upper-body filtering
- write both 2D exo COCO133 predictions and triangulated 3D world COCO133 predictions
- use SimpleCV custom confidence types and MVAPI annotation context
- write output RRD under packages/mv-api/artifacts/catalog_layers and register it immediately as layer mvapi_coco133_upper_body_v1
- fail on duplicate layer by default
- default max_frames=10
- validate the resulting prediction layer with the rerun-viewer-validation workflow and native Rerun screenshots, including one screenshot for each exo 2D camera view showing the prediction overlay

Add focused tests for schema discovery, validation failures, tensor conversion, deterministic segment/path/layer construction, and CLI/task wiring. Use Pixi only; never use pip or uv. Verify with the relevant pixi test task or direct pixi run command for mv-api-catalog-dev.
```
