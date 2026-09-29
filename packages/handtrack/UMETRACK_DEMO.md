# UmeTrack keypoint stage demo

`--keypoints umetrack` runs DetNet acquisition, the pretrained UmeTrack network,
our LM fit, and our detection-by-tracking state machine. It writes the same
per-segment `.npz`, metadata, MKPE and tracking metrics as the KeyNet path.

`CropRequest` now carries our crop-planning pose, headset transform, circles,
and native images. The adapter uses upstream 63-point perspective crops for
tracked hands and the frozen validation circle scale for acquisition. Each hand
runs once per frame on its requested views, with both hands in one batch.
The network supports at most two views per hand. Acquisition follows our existing
single-camera round robin; tracking chooses the two views with most visible
landmarks. Predictions are skinned to 21 world landmarks, projected through our
camera model into net pixels, and converted to relative camera distances in
generic-hand millimetres. Finite, in-front, in-image landmarks have weight one;
other landmarks have weight zero. Our LM fit produces the final pose.

Upstream temporal memory resets before every inference. Our fitted pose supplies
the next crop and our tracker owns all tracking state. Crop refinement is rejected
for this backend to preserve one inference per hand/frame. KeyNet behavior is
unchanged.

UmeTrack has no presence head. The tracker runs DetNet on requested cameras and
requires probability greater than `tracker.umetrack_presence_threshold` (0.5).
Any confirmed view resets the miss count, and only confirmed views enter that
frame's fit. Without a confirmation, all usable views remain in the fit for up to
`tracker.umetrack_miss_frames - 1` frames (default two). The third consecutive
miss drops the hand and clears history. Acquisition requires confirmation at once.
Missing or invalid pose output, no usable view, a lost headset pose, non-finite
fitted pose, or a wrist beyond `tracker.max_reach_m` also ends tracking.

Output compatibility: `keynet_sha256` holds the UmeTrack weights digest,
`keynet_views` counts requested keypoint views, and the `keynet` timing includes
crop conversion and UmeTrack inference. `presence` stores raw DetNet confirmation
scores; it can be low during the grace period. `detnet_camera` and `detnet_runs`
still describe round-robin acquisition; confirmation time is included in `detnet`.

SHOW3D uses native 1024×1280 images for UmeTrack. Only DetNet sees the rotated,
resized 640×480 net image. Perspective crops use -90° source-camera roll to match
the clockwise net-image convention. The record, layer writer and blueprint handle
its two cameras under `/world/rig_01`. Ground-truth overlays use confidence >0.1
and a valid headset pose. SHOW3D has no `projections` layer, so standalone export
requires `base` and `hand_pose`, and merges available `hand_mesh`/`projections`.

The catalog reader documents that SHOW3D's official `test` scenes have no hand
labels. Use labelled held-out subjects (`--split val`) for scored demo segments.
Explicit `--segments` selects labelled IDs in the chosen dataset without applying
split, domain, or interaction filters. Unlabelled IDs fail with a named error.

Run from `/home/pablo/handtrack-wt/demo` on the GPU host. Replace each `<IDS>` with
two or three space-separated IDs from that dataset. Use a new run name if settings
change; run identities include the dataset, weights digest and frozen calibration.

```bash
pixi run -e handtrack --frozen python packages/handtrack/tools/run_pipeline.py \
  --name umetrack-synthetic --dataset dataforge-umetrack --domain synthetic \
  --segments <IDS> --detector detnet --keypoints umetrack --device cuda \
  --detnet-weights /home/pablo/handtrack-data/runs/detnet-overnight-1/detnet/best.weights.pt \
  --umetrack-calibration /home/pablo/handtrack-data/umetrack_baseline/reference/calibration.json \
  --output-root /tmp/fleet-artifacts/handtrack/demo-0600

pixi run -e handtrack --frozen python packages/handtrack/tools/run_pipeline.py \
  --name show3d --dataset dataforge-show3d --split val \
  --segments <IDS> --detector detnet --keypoints umetrack --device cuda \
  --detnet-weights /home/pablo/handtrack-data/runs/detnet-overnight-1/detnet/best.weights.pt \
  --umetrack-calibration /home/pablo/handtrack-data/umetrack_baseline/reference/calibration.json \
  --output-root /tmp/fleet-artifacts/handtrack/demo-0600

pixi run -e handtrack --frozen python packages/handtrack/tools/export_layers.py \
  --dataset dataforge-umetrack \
  --run-dir /tmp/fleet-artifacts/handtrack/demo-0600/umetrack-synthetic/known \
  --detnet-dir /tmp/fleet-artifacts/handtrack/demo-0600/umetrack-synthetic/detnet \
  --clips <IDS> --export-name umetrack-synthetic \
  --layers-root /tmp/fleet-artifacts/handtrack/demo-0600/umetrack-synthetic/layers \
  --export-dir /tmp/fleet-artifacts/handtrack/demo-0600/export

pixi run -e handtrack --frozen python packages/handtrack/tools/export_layers.py \
  --dataset dataforge-show3d \
  --run-dir /tmp/fleet-artifacts/handtrack/demo-0600/show3d/known \
  --detnet-dir /tmp/fleet-artifacts/handtrack/demo-0600/show3d/detnet \
  --clips <IDS> --export-name show3d \
  --layers-root /tmp/fleet-artifacts/handtrack/demo-0600/show3d/layers \
  --export-dir /tmp/fleet-artifacts/handtrack/demo-0600/export
```

These export commands do not register or mutate the catalog. The standalone files
are `<export-name>__<segment>.rrd`. They include source video, available GT layers,
the fitted handtrack layer and the dataset-specific blueprint. DetNet-alone layers
are also written under the requested layers root.

The upstream checkout and shim default to
`/home/pablo/handtrack-data/umetrack_baseline/{UmeTrack,shim}`. Override them with
`--umetrack-root` and `--umetrack-shim`. `--umetrack-weights` overrides
`<umetrack-root>/pretrained_models/pretrained_weights.torch`. DetNet requires
`best.weights.pt.sha256` next to the supplied checkpoint. The default calibration
was present when this adapter was built (median scale 0.8733532444680852); a missing
file fails with the `--umetrack-calibration` override in the error.

CPU checks cover faked pose predictions through the real LM fit, detector misses
and reacquisition, single-view confirmation, SHOW3D native pixels and distance
units, two-camera records/metrics/export, and the installed pretrained network on
synthetic inputs. Real catalog clips, GPU/NVDEC execution, accuracy, throughput,
SHOW3D crop-roll quality on real images, and Rerun rendered pixels still need host
validation. No demo metrics or Viewer pixel proof are claimed by those CPU checks.
