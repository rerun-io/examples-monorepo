# Sampled EgoHand OAK calibration

`selfcap-calibrate` and `selfcap-post` now default to EgoHand for OAK alignment.
The exo-phone Sapiens alignment path is unchanged. `--oak-method sapiens` selects
the prior 32-frame OAK path for comparison; `--oak-free-rotation` remains an
explicit alternative to the default gyro-fixed fit.

## Inputs and sampling

Both SelfCap environments compose `posekit-egohand`. The local asset directory
is configured with `--oak-egohand-assets PATH` (default `data/egohand` relative
to the command's working directory). Its layout matches the EgoHand package:
`weights/detector/hand_detector.onnx`, `weights/egohand/egohand_s060000.ckpt`,
`weights/soma/`, cached `engines/`, and `dinov3-bf16.onnx`. Setup reads local
assets; it does not download them. The dev-cycle driver accepts `EGOHAND_ASSETS`
and passes it to this option.

Per episode, the sampler projects Quest hands through the current OAK camera
pose and full 14-coefficient lens at every source RGB timestamp. A hand is
eligible only when all 15 mapped finger joints are tracked, inside the image,
and moving at no more than 300 px/s. A frame is eligible if either hand is.
Select 64 frames uniformly without replacement with fixed seed 20261030, or all
eligible frames if fewer exist. The sorted indices use the existing NVDEC
picked-frame decoder, now created and destroyed on its own thread.

A fresh calibration has no solved camera translation. In that case the
sampling pose uses the gyro rotation and zero translation from the Quest rig;
if a calibrated OAK pose is already present, the sampler uses it. This bootstrap
only selects observations; the weighted fit solves the final translation.

RT-DETR runs independently on each selected frame. Identity assignment is
reset for each frame; there is no reuse of a prior frame's boxes or labels.
The eligible hands are reconstructed by EgoHand in batches of eight frames.
Box confidence is copied to each joint. Missing/ineligible hands stay missing.

The fit keeps the existing mapped finger subset, 48 ms hand latency, and gyro
rotation. It uses Cauchy loss with scale 6 px and one 30 px trimming pass. The
new path does **not** estimate speed from sparsely sampled EgoHand neighbours:
Quest's dense projected speed already supplies the gate. The precomputed
`--oak-observations` path retains its dense image-speed gate for replay.

## Layers and worker memory

The newly computed source is a separate property-free layer:

| Data | Path / entity | Clock |
|---|---|---|
| Sampled EgoHand observations | `calib/egohand/<segment>.rrd` | `video_time` |
| Joint pixels and box confidence | `/world/rig_04/cam_00/pinhole/pred/egohand_calib/coco133_uv` for this episode | temporal |
| COCO labels/links | parent `.../pred/egohand_calib` | static |

Register the file as `egohand_calib`, distinct from full-segment reconstruction
layers named `egohand`. These sparse calibration samples are hidden by default
in SelfCap camera panes, as Sapiens calibration detections are, to avoid a stale
held sample being mistaken for a current-frame prediction.

The calibrate stage releases Sapiens models before EgoHand when they are no
longer needed. EgoHand detector, backbone, SOMA model and decoded frames are
released per episode, including failure paths; the successful d31bf1da run
returned to 32 MiB of Torch allocations before leaving the episode.

## d31bf1da validation only

Segment: `selfcap__2025-12-03__d31bf1da__ep001`. No other real segment was tested.
The sampler selected 64 of 4,247 eligible frames and produced 80 hand
observations. The sampled full-episode translation differs from the previous
all-frame gyro-fixed fit by **3.10 mm**.

For a strict test, split the actual 64 observations by alternating 10-second
blocks, with a 200 ms guard, fit one half and score the other half's dense
reference observations. This leaves **23 / 38 training frames** in the folds.
The independent mono reference is the previously inferred OAK-left stream.
It is never used for fitting. All scores below use mapped finger joints.

| Fit | Held-out RGB median / p90 px | Independent mono median / p90 px |
|---|---:|---:|
| All-frame reference | 9.28 / 18.79 | 14.62 / 24.39 |
| Sampled EgoHand | 9.77 / 19.34 | 14.64 / 24.06 |

The RGB median is 0.49 px higher in this strict split, rather than the roughly
0.3 px anticipated from the earlier 64-*training*-frame experiment. That earlier
experiment used 64 observations per fold, not 64 for the whole episode. The
sampled result still improves over the old Sapiens calibration (11.00 / 21.42 px
RGB, 19.44 / 30.02 px mono). There was no seed search or change after scoring.

Cached-engine calibration benchmarks used separate fresh workers over the same
cached aligned base and exo Sapiens layer. Timers include model/engine setup,
video decoding, inference, fitting and output writing; process startup/import
is outside the timer. Peak GPU usage was polled per worker PID at 200 ms.

| OAK source | Calibrate wall seconds | Peak GPU MiB |
|---|---:|---:|
| Sapiens, 32 picks | 7.84 | 5957 |
| EgoHand, 64 seeded samples | 24.32 | 8331 |

The complete dev cycle took 56.0 s processing (calibrate 19.81 s), plus 1.0 s
fetch and 2.1 s publication. Its EgoHand setup/inference timer was 17.09 s. Only
this episode's layers in `selfcap-post-dev` were replaced. The driver at
`/tmp/selfcap-5090/dev_cycle.py` adds the new detection layer when present and
publishes under a unique `runs/<tag>/` path. During screenshot checks the shared
segment was replaced with a Sapiens result (source `sapiens`, 13.47 px), and open
viewers reported missing/invalid lazy chunks. The completed EgoHand output was
re-registered from its own run paths; the final source is `egohand`, 9.27 px. No
extra inference or additional segment was run.

Targeted checks: 53 SelfCap tests passed, four integration tests deselected;
changed-file Ruff and targeted type checks passed. The new tests cover seeded
sampling, visibility/tracking/speed exclusion, per-frame detection, score and
joint mapping, decoder thread ownership, the default calibrate path, and sparse
layer identity/timestamps/confidence. No full gate was run.

The local merge of posekit `775413837` was clean (`36eb093a`). The shared runtime
feature supplies Pillow <13, python-dateutil <3, AnyIO <4.15, py-soma-x and Warp;
no Gradio pin changed. Both SelfCap environments installed successfully and the
whole-workspace `pixi lock --check` passed.

## Pixel evidence

The four maximized native OAK views were captured from the updated dev catalog:

- `/tmp/egohand-calib/rgb-39862.png`
- `/tmp/egohand-calib/left-39862.png`
- `/tmp/egohand-calib/rgb-4144-final.png`
- `/tmp/egohand-calib/left-4144-final.png`

All four images were inspected. At 39.862 s, the blue Quest fingers follow the
visible fingers in RGB and mono-left; some fingertip and wrist offsets remain.
At 4.144 s, the right fingers likewise follow the visible hand in both cameras.
The left hand is blurred and partly hidden behind the microwave door, so that
hand is not useful pixel evidence. Initial 4.144 s captures lacked the blue
overlay while the shared registration changed; the `-final` captures above were
retaken after isolated re-publication and contain it. The final viewer logs have
no chunk-load warnings.

Reproduction artifacts live in `/tmp/egohand-calib`: `bench_stage.py`,
`monitor.py`, `score.py`, benchmark logs and memory samples, `accuracy.json`,
`dev-cycle.log`, and `catalog-check.log`. No new npy/npz output was written.
The NPZ files read by the model are upstream SOMA assets.
