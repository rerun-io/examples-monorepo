# OAK / Quest hand projection diagnosis

Date: 2026-10-29. Code inspected: `74e69c8ea1112ca8929d878235f654ae064ce7f6`.
Scope: `selfcap__2025-12-03__d31bf1da__ep001` only.

The subsequent [intrinsics test](oak-quest-d31bf1da-intrinsics.md) adds weighted
focal/principal-point fits and independent mono-left inference.
The latest [visible-finger fit](oak-quest-d31bf1da-dense-hands.md) implements a
gyro-fixed dense-observation option under the revised acceptance criteria.

## Decision

Keep the production calibration and timing unchanged. A large clock error is
not supported. A free mount fit reduces the median against EgoHand, but its
answer depends on which joints are fitted, and it does not consistently reduce
the whole-hand tail. This is not enough evidence for a physical mount correction.

The residual is structured: wrists and knuckles differ more than fingertips,
and it remains when hands are almost still. Random jitter cannot explain that
pattern. The measurements do **not** identify how much comes from Quest's hand
shape/pose estimate, EgoHand's estimate, anatomical landmark definitions, or a
remaining camera calibration error. EgoHand is an independent estimator, not
ground truth. Do not turn this result into a claim that Quest alone is wrong.

No production code, clock, mount, or catalog layer was changed. The conditional
calibration implementation and dev cycle were not run because a correction that
helps the whole hand was not established. This commit records the diagnosis.

## Inputs and checks

Read hand, head, camera and EgoHand streams from `selfcap-post-dev` at
`rerun+http://dgx-spark:51235`. No other segment was queried for measurements.
The RGB stream has 4,368 frames, from 0.04493 to 181.99901 seconds. There are
7,733 finite EgoHand hand observations: 3,370 left and 4,363 right. Matching
finite joint slots give 162,393 pairs. Missing detections remain missing.

Use the existing Quest-to-COCO-21 mapping and EgoHand's SOMA-25-to-21 mapping.
Interpolate Quest hands with a 40 ms gap limit and head translation/rotation
with lerp/slerp. Use all 14 Brown-Conrady coefficients. The diagnostic projection
and production `project_ego` differ by at most 0.00424 px on finite samples.

Re-run Sapiens2 on the original 32 selected RGB frames, with its production
full-frame crop and batch size four. Its times match the RGB observation times
within 1.5e-14 seconds. The fixed-rotation production solver reproduces
**13.45965018 px**, versus stored **13.45965017 px**, with 81.25% of hand groups
kept. The reported calibration score is an undistorted, selected-joint,
trimmed-fit score; it is not a whole-segment distorted-image error.

## Timing and speed

Sweep both parameters independently from -150 to +150 ms, in 5 ms steps:

- Clock change: shift the Quest head and hand streams together relative to RGB.
- Hand latency change: advance only the sampled hand pose relative to the head.

The current RGB residual is **14.858 / 29.979 px** (median / p90).
For the sweep, hold the 162,267 pairs finite at every tested offset fixed;
their baseline is **14.860 / 29.986 px**.
The best clock change is **-5 ms**, giving **14.803 / 30.159 px**.
The best hand latency change is **+5 ms**, giving **14.851 / 30.426 px**.
Neither gives a material improvement. The existing 48 ms latency is not
supported as the main cause of the observed displacement.

Error does grow with image speed, but that alone does not identify lag:

| EgoHand joint speed, px/s | Pairs | Median px | p90 px |
|---|---:|---:|---:|
| 0–20 | 19,433 | 12.15 | 22.37 |
| 20–100 | 70,933 | 13.00 | 23.92 |
| 100–300 | 37,330 | 14.95 | 27.76 |
| 300–1,000 | 25,076 | 22.82 | 51.48 |
| above 1,000 | 4,077 | 25.90 | 49.49 |

The speed is the norm of the central time derivative of EgoHand's image
coordinates. Its own jitter and failures can therefore raise the measured
speed. The sweep, rather than this correlation, is the timing test.

## Static spread and landmark definitions

A stricter near-static mask requires wrist speed below 20 px/s and every finite
fingertip below 40 px/s. It leaves 18,186 joint pairs: **12.12 / 22.28 px**.
Subtract each side/joint's median residual within each two-second window, using
windows with at least eight static samples. The remaining 11,277 pairs have
**1.46 / 3.94 px** residual magnitude. Thus the persistent offset is much larger
than the short-window spread. This is the combined estimator spread, not an
isolated measurement of Quest's absolute tracking accuracy.

The left/right directions of the residual differ. For example, the left wrist
has median displacement (-12.12, +19.34) px; the right wrist has
(+9.48, +22.58) px. At the index knuckle these become (-5.36, +15.80) and
(+5.59, +14.86) px. A uniform image translation cannot remove that pattern.

The mappings have the expected anatomical order. OpenXR places the wrist at
its pivot and distinguishes metacarpal, proximal, intermediate, distal and tip
joints. The existing solver already excludes the wrist and the ambiguous base
slots by fitting indices `2,3,4,6,7,8,10,11,12,14,15,16,18,19,20`.
Correct ordering does not prove that two estimators place a joint at exactly
the same point inside the hand.

On the original Sapiens frames, Sapiens2 and EgoHand themselves disagree by
**10.28 / 24.71 px** over 885 common confident pairs. Quest versus EgoHand on
those pairs is **16.56 / 40.25 px**. Quest versus Sapiens2 is
**15.03 / 44.25 px** over its 960 confident pairs, before the solver's trimming.
A free Sapiens mount lowers its own trimmed fit score to 11.10 px but only
changes whole-segment EgoHand error to **14.18 / 31.69 px**. This also worsens
the tail and is not independent proof of a better mount.

## Viewer evidence

Captured and inspected maximized native Rerun 0.38.1 views, using
`/tmp/selfcap-5090/video/shot_url.py`, a 20-second settle, and isolated viewer
processes. Blue Quest and orange EgoHand skeletons are both visible.

| Cursor, ns | RGB PNG | Left PNG |
|---|---|---|
| 39862079571 | `/tmp/oak-quest-diag/baseline-rgb-39862.png` | `/tmp/oak-quest-diag/baseline-left-39862.png` |
| 4144403454 | `/tmp/oak-quest-diag/baseline-rgb-4144.png` | `/tmp/oak-quest-diag/baseline-left-4144.png` |

The displacement is visible in both cameras. At 4.144 s the left hand is near
the monitor edge and the estimators visibly disagree about its pose; the right
hand is clearer. These images do not justify treating one estimator as truth.

The viewer uses the latest sample at the cursor, not the nearest future frame.
The visible RGB source times are 39.835666282 and 4.128168524 seconds. The latest
Quest samples have ages 17.24 and 4.65 ms at the requested cursors. This matters
when comparing a screenshot with interpolated per-frame measurements.

Across all RGB timestamps, emulating the viewer's latest Quest sample gives
**15.10 / 30.74 px**, versus **14.86 / 29.98 px** when interpolating at the image
timestamp. Sampling contributes some motion error but does not explain the
stable displacement.

## Limits and next discriminating evidence

The mono reference is EgoHand's RGB-derived 3D projected through the same
factory stereo calibration. It is useful for measuring projected disagreement,
but **cannot independently validate the stereo extrinsics**. Agreement of the
two views does not prove those extrinsics are correct. Likewise, agreement with
OpenCV checks projection code, not the physical accuracy of the factory lens.

A small set of manually labelled visible fingertips in both cameras, plus an
independent rigid target or hand geometry reference, would separate optical
calibration from estimator anatomy. Without that evidence, replacing a
gyro-constrained mount with a hand-model fit trades one bias for another.

The current Meta documentation says default hand poses are extrapolated to the
requested display time; its unextrapolated API supplies a separate capture
time. That is a reason to preserve timestamp provenance in future captures,
not evidence that this older recording used that API or needs a new fixed lag.

References checked:

- [OpenXR joint order](https://registry.khronos.org/OpenXR/specs/1.0/man/html/XrHandJointEXT.html)
- [OpenXR hand-tracking specification](https://registry.khronos.org/OpenXR/specs/1.1/html/xrspec.html)
- [Meta unextrapolated hand poses](https://developers.meta.com/vr/documentation/native/android/native-openxr-hand-tracking-unextrapolated-poses/)
- [NVIDIA EgoHand inputs and outputs](https://nvidia.github.io/IsaacCapture/main/ego_post_processing/egohand.html)

## Reproduction artifacts

Scratch root: `/tmp/oak-quest-diag`. Catalog streams are cached as Parquet;
no raw capture files were reparsed and no npy/npz files were written.

- `fetch.py`, `fetch_ego.py`: bounded segment and entity reads.
- `inputs.py`: cached streams, clock mapping, camera and joint projection.
- `sapiens.py`: original selected-frame calibration reproduction.
- `full_fit_filtered.py`: all-correspondence mount fits with original hand gates.
- `joint_all_filtered.py`: joint mount/clock/latency fits with the same gates.
- `camera_metrics.py`, `additional.py`, `cursors.py`: detailed residual checks.
- `plot.py`, `residuals.png`: time, lag and joint plots.
- `common_sweep.py`: timing scores on one fixed finite-pair mask.

The exploratory point-subsampled fits in `analyze.py` are superseded by the
all-correspondence fits. Do not use their mount values. The committed numerical
evidence below uses all eligible training correspondences, without point
subsampling. Re-run scripts with the existing selfcap-curate environment and
this worktree's `packages/selfcap-curate` on `PYTHONPATH`.

## Numerical evidence

All pixel values below are in the original 1280 × 720 image coordinates, not the maximized viewer coordinates.

### Camera and joint groups

| Camera | Group | Pairs | Median px | p90 px | Median dx, dy px |
|---|---|---:|---:|---:|---|
| RGB | all | 162,393 | 14.86 | 29.98 | -0.81, +6.00 |
| RGB | wrist | 7,733 | 24.48 | 37.76 | +5.61, +21.61 |
| RGB | mcps | 38,665 | 19.88 | 31.13 | +6.92, +10.05 |
| RGB | tips | 38,665 | 12.58 | 28.63 | -6.59, +3.70 |
| RGB | intermediate | 77,330 | 12.01 | 28.02 | -0.55, +4.74 |
| Left, derived reference | all | 162,393 | 11.40 | 21.97 | -2.99, +4.42 |
| Left, derived reference | wrist | 7,733 | 17.37 | 26.97 | +2.40, +15.42 |
| Left, derived reference | mcps | 38,665 | 14.18 | 22.45 | +3.47, +7.28 |
| Left, derived reference | tips | 38,665 | 10.87 | 21.04 | -7.61, +2.79 |
| Left, derived reference | intermediate | 77,330 | 9.18 | 21.20 | -2.91, +3.50 |
| Right, derived reference | all | 162,393 | 10.57 | 21.43 | +1.81, +4.32 |
| Right, derived reference | wrist | 7,733 | 17.71 | 26.34 | +5.07, +15.62 |
| Right, derived reference | mcps | 38,665 | 14.52 | 22.51 | +6.26, +7.30 |
| Right, derived reference | tips | 38,665 | 8.05 | 20.34 | -1.76, +2.61 |
| Right, derived reference | intermediate | 77,330 | 8.92 | 19.63 | +2.15, +3.38 |

### Held-out mount fits

Fit even 10-second blocks and score odd blocks (fold 0), then reverse (fold 1). Drop 200 ms around block boundaries. Use every eligible training correspondence, the production tracking gates, and Cauchy loss with scale 6 px. Scores below use all 21 hand joints, including joints excluded from fitting. `mapped` is the existing 15-joint subset.

| Fitted joints | Fold | Training pairs | Baseline median / p90 px | Refit median / p90 px | Mount translation mm | Mount rotation deg |
|---|---:|---:|---:|---:|---:|---:|
| mapped | 0 | 56,175 | 14.96 / 30.60 | 13.16 / 31.76 | 16.83 | 1.43 |
| mapped | 1 | 55,605 | 14.82 / 29.55 | 13.05 / 30.76 | 20.18 | 1.77 |
| all | 0 | 78,645 | 14.96 / 30.60 | 12.82 / 30.46 | 30.80 | 2.82 |
| all | 1 | 77,847 | 14.82 / 29.55 | 12.70 / 30.35 | 32.99 | 2.98 |
| tips | 0 | 18,725 | 14.96 / 30.60 | 14.82 / 36.18 | 23.19 | 1.23 |
| tips | 1 | 18,535 | 14.82 / 29.55 | 14.14 / 34.47 | 20.73 | 1.05 |
| wrist_mcp | 0 | 22,470 | 14.96 / 30.60 | 15.75 / 33.11 | 41.87 | 3.70 |
| wrist_mcp | 1 | 22,242 | 14.82 / 29.55 | 15.52 / 33.20 | 44.36 | 4.02 |

The fitted transforms vary with the chosen anatomy: roughly 17–20 mm for the existing subset, 21–23 mm for tips, and 42–44 mm for wrists/knuckles. This is evidence of a fit tradeoff, not a single identified mounting error.

### Joint mount and timing fits

Use the same 15 mapped joints and alternating blocks. The timing search is bounded at ±150 ms. Score on a common finite mask across the 5 ms sweep grid. A positive clock delta moves both Quest streams later. A positive latency delta advances the sampled hand pose only.

| Fold | Fitted timing | Held-out median / p90 px | Clock delta ms | Hand latency delta ms | Mount translation mm | Mount rotation deg |
|---|---|---:|---:|---:|---:|---:|
| 0 | mount | 13.16 / 31.75 | +0.00 | +0.00 | 16.83 | 1.43 |
| 0 | clock | 13.09 / 31.85 | -2.75 | +0.00 | 17.28 | 1.47 |
| 0 | latency | 13.15 / 31.75 | +0.00 | +0.17 | 16.88 | 1.44 |
| 0 | both | 13.13 / 31.50 | -10.45 | -8.07 | 15.96 | 1.33 |
| 1 | mount | 13.05 / 30.78 | +0.00 | +0.00 | 20.18 | 1.77 |
| 1 | clock | 13.00 / 30.76 | -1.53 | +0.00 | 20.54 | 1.80 |
| 1 | latency | 13.05 / 30.77 | +0.00 | -0.23 | 20.08 | 1.76 |
| 1 | both | 13.01 / 30.28 | -9.65 | -8.08 | 18.84 | 1.64 |

Joint fitting adds little beyond a free mount. When both timing variables move, they compensate for one another: approximately -10 ms clock and -8 ms hand latency. This is not evidence for changing either production constant.

### Every joint

Each entry is median / p90 px, pooled across anatomical sides. Non-thumb MCP slots correspond to the OpenXR proximal-joint centers.

| COCO hand index | Joint | RGB | Left, derived | Right, derived |
|---:|---|---:|---:|---:|
| 0 | wrist | 24.48 / 37.76 | 17.37 / 26.97 | 17.71 / 26.34 |
| 1 | thumb base | 19.68 / 26.76 | 15.20 / 20.30 | 13.09 / 18.22 |
| 2 | thumb proximal | 18.57 / 25.89 | 14.43 / 19.88 | 12.40 / 17.82 |
| 3 | thumb distal | 23.79 / 33.24 | 18.47 / 25.15 | 15.60 / 23.59 |
| 4 | thumb tip | 13.67 / 27.70 | 11.16 / 20.21 | 8.83 / 19.61 |
| 5 | index base | 17.37 / 30.05 | 12.60 / 21.81 | 12.71 / 21.71 |
| 6 | index intermediate | 12.67 / 24.29 | 9.78 / 17.66 | 8.91 / 17.54 |
| 7 | index distal | 13.54 / 24.73 | 11.12 / 18.38 | 8.91 / 17.58 |
| 8 | index tip | 12.01 / 27.33 | 10.88 / 20.78 | 7.28 / 18.47 |
| 9 | middle base | 19.65 / 32.18 | 13.71 / 23.26 | 14.75 / 23.06 |
| 10 | middle intermediate | 7.19 / 23.84 | 5.52 / 17.72 | 6.11 / 16.75 |
| 11 | middle distal | 9.50 / 25.88 | 8.48 / 18.86 | 6.18 / 18.28 |
| 12 | middle tip | 14.64 / 30.00 | 12.75 / 22.54 | 9.46 / 21.53 |
| 13 | ring base | 20.25 / 31.95 | 13.93 / 22.96 | 15.42 / 23.01 |
| 14 | ring intermediate | 10.16 / 26.95 | 6.83 / 19.69 | 8.82 / 19.29 |
| 15 | ring distal | 7.34 / 26.97 | 6.87 / 19.55 | 5.47 / 19.19 |
| 16 | ring tip | 10.96 / 31.04 | 9.96 / 22.43 | 6.99 / 22.43 |
| 17 | little base | 22.42 / 34.79 | 15.36 / 24.98 | 17.16 / 25.11 |
| 18 | little intermediate | 10.94 / 26.46 | 7.23 / 19.45 | 9.50 / 19.18 |
| 19 | little distal | 8.78 / 28.28 | 6.91 / 20.63 | 7.38 / 20.34 |
| 20 | little tip | 10.89 / 28.63 | 9.69 / 20.47 | 6.78 / 21.07 |

### Residual over time

Each entry is median / p90 px. Bins are on video_time.

| Interval s | RGB | Left, derived | Right, derived |
|---|---:|---:|---:|
| 0–10 | 15.18 / 33.19 | 11.31 / 24.00 | 11.01 / 23.88 |
| 10–20 | 15.55 / 33.80 | 11.90 / 24.66 | 10.66 / 23.73 |
| 20–30 | 14.93 / 30.31 | 11.48 / 22.00 | 10.49 / 21.54 |
| 30–40 | 15.32 / 30.52 | 11.79 / 22.22 | 10.70 / 22.05 |
| 40–50 | 14.59 / 29.50 | 11.32 / 22.18 | 10.37 / 20.76 |
| 50–60 | 15.46 / 32.77 | 12.08 / 24.54 | 10.59 / 22.93 |
| 60–70 | 14.69 / 28.90 | 11.30 / 21.88 | 9.94 / 20.42 |
| 70–80 | 14.93 / 31.64 | 11.64 / 23.08 | 10.48 / 22.63 |
| 80–90 | 14.19 / 28.49 | 11.09 / 20.99 | 10.26 / 20.62 |
| 90–100 | 14.33 / 30.77 | 11.08 / 22.38 | 10.28 / 21.84 |
| 100–110 | 15.32 / 30.20 | 11.50 / 21.98 | 11.02 / 21.74 |
| 110–120 | 14.47 / 28.84 | 11.13 / 21.30 | 10.30 / 20.66 |
| 120–130 | 14.55 / 28.29 | 11.14 / 20.63 | 10.37 / 20.65 |
| 130–140 | 14.56 / 27.97 | 11.34 / 20.62 | 10.61 / 20.23 |
| 140–150 | 14.89 / 29.29 | 11.38 / 21.79 | 10.62 / 20.91 |
| 150–160 | 14.82 / 30.71 | 11.47 / 22.59 | 10.62 / 21.91 |
| 160–170 | 14.79 / 27.37 | 11.09 / 19.49 | 10.70 / 19.90 |
| 170–180 | 14.80 / 28.23 | 11.28 / 20.16 | 10.86 / 20.60 |
| 180–190 | 15.28 / 29.82 | 11.05 / 20.90 | 10.94 / 21.82 |

The per-block median is stable across the three-minute segment. The temporal and joint curves are plotted in `/tmp/oak-quest-diag/residuals.png`.
