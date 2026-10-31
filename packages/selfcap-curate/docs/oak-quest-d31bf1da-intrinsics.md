# OAK intrinsics hypothesis: d31bf1da, episode 1

Follow-up to the mount/timing diagnosis. Scope is only
`selfcap__2025-12-03__d31bf1da__ep001` in `selfcap-post-dev`.
Quest 3D is treated as the trusted reference; EgoHand 2D is the image observation.

**No C/D fit passes the requested test.** Intrinsics freedom reduces the RGB
median, but does not collapse the wrist-versus-tip residual. The RGB correction
also makes independent mono-left error much worse. No solver option, camera
calibration, clock, or catalog layer was changed; no dev cycle was run.

## Protocol

- RGB: all 4,368 frames. Recover RT-DETR box scores with the same posekit code
  and cached detector; use one score per hand, not a claimed per-joint confidence.
  There are 7,743 scored boxes; median score 0.881, minimum 0.500. The existing
  RGB reconstruction supplies 7,733 finite hands and 162,393 joint pairs.
- Apply the original Quest tracking gates, remove EgoHand joint image speeds
  above 300 px/s, and require finite pairs over the ±150 ms clock search. This
  leaves **127,612** pairs before the held-out boundary guard, **122,728** after it.
- Fit even 10-second blocks and score odd blocks, then reverse. Exclude 200 ms
  on each side of block boundaries. Use every eligible pair; no point subsampling.
- Minimize `sqrt(box_score) * (projected - observed)` with Cauchy loss, scale
  6 px. Fit once, discard training pairs with unweighted Euclidean residual
  above 30 px, then refit once. Test pairs are never trimmed using fit residuals.
- A = today's calibration. B = six-DoF mount. C = B plus common focal scale and
  independent cx/cy corrections, with all 14 distortion coefficients fixed.
  D = C plus a common Quest head/hand clock shift. Search bounds: focal scale
  0.7–1.3, principal-point correction ±150 px, clock ±150 ms. No bound was hit.
- B was rerun with the new score, speed and trimming rules, rather than copying
  the previous unweighted result. A–D therefore use identical held-out masks.
- Independent mono: decode every third OAK-left frame, repeat grey into three
  channels, and run EgoHand with **that camera's own 14-coefficient lens**.
  There are **1,456 frames / 2,723 reconstructed hands**. The common finite,
  tracking and speed mask has 47,102 joint pairs, or **45,334** after the guard.
  No mono point enters any fit. The RGB focal/principal-point correction is
  never applied to the mono lens: only the fitted physical mount and clock carry
  over to the mono projection.

## Main result

The table uses the all-joint training subset. Each entry is **median / p90 px**,
pooled over both held-out folds without double-counting. All pixels refer to the
native 1280×720 image. Knuckles = slots 1,5,9,13,17; mid = 2,3,6,7,10,11,14,15,18,19;
tips = 4,8,12,16,20. The wrist is slot 0.

| Fit | RGB all | Wrist | Knuckles | Mid | Tips | Independent mono all |
|---|---:|---:|---:|---:|---:|---:|
| A | 13.47 / 24.87 | 24.26 / 32.93 | 19.31 / 26.83 | 10.53 / 22.60 | 11.59 / 18.42 | 19.34 / 30.55 |
| B | 10.92 / 24.34 | 17.63 / 28.41 | 15.51 / 27.54 | 8.85 / 21.05 | 9.65 / 21.40 | 17.33 / 27.57 |
| C | 10.48 / 24.19 | 24.81 / 36.71 | 15.15 / 26.37 | 8.29 / 19.07 | 9.56 / 19.84 | 49.21 / 66.65 |
| D | 10.32 / 24.26 | 24.27 / 36.20 | 15.01 / 26.39 | 8.11 / 19.01 | 9.54 / 19.85 | 52.83 / 68.79 |

C leaves the wrist at 24.81 px and tips at 9.56 px. D leaves them at 24.27 and
9.54 px. Across the four joint subsets, C's mono median ranges from 49.21 to
92.35 px, versus 19.34 px for A. D's range is 52.83–95.04 px. Thus neither an
all-joint nor a subset fit supplies the required cross-camera improvement.

## Fitted parameters and retained correspondences

The entries below are per fold. cx/cy are final pixel coordinates, not offsets.
Clock is the delta from today's Quest head/hand offset; positive moves both
streams later. Mount deltas are measured from today's mount. The full six-DoF
matrices, B parameters, group scores and subset comparisons are in the adjacent
`oak-quest-d31bf1da-intrinsics-metrics.json`.

| Subset | Fold | Fit | Focal s | cx | cy | Clock ms | Mount mm / deg | Retained / initial training pairs |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| mapped | 0 | C | 0.7935 | 620.55 | 409.34 | +0.00 | 121.62 / 5.39 | 43,615 / 44,283 |
| mapped | 0 | D | 0.7927 | 619.86 | 417.58 | -28.67 | 122.14 / 5.83 | 43,575 / 44,283 |
| mapped | 1 | C | 0.7825 | 618.03 | 400.69 | +0.00 | 127.28 / 4.74 | 42,421 / 43,183 |
| mapped | 1 | D | 0.7818 | 618.29 | 409.84 | -21.23 | 127.49 / 5.32 | 42,361 / 43,183 |
| all | 0 | C | 0.8783 | 583.24 | 366.76 | +0.00 | 81.85 / 1.20 | 59,534 / 62,130 |
| all | 0 | D | 0.8912 | 574.48 | 371.53 | -29.73 | 79.23 / 0.88 | 59,570 / 62,130 |
| all | 1 | C | 0.8770 | 574.83 | 361.89 | +0.00 | 84.83 / 1.11 | 57,842 / 60,598 |
| all | 1 | D | 0.8883 | 567.96 | 366.37 | -21.90 | 82.11 / 0.76 | 57,848 / 60,598 |
| tips | 0 | C | 0.9062 | 743.43 | 414.28 | +0.00 | 65.30 / 9.17 | 14,458 / 14,733 |
| tips | 0 | D | 0.9052 | 743.18 | 417.58 | -25.31 | 66.33 / 9.14 | 14,455 / 14,733 |
| tips | 1 | C | 0.8951 | 754.69 | 402.51 | +0.00 | 69.86 / 9.93 | 13,994 / 14,340 |
| tips | 1 | D | 0.9040 | 755.94 | 411.13 | -21.32 | 65.23 / 10.17 | 13,968 / 14,340 |
| wrist_mcp | 0 | C | 0.8614 | 595.23 | 418.38 | +0.00 | 74.67 / 4.31 | 17,005 / 17,847 |
| wrist_mcp | 0 | D | 0.8660 | 591.84 | 423.71 | -34.26 | 74.25 / 4.34 | 16,970 / 17,847 |
| wrist_mcp | 1 | C | 0.8851 | 555.49 | 431.09 | +0.00 | 74.51 / 5.11 | 16,717 / 17,415 |
| wrist_mcp | 1 | D | 0.8946 | 551.16 | 424.90 | -23.96 | 72.03 / 4.38 | 16,679 / 17,415 |

For all-joint fitting, B retains 59,455/62,130 and 57,587/60,598 training pairs.
C retains 59,534/62,130 and 57,842/60,598. D retains 59,570/62,130 and
57,848/60,598. Different subsets imply very different principal points and
mounts: the C cx estimates range from 555.49 to 754.69 px. Similar values across
the two temporal folds do not rescue this disagreement across anatomy or the
failed mono result.

## Calibration provenance

The frozen base's `/world/rig_04:raw_calibration_json` contains the producer's
`CAM_A` / `center` record. Its lens is:

- fx = 765.045349, fy = 764.663025, cx = 635.092285, cy = 356.912048.
- Capture resolution, active array and pixel-array fields all say 1280×720.
- The record contains the same 14 coefficients logged on the base pinhole.

`dataforge/datasets/selfcap_raw.py:read_calibration` copies the lens values,
capture resolution and 14 coefficients. `dataforge/datasets/selfcap.py` selects
the `center` entry for RGB and logs those values as the pinhole and
`DistortionCoefficients`, narrowing to float32. It applies **no extra scale or
crop correction**. This correctly preserves the supplied 1280×720 calibration;
it cannot establish whether the capture producer computed that calibration
correctly.

The actual capture-side writer and its `getCameraIntrinsics` call were not
found in the available capture repositories. The record does not preserve the
native EEPROM dimensions or API arguments, so it cannot verify the suggested
4056-versus-3840 crop. Do not infer native sensor dimensions from its 1280×720
metadata fields.

DepthAI's API supports output width/height, crop corners and `keepAspectRatio`.
The inspected `CalibrationHandler.cpp` scales K to the requested output size,
shifts its principal point for aspect-ratio cropping, and subtracts the explicit
crop origin. Correct arguments can therefore account for the crop; the API's
existence is not proof that this recording used them correctly.

A 4056/3840 mismatch predicts a factor **1.05625** (or reciprocal **0.94675**,
depending on direction). C instead fits **0.8770–0.8783** on all joints; across
all subsets it fits 0.7825–0.9062. This is not a recovered 5.6% crop correction.
The tested model does not support that proposed explanation.

References:

- [DepthAI intrinsics API](https://docs.luxonis.com/software-v3/depthai/api/cpp)
- [DepthAI calibration implementation](https://github.com/luxonis/depthai-core/blob/main/src/device/CalibrationHandler.cpp)
- Local inspected implementation: `/home/pablo/0Dev/repos/depthai-core/src/device/CalibrationHandler.cpp`.

## Visual and numerical checks

Posekit used commit `775413837`. Both mono hands are detected in the inspected
4.166, 39.915 and 90.039 second samples; these are the nearest sampled frames,
not exact requested RGB timestamps. The resulting joints lie on the visible
mono hands, with the expected uncertainty at the partly occluded left hand in
the first sample. The independent observations are usable for rejecting the
large C/D transfer errors.

- Mono proof: `/tmp/oak-intrinsics-diag/mono-proof.png`.
- Individual decoded samples: `mono-4.144.png`, `mono-39.862.png`, `mono-90.000.png`
  under the same directory.
- The C analytic Jacobian was checked against central finite differences at
  real points: maximum absolute discrepancy **1.94e-7**. No optimizer hit its
  iteration or parameter bound. C/D runs converged by tolerance; residual and
  optimizer diagnostics are retained in the numerical report.
- No post-change viewer screenshots exist because no candidate passed. The
  prior baseline screenshots at 39862079571 and 4144403454 remain under
  `/tmp/oak-quest-diag/baseline-{rgb,left}-{39862,4144}.png`.

Scratch scripts and streams: `/tmp/oak-intrinsics-diag/{infer.py,fit.py,evaluate.py,check_math.py}`,
`rgb-scores.parquet`, `mono-predictions.parquet`, `fits.json`, `evaluation.json`.
No npy/npz files, dataset writes, or process-name kills were used. No production
implementation or full gate was run because the requested acceptance check
failed.
