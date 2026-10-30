# Dense visible-hand OAK alignment

Scope: `selfcap__2025-12-03__d31bf1da__ep001` only. This follows the instruction
to drop toes and fit only visible hands. No toe or static-scene point enters
these results. Static-corner checks were skipped; they were optional and did
not offer a quick independent measurement.

## Result and choice

Dense EgoHand observations improve the mount translation while preserving the
production gyro rotation. Free rotation improves the RGB median another
0.64 px, but has a worse RGB p90 and worse independent mono median than the
fixed fit. Use the gyro-fixed result rather than infer a physical rotation
correction from the narrow hand depth range.

Held-out **mapped finger joints only**, median / p90 pixels:

| Fit | RGB | Independent mono-left | Rotation change, folds 0 / 1 |
|---|---:|---:|---:|
| Today | 11.00 / 21.42 | 19.44 / 30.02 | 0 / 0 deg |
| Dense hands, gyro fixed (selected) | 9.28 / 18.79 | 14.62 / 24.39 | 0 / 0 deg |
| Dense hands, rotation free | 8.64 / 19.05 | 15.38 / 23.65 | 3.38 / 4.46 deg |

The RGB score uses 87,526 held-out correspondences and mono uses 32,353.
The two evaluation sets each include every eligible observation exactly once:
train on even 10-second blocks and test odd, then reverse. Exclude 200 ms at
block edges. Wrists and knuckle/base slots are excluded from fitting and scoring.
The 15 mapped indices are `2,3,4,6,7,8,10,11,12,14,15,16,18,19,20`.

Use the original Quest tracking checks, finite image coordinates, per-hand
RT-DETR scores, a 300 px/s EgoHand image-speed gate, Cauchy loss (6 px), and one
30 px correspondence-trimming pass. All eligible training pairs are used;
there is no point subsampling. The independent mono predictions come from
actual grayscale images, not RGB 3D projected into mono.

## Conditioning and final fit

The eligible fingers span:

- Camera depth z: **0.404–0.694 m**; central 90% **0.454–0.551 m**.
- Camera distance: **0.430–0.833 m**; central 90% **0.477–0.597 m**.
- Image x: **6–910 px**; central 90% **295–856 px**.
- Image y: **−25–603 px**; central 90% **263–472 px**.

The small negative y values are finite extrapolated fingertip estimates at the
image boundary; they are not invented 3D observations. There is wide lateral
coverage but little depth diversity. The free-rotation fits shift the mount by
34.93 / 45.60 mm and rotate it 3.38 / 4.46 degrees across folds. The fixed fits
shift it 10.40 / 9.12 mm. Neither comparison establishes that the pooled gyro
rotation is physically wrong.

Fit the selected fixed model on the full segment: 91,010 eligible pairs before
trimming, 89,430 retained. The final translation changes by **9.88 mm**:
`[49.478, 35.567, -81.588]` to `[47.288, 42.205, -74.611]` mm in Quest-rig axes.
Rotation, RGB intrinsics, all 14 lens coefficients, clock offset, and the 48 ms
hand latency stay unchanged. The full-segment retained-pair median is **9.13 px**.

## Rotation provenance

`oak_device_rgb_R_imu.json` was introduced here by commit `68a0f6c30`; its table
is dated 2026-10-26. `pool_rotations` and the `pool_oak_rotation` CLI establish
how it is made: `R_rgb_from_head` from a free hand PnP fit is composed with
`R_head_from_imu` from gyro alignment. Eligible sessions require gyro spread at
most 1 degree, hand-fit median at most 20 px, and at least 50% retained hands.
Rotations are averaged after trimming deviations above max(3 degrees,
2.5 × median deviation). This is a **hand-derived pooled calibration**, not an
independent factory IMU extrinsic.

For device `OAK - 14442C10C1F3CFD200`, the table lists **351 source sessions and
327 retained sessions**. This session is in both lists. Its regenerated gyro
fit is valid, with episode spread **0.00864 degrees**, below the production
1.5-degree limit. `fixed_rotation` reproduces the production mount rotation to
**6.85e-7 degrees**, confirming the fixed branch was used.

The base's preserved `raw_calibration_json` has only `device_id`, `intrinsics`,
and `extrinsics`; the only extrinsics are `left_to_right` and `left_to_rgb`.
It contains **no `imuExtrinsics`**, so a numerical comparison with the device's
factory IMU calibration cannot be made from this recording. No factory value
was guessed.

## Implementation and validation

`selfcap-calibrate` and `selfcap-post` accept an optional `--oak-observations`
directory containing `<segment>.parquet`. The columns are `segment_id`, int64
`video_time` nanoseconds, COCO-133 `xy`, and `scores`. Missing joints remain NaN
with zero confidence. The reader checks identity, timestamps, dimensions and
confidence range. Supply the same box score for all joints of a hand.

With this input, `solve_oak(weighted=True)` uses dense mapped-finger observations,
image-speed gating, confidence weights, all 14 distortion coefficients, and one
trimming pass. The default keeps the gyro rotation; `--oak-free-rotation` is an
explicit alternative. A requested fixed fit fails clearly if the gyro rotation
is unavailable. Existing calls without `--oak-observations` retain the original
Sapiens selection and solver path. Calibration metadata names `egohand` as the
observation source.

The regression tests first failed before the new interfaces existed, then
passed: **43 targeted tests passed**, four integration tests were deselected.
Tests cover known-mount recovery with the full lens for fixed/free rotation,
ignoring base joints, trimming a persistently bad finger, and observation
identity/round trips. Changed-file Ruff checks passed. Targeted Pyrefly reported
zero errors, with warnings about other uninstalled workspace environments.
No full gate was run.

The dev driver mirrors `/tmp/selfcap-5090/dev_cycle.py`, with its worktree check
updated and the new observation directory passed to `post.Config`. It completed
in **40.9 s processing**, plus 1.1 s fetch and 3.0 s publication. Only this
segment's layers in `selfcap-post-dev` were replaced. No other shared dataset was
written. Saved calibration confirms the selected rotation and translation.

## Pixel evidence

Updated maximized native Rerun screenshots:

- `/tmp/oak-hands-final/rgb-39862.png`
- `/tmp/oak-hands-final/left-39862.png`
- `/tmp/oak-hands-final/rgb-4144.png`
- `/tmp/oak-hands-final/left-4144.png`

All four PNGs were inspected. At 39.862 s, the blue finger chains sit closer to
the visible hands in both RGB and mono-left. At 4.144 s, the partly occluded
left hand remains imperfect; the visible right fingers align better. Wrist/base
mismatches remain because those definitions were explicitly excluded.

On the RGB source frame displayed at 39.862 s, the mapped-joint median versus
EgoHand changes from 13.48 to 8.23 px. At the 4.144 s source frame it changes
from 22.73 to 19.71 px, while the p90 increases (49.46 to 53.51 px). The latter
is not an across-the-board per-frame improvement. The held-out and independent
mono tests, rather than either single frame, justify the selected option.

The dev catalog reports `calib_oak_source = egohand` and
`calib_oak_median_px = 9.12845516`, so these screenshots use the new calibration.

Scratch reproduction: `/tmp/oak-hands-final/compare.py`, `export.py`,
`provenance.py`, `dev_cycle.py`, `cycle.log`, and `observations/`.
The adjacent metrics JSON preserves the held-out scores, fitted matrices,
conditioning ranges, and provenance checks. No npy/npz files were written.
