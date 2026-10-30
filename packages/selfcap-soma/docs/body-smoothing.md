# Body acceleration default for the RTMW batch path

The RTMW batch default now includes a body-only acceleration prior. It covers
Hips, Spine1/2, Chest, Neck1/2, Left/RightShoulder and Left/RightArm. Elbows,
wrists and fingers are excluded from this added prior. Quest hand position and
orientation terms are unchanged. The existing mild all-pose prior is retained.

Acceleration uses actual timestamps, normalized to a 4 mm second difference
at 24 Hz, with weight 1.0. Intervals over 100 ms do not couple. The root/chest
rotation-velocity-change weight rises from 0.1 to 1.0. During the second half
of the pose pass, Adam's learning rate decays geometrically from 0.025 to 0.001.
The final decay prevents high-rate optimizer oscillation from defeating the
stronger prior; no temporal filtering or delayed hand targets are introduced.
The objective remains a simultaneous, offline fit over the whole segment.

## Validation on d31bf1da

Dataset `selfcap-post-ep1-v2`, RTMW, 4,367 frames, fresh per-segment shape,
600 pose iterations. Before is the runner's original batch output under
`/tmp/soma-batch/selfcap-post-ep1-v2/selfcap__2025-12-03__d31bf1da__ep001`.
After is the exact new batch CLI with defaults and `--no-publish`, under
`/tmp/soma-fit/body-smooth/batch/selfcap-post-ep1-v2/selfcap__2025-12-03__d31bf1da__ep001`.
The existing catalog layer was not replaced during this validation.

Second-difference magnitude at 24 Hz, median / p90 mm:

| Group | Before | After |
|---|---:|---:|
| Pelvis/root | 14.08 / 32.24 | 0.35 / 0.66 |
| Spine and neck | 10.70 / 28.76 | 0.30 / 0.60 |
| Shoulder chain | 10.02 / 35.82 | 0.39 / 1.12 |
| Elbows and wrists | 8.40 / 21.70 | 1.54 / 9.24 |
| Fingers | 9.67 / 24.07 | 1.13 / 10.20 |

Both pelvis and spine are below the requested ~4 mm median. Reduced finger
jitter is an indirect result of removing body wobble and converging the pose
optimizer; the new acceleration loss never receives finger positions.

Observation residuals, median / p90 mm:

| Source/group | Before | After |
|---|---:|---:|
| RTMW body (COCO 5–12) | 23.27 / 59.03 | 23.08 / 46.91 |
| RTMW shoulders | 18.92 / 35.54 | 20.83 / 43.84 |
| RTMW ears | 3.82 / 7.06 | 3.66 / 6.81 |
| Quest hands | 6.14 / 11.38 | 5.09 / 10.31 |

The shoulder trade-off is +1.91 mm median and +8.30 mm p90; other reported
residuals improve. The source residuals are not independent accuracy measures.
A +/-3 frame temporal-offset check against Quest hands, on the same valid
samples at every offset, has its minimum at zero both before and after.
After-fit median errors at -1/0/+1 frames are 7.40/5.09/7.50 mm. Thus this
check detects no introduced whole-frame hand lag; it does not bound sub-frame
latency. No latency offset was changed.

A prior-only trial already reduced pelvis median to 2.80 mm, but spine stayed
at 4.24 mm. Adding the final optimizer decay removed that remaining numerical
oscillation. A warm-start refinement and the final fresh-shape batch run both
passed the target; only the latter supplies the table above.

## Runner use and checks

The existing command is unchanged; new defaults take effect on importing the
new commit. Restart old worker processes before using it. The checkpoint
algorithm version is `soma-batch-v2-body-acceleration`, so v1 artifacts are
invalidated automatically. Shape is still fitted once per segment, with legs
included only in that stage. No source calibration or catalog blueprint changes.

A `jitter.json` group report is written beside each new fit. Full local evidence:
`temporal-comparison.json` in the after directory, plus `report.json`,
`checkpoint.json`, and the layer RRD. The new run took 43.5 s before publication
under the shared host load (shape 14.3, fit/mesh 12.5, write 3.4, overlays 12.2 s),
with 428.4 MiB peak CUDA allocation. This is not a new concurrency benchmark.

Nine targeted unit tests, one GPU frozen-leg/gradient regression, Ruff and
targeted Pyrefly passed. The head/torso/waist render is checked separately; no
full gate and no dataset write were performed for this test.
