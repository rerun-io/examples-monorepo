# Automatic exo-label validation

Fresh runs on 2026-10-30 used only automatic Sol labels, with no manual
anchors or copied annotation caches. Fixed VP intrinsics, normalized objective
shares, full solve-plus-polish cross-validation, and the existing gate were used.
Quest hands, head poses, and OAK/Quest calibration remain unchanged.

Every other exo phone whose predicted centre is inside the image receives a
540-source-pixel crop. Headset and phone calls are separate. A missing visible
phone gets one automatic retry, with already identified phones as context.
Frontality and a 240-pixel search radius filter identity substitutions.

## Independent d31 manual check

These manual marks are evaluation only: ten unlabelled frames per view, using
the same protocol as the preceding diagnosis. Phone bearings are static.
Before means the previous deployed model from that diagnosis, not the S1
baseline used by the gate. All values below are median pixel errors.

| View | Headset before → after | Phones before → after |
|---|---:|---:|
| rig_00 | 4.50 → 2.73 | 8.04 → 2.14 |
| rig_01 | 25.60 → 5.23 | 65.40 → 19.09 |
| rig_02 | 32.88 → 6.47 | 5.66 → 5.94 |
| rig_03 | 23.51 → 4.94 | 3.06 → 9.10 |

Sol found the ladder phone in rig_01 at (1817, 290), confidence 0.71.
Its final independent projection error is 3.51 px. Not every phone improves:
the rig_01 aggregate includes the other phone, and rig_03 worsens. The automatic
gate's phone score is consistency against shared static training anchors,
not an independent phone validation score.

## Complete automatic runs

The hand and body scores below cross-validate the whole solve-plus-polish
procedure, from the unrefined S1 baseline. Pixel medians:

| Session | Sapiens hands | RTMW hands | Body transfer | Decision | Wall time |
|---|---:|---:|---:|---|---:|
| d31bf1da | 8.39 → 9.05 | 9.59 → 10.33 | 25.28 → 15.27 | Accept, 30 labels | 42.2 s |
| 5adb9b92 | 11.07 → 12.58 | 17.22 → 14.85 | 56.91 → 14.92 | Reject: Sapiens +1.52 > 1.0 | 34.7 s |

d31 headset/phone gate medians: 39.14 → 3.19 / 104.63 → 0.99.
5adb headset/phone gate medians: 16.58 → 3.19 / 146.32 → 0.69.
No thresholds changed. Rejected runs retain candidate.json for diagnosis but
do not emit refinement.json or calib.json.

One raw-calibration query of selfcap-post-ep1-v2 returned 43 sessions; none
contained Comet, Hammer, or Titanium-57BB phones. The third test was skipped
as instructed. No protected dataset was read and no catalog writes were made.

Artifacts: /tmp/exo-auto-d31-v3, /tmp/exo-auto-5adb-v3,
/tmp/exo-v2-devices.parquet. The four-view local rendering was inspected:
/tmp/exo-auto-proof/exo-centers-39862079571.png (and matching .rrd).
The headset centres are near the apertures; remaining phone errors are reported
above rather than hidden by the successful gate.
