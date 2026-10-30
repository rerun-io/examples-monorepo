# d31 normalized optical refinement

The former gate scored fold models but deployed a separately fitted all-frame
model. Each target kept 20 virtual anchor observations while hand/body counts
grew, so the all-frame model changed the objective balance. Its headset error
on its own labels was 24.40 px although the reported fold median was 4.11 px.

The full procedure now uses mean robust costs with relative shares
headset:phones:hands:body = 1:0.25:1.5:0.5, equal active-view shares within each
block, and a 6 px Cauchy scale. The first solve uses the same three relevant
shares, followed by anchored body BA. The full procedure is independently run
in each fold and once on all observations. Both fold and deployed per-view
label scores are recorded. A deployed label median more than 2 px above its
fold score rejects the fit. Intrinsics and the Quest/OAK geometry remain fixed.

One manual phone0 observation was added in rig01 at 14.751758516 s:
(1813,292) source pixels, confidence 0.65. The back lens is visible frontally
in the existing packet. No held-out headset labels entered the fit. Static
phone coordinates repeated in the check are consistency checks, not independent
3D information.

The shares were tuned on d31, including its manual diagnostic protocol; the
manual numbers below are not a blind benchmark for other sessions. The protocol
uses 10 unlabelled frames per view, 59 visible headset apertures in total.
Numbers are median/p90 source pixels; "before" is the previously deployed model.

| View | Headset before | Headset after | Phones before | Phones after |
|---|---:|---:|---:|---:|
| 00 | 4.50/8.41 | 2.38/6.68 | 8.04/8.04 | 7.86/7.86 |
| 01 | 25.60/26.26 | 2.89/8.00 | 65.40/113.95 | 4.21/8.37 |
| 02 | 32.88/40.18 | 8.07/14.40 | 5.66/5.66 | 5.91/5.91 |
| 03 | 23.51/25.10 | 2.61/3.47 | 3.06/3.31 | 2.66/2.67 |

Full-procedure CV: headset 2.88 px, phones 0.47 px, Sapiens hands
8.390 -> 9.387 (+0.997), RTMW hands 9.595 -> 10.104 (+0.509), body transfer
25.277 -> 14.168 px. Both folds and the final optimization converged.
Deployed in-sample headset median is 2.33 px, with per-view medians
1.68/2.35/7.00/2.32 px. Deployed phone median is 0.43 px.
The independently rechecked ladder-phone error is 113.95 -> 0.06 px.

Local evidence: `/tmp/exo-normalized-proof/exo-centers-39862079571.png`.
Manual annotations and measurements: `/tmp/exo-gap-d31/manual-labels.json`
and `/tmp/exo-gap-d31/manual-scores.json`. Training annotations and gate results
are the adjacent typed JSON records. No catalog registrations or corpus runs
were made, and this change was not sent to the batch runner.
