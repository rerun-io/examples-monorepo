# Shape selection after S2 confidence changes

The prior all-six-anchor confidence >=0.7 gate was incompatible with S2's
mean-inlier confidence and support/bridge downweighting. On d31 it found only
4 early frames before striding; all 596 early frames had finite anchors. Hip
median confidence was 0.64/0.68. The gate used ears/shoulders/hips, not knees or
feet.

The consumer now reads `support_view_count` in the same windowed query as 3D
positions/confidence. It prefers finite, positively weighted anchors with at
least two supporting views. Confidence ranks evidence and continues to weight
the loss; it is no longer compared against a detector-level 0.7 threshold.

If the first 25 s do not yield 64 sparse calibration frames, take the best 96
supported frames over the whole segment, in chronological order. If direct
support is sparse, use the best finite positive-weight evidence, ranking
bridged points below directly supported points. Fail only when fewer than
three usable whole-segment anchor frames remain. Missing support metadata
falls back to finite positive-weight evidence for older layers.

For current d31 S2, 31 early support-qualified sparse frames triggered the
full-segment fallback, selecting 96 frames from 3.83 to 179.33 s. Tests cover
lowered two-view scores, absent early evidence, and sparse direct support.
The batch cache version changes to v3, and support counts enter its signature.

A fresh shape check gave global scale 1.003, rather than the suggested
1.10–1.12. This scalar does not alone describe size when native identity and
bone ratios are also fitted. Physical dimensions match the new source better:

| Dimension | Old shape mm | New shape mm | Source median mm |
|---|---:|---:|---:|
| Shoulder width | 445.1 | 376.7 | 383.3 |
| Hip width | 216.3 | 214.8 | 215.2 |
| Left upper arm | 255.6 | 240.5 | 249.4 |
| Left forearm | 248.8 | 216.2 | 215.6 |

The checked old local calibration had scale 1.059, not 1.10–1.12. No scale was
forced to satisfy a nominal value. Evidence: `/tmp/soma-fit/check-shape-full.log`
and `/tmp/soma-fit/s2-shape-dimensions.log`.

## Partial-body observations and runner status

The `5cb1e14c` failure was an overly strict all-six-anchor requirement. Among
1,682 frames, two-view support counts were ears 1,177/1,285, shoulders
1,678/1,561, and hips 0/1,339. All three exo cameras had raw left-hip detections
above 0.3 in 997/1,236/1,545 frames, but the geometric triangulation rejected
that hip throughout. This is not a missing-camera or whole-body absence case.
The exact physical cause of the rejected hip is not inferred from detector
confidence alone.

Shape selection now permits two shoulders, at least one hip and at least one
ear. Initialization uses the actually observed torso anchors (minimum three),
without filling or adding a target for the rejected hip. Missing bone-length
targets remain excluded from the loss as before.

A genuinely insufficient selection raises `InsufficientBodyObservation`. The
batch CLI prints `SOMA_INPUT_LIMITATION insufficient_body_observation` with the
segment and exits with status **20**, distinct from ordinary failures (1).
Runners should record exit 20 as an input limitation. No layer is published
for that case. The checkpoint version is v7 to invalidate the old strict gate.

The complete `5cb1e14c` RTMW batch validation succeeded locally in 42.37 s,
without publication. Nine targeted tests and static checks passed, including
missing-hip selection and exit-status classification.
