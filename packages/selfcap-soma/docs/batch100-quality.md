# Batch-100 residual diagnosis

Read-only review of saved reports and published input/output geometry in
`selfcap-post-ep1-v2`. No fit or catalog layer was changed.
Review source: `/tmp/selfcap-night2/batch100-review.json`; detailed read-only
queries: `/tmp/soma-fit/diagnose-<segment>.log`.

| Session suffix | Body / shoulders / ears / Quest hands median mm | Input ear span p10 / median / p90 mm | Fitted ear span mm | Screening |
| --- | --- | --- | --- | --- |
| d31bf1da | 18.65 / 19.83 / 1.93 / 5.54 | 146 / 160 / 165 | 161 | pass |
| 819a09a4 | 43.70 / 54.25 / 46.29 / 5.48 | 59 / 237 / 387 | 82 | flag |
| e9dd8490 | 61.62 / 48.14 / 33.91 / 6.60 | 25 / 150 / 209 | 127 | flag |
| 0b36cf91 | 56.45 / 81.11 / 47.75 / 9.04 | 89 / 314 / 470 | 442 | flag |

d31 uses accepted refined calibration. All three bad sessions had rejected
calibration candidates and retained unrefined exo calibration. The input ear
span varies far more than skull geometry permits. In 819a09a4 and 0b36cf91 no
ear observation has three-camera support; their two-view-supported ear residual
medians remain 35 and 37 mm. For e9dd8490 even three-view-supported ears have a
33 mm median residual. Removing unsupported frames alone will not fix these.
The 0b36cf91 fitted global scale is 1.586, versus 1.10 in d31 and e9dd8490.

This confirms inconsistent triangulated input and a shape stage that absorbs
it, particularly in 819a09a4 and 0b36cf91. It does not isolate calibration error
from detector errors. A constant RTMW ear-definition offset cannot explain the
time-varying span. Soft Quest rotation may contribute to residual conflict,
but disabling it without a controlled fit would be speculative. No solver
change is included: stricter support gating would remove all ear supervision
in two bad sessions and cannot be claimed safe for good sessions.

New fit/export outputs include `/__properties/soma_quality`, a static property
with queryable status, medians and thresholds, plus a JSON record. Fit outputs
save `quality.json`; layer exports save `<layer-stem>.quality.json`. A flag means
any median exceeds body 40 mm, shoulders 50 mm, ears 30 mm or Quest hands 15 mm,
or a metric is missing. The body/ears/hands thresholds follow the batch review;
50 mm is an explicit additional shoulder screening threshold. Equality passes.
These are finite-point training residuals, not independent accuracy or coverage
tests. A pass does not prove that the calibration or mesh is correct.

Existing layers are unchanged; published checkpoints are not automatically
rerun or backfilled. The new property is written on the next fit/export.
Four targeted tests cover thresholds, missing data, upper-body selection and
local RRD property round-trip. No mesh or pose computation is changed.
