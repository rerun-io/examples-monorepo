# Position-only Quest hands and dynamic session rigs

The CLI is unchanged. Missing quaternion columns no longer fail a segment.
Orientation availability is recorded per hand as `quest` or `none` in the
checkpoint, `hand-orientation-source.json`, fit report and parameter recording.
A mixed capture keeps measured rotations for the hand that has them.

For a position-only hand, the fitter derives a bone direction and palm-normal
frame from the 26 raw OpenXR joint positions. This supplies a roll reference,
not a claimed measured orientation. Its rotation weight is one tenth of the
measured Quest rotation term. The normal Quest 21-point position loss remains.
Additional priors apply only to position-only hands: strong axial finger-roll
regularization, finger flexion/hyperextension and sideways bounds, and wrist
flexion/sideways/twist bounds. Thumb limits are broader. The local flexion sign
comes from the position-based initialization; these are broad plausibility
priors, not a learned hand-pose model.

The raw orientation reader preserves missing samples and no longer demands
first-25-second orientation calibration when quaternions are absent. Measured
orientation captures retain their existing alignment and weights.

## Validation: 027d8215 versus d31

`selfcap__2025-11-24__027d8215__ep001` completed the full RTMW batch pipeline
locally, with both sources `none`: 3,625 frames, 69.4 seconds under shared load,
381.6 MiB peak CUDA allocation. Quest position residual median/p90 is
4.91/10.52 mm. Body is 19.17/37.52 mm; ears 4.26/9.04 mm.

Whole-segment plausibility proxies:

| Metric | 027d8215, position-only | d31, measured rotations |
|---|---:|---:|
| Finger roll p90 degrees | 25.56 | 77.96 |
| Broad finger-limit violations % | 1.14 | 18.42 |
| Wrist twist p90 degrees | 41.49 | 25.12 |
| Flipped hand faces p90 % | 1.04 | 0.23 |
| Collapsed hand faces p90 % | 0.40 | 0.37 |

These compare different captures, not paired ground-truth accuracy. The
flipped-face proxy is worse for the position-only capture. Close-up inspection
at 39.858 s and the two worst proxy frames (117.518/121.160 s) found no open or
collapsed wrist and no hand folded back over itself. Low-LOD creases and some
knuckle folds remain. Evidence:
`/tmp/soma-fit/xyz-hands/selfcap-post-ep1-v2/selfcap__2025-11-24__027d8215__ep001/hand-closeups.png`.

## Dynamic rigs

The catalog `kind` properties now identify exo, OAK/ego and Quest roles.
Head-pose reads, exo camera enumeration, OAK factory poses/lens projections,
EgoHand lookup and OAK overlay entity paths use those roles. No fixed
`rig_04`/`rig_05` assumption remains in these consumer paths.

Live reads confirmed b910c05f has exo rigs 00–02, OAK 03, Quest 04; the first
5 s yielded 145 finite head samples. The four-phone d31 has exo 00–03, OAK 04,
Quest 05; the same check yielded 116 finite head samples. Per-camera reads
returned three and four exo cameras respectively and the correct OAK index.

The checkpoint algorithm version changes to v6. Use the pinned commit for
retries; do not import the actively edited worktree into running jobs. Local
validation did not publish either test capture.

The full b910c05f batch path also completed locally in 55.95 s. All three
explicit OAK overlays were written beneath `rig_03`, while head poses came
from `rig_04`; provenance correctly reported both hands as `none`. Output:
`/tmp/soma-fit/rig-fallback/selfcap-post-ep1-v2/selfcap__2025-11-27__b910c05f__ep001/`.
Sixteen targeted unit tests passed, with one existing hardware test deselected;
Ruff, targeted type checking and entry-point imports passed. The 027d8215
real run and close-up render provide the GPU/integration evidence.
