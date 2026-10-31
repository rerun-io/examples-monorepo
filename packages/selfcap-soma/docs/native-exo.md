# Prepared native exo/Quest projections

Status: the runner's native Mesh3D pixel test passed in Rerun 0.38.1.
Enabled for the d31 review layer only; the CLI remains opt-in.

Add `--native-exo` to the batch or overlay CLI. It omits every baked exo SOMA
Points2D/LineStrips2D entity. Quest already had no baked SOMA overlays, so only
OAK's three cameras retain explicit projections through the full 14-coefficient
lens model. The 3D mesh, joints and ears remain available for native projection
through calibrated exo and Quest Pinhole views. Default remains disabled. Native SOMA joint and ear radii use 2.5 UI points,
so toggled markers do not shrink with camera distance.

The blueprint must include `/world/pred/soma_x/**` and `/world/skeleton` in the
exo/Quest views; this code does not change blueprints or source keypoints.
Camera-specific cyan/grey styling belonged to the baked exo overlays and is
not retained by a single native 3D representation.

Local export validation:
`/tmp/soma-fit/native-exo-prepared/selfcap__2025-12-03__d31bf1da__ep001.rrd`
contains all 4,367 mesh frames and exactly three baked 2D joint entities, under
`/world/rig_04/cam_00..02/pinhole/pred/soma_x/joints_uv`. There are no baked
exo or Quest 2D entities. The pixel proof is `/tmp/selfcap-night2/evidence/native-mesh-test.png`
(red Mesh3D triangle and blue Points3D point over the image).

The UI-radius review artifact is under `/tmp/soma-fit/native-exo-ui/`.
Cleanup is off and local d31 evidence is retained.
