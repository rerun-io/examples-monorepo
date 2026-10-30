# Prepared native exo/Quest projections

Status: opt-in preparation only. No catalog layer is published by this change.
Await the runner's native Mesh3D pixel test before enabling it for d31.

Add `--native-exo` to the batch or overlay CLI. It omits every baked exo SOMA
Points2D/LineStrips2D entity. Quest already had no baked SOMA overlays, so only
OAK's three cameras retain explicit projections through the full 14-coefficient
lens model. The 3D mesh, joints and ears remain available for native projection
through calibrated exo and Quest Pinhole views. Default remains disabled.

The blueprint must include `/world/pred/soma_x/**` and `/world/skeleton` in the
exo/Quest views; this code does not change blueprints or source keypoints.
Camera-specific cyan/grey styling belonged to the baked exo overlays and is
not retained by a single native 3D representation.

Local export validation:
`/tmp/soma-fit/native-exo-prepared/selfcap__2025-12-03__d31bf1da__ep001.rrd`
contains all 4,367 mesh frames and exactly three baked 2D joint entities, under
`/world/rig_04/cam_00..02/pinhole/pred/soma_x/joints_uv`. There are no baked
exo or Quest 2D entities. Native Mesh3D rendering remains pending the runner's
pixel test; structural validation alone does not establish viewer support.
