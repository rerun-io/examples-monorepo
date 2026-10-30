# Prepared waist and observation polish

Status: local preparation only. Do not publish until the Sol-refined exo
calibration and regenerated `keypoints_sapiens2` layer are confirmed registered.
Keep the optimizer settings and shared shape from `4c9e4175`.

## Waist

The display selection now uses a horizontal plane in the fitted **rest pose**,
25 mm below the mean hip-joint height. Faces touching any vertex below the
plane are dropped; retained vertex IDs are fixed for all frames. The retained
boundary is moved to the plane in rest coordinates and then skinned. This last
display-only step avoids saw-tooth flaps from the coarse triangles. It does not
change the fit, loss targets, or internal joint positions. Normals are recomputed
by the exporter. The underside remains open, without a cap.

With the current shared shape, the plane is at rest Y=-0.11837 m (this is not a
world floor height). The crop retains 3,483 vertices and 6,942 faces. Front,
side and back were inspected in `/tmp/soma-fit/waist-polish/cut-preview.png`.
The hem is continuous and flat in rest coordinates, with no crotch point or
rear flaps. It follows pelvis motion when posed.

## Observation styling

The published fit actually uses anatomical joint centres for shoulders, elbows,
wrists and hips, and surface landmarks for ears. It does not use shoulder
surface landmarks. Per the follow-up instruction, the fitting loss is unchanged.

New fit recordings explicitly persist `/world/fit/coco_landmarks`; the exporter
uses this for camera projections and observation checks. For older recordings,
the equivalent joint mapping plus the saved surface ears is used.

For each joint, compute its median 3D residual against the source triangulation
over the segment. In each exo camera, use
`max(15 px, 2 * median_residual_m * max(fx, fy) / depth_m)` as the proximity
radius. A finite raw Sapiens2 detection with confidence >=0.7 is still required.
Missing residuals fall back to 15 px. This is a display tolerance, not a new
confidence score or an accuracy claim. OAK hand evidence rules are unchanged.

Local files under `/tmp/soma-fit/waist-polish/` are previews, not replacements
for the currently published layer. `overlay-preview.rrd` has the new observation
styling over the existing fit; its mesh still has the old selection. The waist
preview separately renders the new cut. No full-segment refit was run.

Pixel check at 39.862 s: hands and forearms show cyan support, but the old fit's
shoulders are still grey against the currently served calibration. The last
query measured shoulder distances 26.7 / 37.1 px against tolerances 18.0 / 15.5
px (raw scores 0.953 / 0.871). Camera projection changed while preparation was
running, so these are not results from a matched calibration/refit. Recheck
only after the announced new layer is ready and SOMA has been refit. Do not
force the torso cyan when it fails the stated rule.

Validation: 11 targeted unit tests, all three existing short GPU integration
tests, the published procedural-twist variant of the frozen-leg test, Ruff,
and targeted Pyrefly passed. The legacy foot test now explicitly opts into its
foot loss because the default foot weight was already zero; this does not
change the published optimizer settings.
