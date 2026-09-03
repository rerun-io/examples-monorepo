# Worked example — X-Lens + LAMP (2026-09-02/03), two ports in parallel through Codex

Both runs used this skill end to end with the user asleep: four Codex jobs (two forks, two ports) in tmux with sentinels,
Claude reviewing, taking the pixel evidence, and sending corrections back through `codex exec resume`.

## X-Lens (calibrated multi-view metric depth, pinhole + fisheye rigs)

Fork `pablovela5620/XLens` (`main` = `e6bdf2f6`, the last Apache-2.0 commit — upstream HEAD relicensed to CC-BY-NC).
Weights gated CC-BY-NC on `henryzhou998/X-Lens` (rev `1d0c9635`), never mirrored. New monoprior family `rig_depth`
(`unit_rays` from simplecv cameras, `RigDepthPrediction(depth_m, confidence, mask, scale)`), PR stack #195 → #198 → #199 → #201.

- Numbers: ETH3D `playground_1l` two-view pinhole **baseline** — metric abs-rel 0.76 / EPE 4.47 px, median-scale-aligned
  abs-rel 0.15 / EPE 1.42 px (a ×1.76 global scale miss: the model normalises translations, so a 6 cm baseline carries no
  metric cue). 5090 bf16: 6 views 504×798 ≈ 0.9 s, 490×938 pair 168 ms; catalog tool 4 views 896×504 ≈ 1.0 s predict + 1.7 s log (PNG depth + TSDF) per frameset, 60 s @5 fps = 652 MB rrd (the six-camera Points3D version was 3.4 GB). 1120×630 ×4 needs a 4.7 GB attention-bias allocation alone.
- Gotchas: bare safetensors needs `xlens_vits.yaml` (else ViT-L defaults); upstream `strict=False` hides partial loads → assert
  empty key lists; beartype claw forced value-preserving annotation fixes in vendored files (PR 2); one-member tyro union;
  per-frame `Points3D` made a 60-s catalog rrd 3.4 GB → rectified pinhole twins + `EncodedDepthImage`; the two robocap eye
  cameras are out of distribution → four outward cameras only; upstream URL typo in the vendored docstring caught in review.

## LAMP (4-camera egocentric 3D people tracking, SMPL output)

Fork `pablovela5620/LAMP` (`main` = `db3e4bf9`, CC-BY-NC code + weights). New package `packages/lamp` (`lamptrack`) on
`common` + `cuda` + `posekit`; 2D stage = posekit RT-DETRv2 + ViTPose-plus-base (upstream's checkpoint); RF-DETR only in the
fork (bumped 1.5.0 → 1.9.x for transformers 5). PR stack #196 → #197 → #200 → #202.

- Numbers: fork on the Aria `test-library` (700 framesets @10 Hz): RF-DETR 8.7 / ViTPose 6.6 / lifter 8.6 / total 28.5 ms,
  lifted-joint reprojection 6.3–7.9 px; port on robocap s29 (four KB4 cameras fed as virtual-pinhole vectors after
  `cv2.fisheye.undistortPoints`, window 1152–1272 s chosen by a 1 Hz detection scan): 163–170 ms per frameset (detector 38 / pose 28 / tracker 53 / lifter 20), 20 tracks in 30 s, 32 in 120 s; 30 s = 334 MB, 120 s = 1.06 GB with per-frame normals + half-res previews (2.2 GB before).
- Equivalence: vendored-vs-pristine lifter bit-identical (16 cases); real-fixture seam (lifter I/O + isolated smoothing window)
  bit-identical / 1e-4 m; full-history smoothing off by ≤0.088 m because the fixture lacks merge events (documented limit).
- Rendering (user request, mirrors `sam3d-body`): posekit `log_person_bbox`/`log_person_points2d` + `person_color` keyed by track id, joints `Points3D(keypoint_ids, class_ids)` under an SMPL-24 `AnnotationContext`, `Mesh3D` with `compute_vertex_normals` + alpha 0.5, `LivePeopleLogger` clears ended tracks, 960×540 previews under `pinhole/preview/` (excluded from the 3D view: a half-res image on the full-res Pinhole plane covers a quarter).
- Gotchas: SMPL ≠ SMPL-X (Hub mirror + chumpy `find_class` shim, private hosting); the port invented a fixture contract before
  the fork produced the real one → the fork owns the layout; ended tracks persist in Rerun without `rr.Clear`; people render
  like sam3d-body (translucent meshes, annotated joints); `requests` needed explicitly in the isolated catalog lane; measure
  performance outside the beartype dev env.

## What the runs changed in this skill

Rerun geometry rules (fisheye twins, Clear on track end, no per-frame clouds, sam3d-body people, OOD sensors), non-stereo
numbers incl. scale-aligned depth metrics and replay fixtures owned by the fork, the follow-up-by-resume pattern, the
new-package lane, one-member tyro unions, beartype-forced vendored annotation fixes, verbatim ports of frame-convention helpers.
