# Gates

A phase is not done until its gate passes. Report failures as failures.

## Gate 1 — Fork runs (end of phase 1)

- Fresh `git clone` → `pixi run demo` succeeds; `pixi lock --check` clean.
- Rerun demo screenshot: rig frusta with images, depth/cloud in the 3D view, no red error badges.
- No display: the producer saves `.rrd`s (`mkdir -p out && python -u demo_rerun.py --rr-config.headless --rr-config.save out/<scene>.rrd`
  — `RerunTyroConfig` opens the sink in `__post_init__`, so the directory must exist), then takes the screenshot from a headless
  viewer (`ViewerClient.spawn(headless=True)` or `rerun --headless` + `rerun viewer-mcp`), or the reviewer does; reviewer
  evidence lives in `/tmp/rerun-viewer-validation/<date>-<model>/`.
- In-script metric equals upstream eval on the same sample (or is recorded as baseline when upstream has none).

## Gate 2 — Reference number (PR 2)

- Slow GPU test loads the released checkpoint and reproduces the fork's number on the ETH3D sample.
  Stereo: EPE + bad1 on non-occluded pixels with `gt < max_disp`; compare to paper (dataset mean — a single
  scene lands near, not on, it) and record both in the test docstring.
- Non-stereo families use the numbers defined in Phase 0 item 5 (metric + scale-aligned depth metrics; tracker counts, per-stage
  ms, reprojection error, fixture replay within a stated tolerance — upstream BF16/CUDA-graph outputs are not bit-identical to
  an fp32 eager replay; bit-identity is for vendored-vs-pristine on synthetic inputs).
- Warm timing: `torch.cuda.synchronize()` around 50 forwards after 10 warm-ups, at the demo resolution;
  record ms/frame in the PR description.

## Gate 3 — Equivalence + dev tasks

- Every PR: the dev tasks below. From PR 1 on: `tests/test_<model>_upstream_equivalence.py` passes (bit-identical
  fp32 outputs for all variants and shared code paths; kernel alternatives with tolerance).
- From repo root: `pixi run -e <pkg>-dev --frozen lint`, `typecheck`, `deadcode`, `tests` all green — and confirm
  the new paths are actually inside the package's `PYREFLY_TARGET` (root `pixi.toml`), else typecheck is vacuous.
- Default `pytest -q` runtime stays in seconds; checkpoint tests are `-m slow`.

## Gate 4 — Hand-roll audit (last PR before pushing)

Grep the diff for the usual suspects and replace with the shared implementation:

| hand-rolled | use instead |
|---|---|
| quaternion → matrix | `scipy.spatial.transform.Rotation.from_quat` (xyzw) |
| K scaling for resized images | `simplecv.camera_parameters.rescale_intri` |
| camera dicts / ad-hoc `K, D, R, t` tuples | `PinholeParameters`, `Fisheye62Parameters`, `Intrinsics.from_k_matrix`, `Extrinsics` |
| rig entity paths + transforms by hand | `simplecv.rig.Rig` + `log_rig_static` |
| fourcc / codec ints | `rr.VideoCodec(value)` |
| open3d mesh → `rr.Mesh3D` | `simplecv.rerun_log_utils.log_open3d_mesh` |
| own `rr.init`/spawn/save flags | `RerunTyroConfig` |
| TSDF loop | `simplecv.ops.tsdf_depth_fuser.Open3DFuser` |
| video decode | `simplecv.rerun_dataloader` / `SegmentNvdecDecoder` |

Keep a hand-rolled version only when no shared helper fits, and say so in the PR.

## Gate 5 — Viewer pixel evidence (before "done")

Use the `rerun-viewer-validation` skill: run with `--rr-config.headless --rr-config.save <rrd>`, open the
`.rrd` in a viewer, take screenshots that show the actual claim (frusta orientation, depth pointing out of the
right camera, mesh faces visible from inside, no error badges). Evidence lives in
`/tmp/rerun-viewer-validation/<date>-<model>/` (screenshots + the `.rrd`); the PR description links the
directory and embeds the key screenshots.
For a model joining an existing family, add a **same-frame comparison** against the incumbent (open both `.rrd`s,
`set_time` to the same `video_time`, screenshot both) — that, not the absolute look, is what the reviewer asks about.
Before a catalog-backed run, confirm the dataset is registered (see port.md PR 5).
Watch for: child `Transform3D` is relative to its *parent*, not the rig (a level mesh does not prove the tree);
`EncodedDepthImage` needs `depth_range` or renders purple; colour range 0–6 m for indoor scenes.
