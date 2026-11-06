# Upstream-source recipes

Each directory here builds one upstream project, pinned to a single commit, as a conda package with
rattler-build (`recipe.yaml` beside a `[package]` manifest, backend `pixi-build-rattler-build`).
Environments consume it as a source dependency: `<name> = { path = "recipes/<name>" }`.
`packages/` holds only our own code; slam-rs and dpretrieval stay there (see the follow-up below).

| Package | Upstream, pinned source | Licence | Patches | Consuming features |
| --- | --- | --- | --- | --- |
| `asmk-src` | [jenicek/asmk](https://github.com/jenicek/asmk) git `2a96d9c` | not declared in the recipe | 2 | `mast3r-slam` (linux-64) |
| `brush-src` | [ArthurBrussee/brush](https://github.com/ArthurBrussee/brush) git `1388f74` | not declared in the recipe | 1 | `gsplat-rust-renderer` |
| `dinov3-src` | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) archive `5aa7e93`, sha256 `194fc7a6…` | DINOv3 License (`LICENSE.md`) | 0 | `posekit`, `mv-api`, `mamma`, `exo-calib`, `selfcap-curate`, `posekit-sam3d-body` |
| `egohand-src` | [NVIDIA/IsaacCapture](https://github.com/NVIDIA/IsaacCapture) archive `479063c` (`src/python/isaaccapture_egopost`), sha256 `49dd5594…`; installs `egohand` and `egohand_inference` | OpenMDW-1.1 (+ `THIRD_PARTY_LICENSES.md`) | 6 | `posekit`, `mv-api`, `mamma`, `exo-calib`, `selfcap-curate` |
| `hexels-src` (+ `hexel-kernel`, `fused-ssim`, `simple-knn`, `image-gs`) | [honeycomb-surfels/hexels](https://github.com/honeycomb-surfels/hexels) git `698d1af`, its submodules and baselines at the revisions upstream records | not declared in the recipe | 0 (applies upstream's own 2DGS `adaptations.patch`, sha256-checked) | `hexels` |
| `mast3r-src` | [naver/mast3r](https://github.com/naver/mast3r) git `f5209af` (+ DUSt3R, CroCo submodules) | not declared in the recipe | 5 (+ 2 replaced files) | `mast3r-slam` (linux-64) |
| `sam-3d-body-src` | [facebookresearch/sam-3d-body](https://github.com/facebookresearch/sam-3d-body) archive `b5c765a`, sha256 `71199eaf…`; installs `sam_3d_body` | SAM License (`LICENSE`) | 3 | `posekit-sam3d-body` |

Bump a recipe:

1. Change the pin (`rev`, or `url` + `sha256`) and the `version` in `recipe.yaml` and the `[package]` manifest.
2. `rattler-build debug setup --recipe recipes/<name>/recipe.yaml` checks out the source with the current
   patches applied; fix each patch that no longer applies in that tree, then
   `rattler-build debug create-patch` writes it back to `patches/`.
3. Install every consuming environment (`pixi install -e <env>`) and run its gate.

## Rules and traps

- Name every recipe `<upstream>-src`. Python import names stay upstream's.
- Pin the source by `url` + `sha256` (an archive or raw file) or by `git` + `rev`, never a branch or tag.
- Ship the upstream licence file (`about.license_file`).
- The build script starts with `rm -rf` on its target: pixi-build reuses the host prefix between rebuilds, and a
  plain `cp -r` nests a second copy inside the stale one.
- A run dependency on a sibling recipe only solves when the environment lists that sibling too: list the
  sibling in every feature that lists the recipe.
- `pixi lock --check` does not see edits to a patch or to `recipe.yaml` (it records the rendered metadata, not
  the file contents). The consuming package's goldens and byte proofs are the check.

## Follow-up

- `packages/dpretrieval` is DPVO's DPRetrieval copied into the repo; it could become a pinned git source here.
- `asmk-src`, `brush-src`, `hexels-src` and `mast3r-src` do not declare their upstream licence yet.
