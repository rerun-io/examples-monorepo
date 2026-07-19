# Computer Vision Examples Monorepo

A Pixi workspace of computer vision applications, reusable libraries, and
source-built dependencies. Runnable projects live under `packages/` and compose
shared features with package-specific dependencies in the root `pixi.toml`.
Most runnable packages have a production environment (`<name>`) and a
development environment (`<name>-dev`) with the same dependency solve.

The workspace targets `linux-64` and `linux-aarch64`. A small number of package
features also support Apple Silicon; check the package README and its feature in
`pixi.toml` before installing on macOS.

## Quick start

1. Install [Pixi](https://pixi.prefix.dev/latest/installation/), then restart
   your shell:

   ```bash
   curl -fsSL https://pixi.sh/install.sh | sh
   ```

2. Clone the workspace and install one development environment:

   ```bash
   git clone https://github.com/rerun-io/examples-monorepo.git
   cd examples-monorepo
   pixi install -e monoprior-dev --frozen
   ```

3. Run that package's demo task from the repository root:

   ```bash
   pixi run -e monoprior-dev --frozen monoprior-relative-depth
   ```

Choose another package from the map below, then use its README or
`pixi task list -e <name>` to find its demo task. Tasks with download
dependencies fetch their example assets on first run.

### Optional: direnv

The root `.envrc` activates `dev`. Package directories that contain their own
`.envrc` default to `<name>-dev`, so after installing
[direnv](https://direnv.net/) you can opt in per directory:

```bash
cd packages/monoprior
direnv allow
pytest -q
```

Set `PIXI_ENV` in a gitignored `.envrc.local` to override a directory's default.

## Package map

| Package | What it contains |
| --- | --- |
| [arkitscenes-download](packages/arkitscenes-download/) | ARKitScenes downloader and ingest pipeline that emits layered Rerun recordings. |
| [asmk](packages/asmk/) | Pixi-build recipe for the ASMK image-retrieval dependency used by MASt3R-SLAM. |
| [dpretrieval](packages/dpretrieval/) | Pixi-build recipe for a DBoW2/pybind11 image-retrieval extension used by DPVO. |
| [dpvo](packages/dpvo/) | Deep Patch Visual Odometry with Rerun and Gradio integrations. |
| [egoexo-forge](packages/egoexo-forge/) | Rerun and Gradio tools for egocentric and exocentric human datasets. |
| [gsplat-rust-renderer](packages/gsplat-rust-renderer/) | GPU Gaussian-splat viewer implemented as a custom Rerun visualizer. See its [README](packages/gsplat-rust-renderer/README.md). |
| [live-rerun](packages/live-rerun/) | Zero-transcode live H.264/H.265 sensor streaming into Rerun. |
| [mamma](packages/mamma/) | Streaming multiview body capture from decode through SMPL-X fitting and Rerun logging. |
| [mast3r](packages/mast3r/) | Pixi-build recipe bundling MASt3R, DUSt3R, and CroCo. |
| [mast3r-slam](packages/mast3r-slam/) | Dense visual SLAM built on MASt3R reconstruction priors. |
| [monoprior](packages/monoprior/) | Monocular relative depth, metric depth, surface-normal, and calibration tools. |
| [mv-api](packages/mv-api/) | Full egocentric/exocentric multiview processing for raw HOCap datasets. |
| [posekit](packages/posekit/) | Design survey and comparison artifacts for a reusable pose-pipeline abstraction. |
| [prompt-da](packages/prompt-da/) | Prompt Depth Anything depth completion for Polycam captures. |
| [pysfm](packages/pysfm/) | COLMAP/pycolmap structure-from-motion with Rerun visualization. |
| [pyvrs-viewer](packages/pyvrs-viewer/) | VRS-to-Rerun conversion with compressed video and sensor streams. |
| [robocap-slam](packages/robocap-slam/) | Multicamera visual odometry and SLAM using NVIDIA cuVSLAM. |
| [sam2-streaming](packages/sam2-streaming/) | Vendored, inference-only SAM2 fork with frame streaming and bounded-memory banks. |
| [sam3](packages/sam3/) | Text-conditioned SAM3 image and video segmentation with Rerun. |
| [sam3d-body](packages/sam3d-body/) | Promptable SAM3D Body reconstruction and visualization playground. |
| [sapiens-coco133-pose](packages/sapiens-coco133-pose/) | COCO-133 human pose pipelines using Sapiens2 or RTMLib backends. |
| [sapiens2-pose](packages/sapiens2-pose/) | Top-down 308-keypoint human pose estimation with Rerun. |
| [simplecv](packages/simplecv/) | Shared Python utilities for datasets, geometry, video, and Rerun logging. |
| [slam-evals](packages/slam-evals/) | VSLAM benchmark ingestion and browsing through a Rerun catalog. |
| [vistadream](packages/vistadream/) | Single-image 3D reconstruction and Gaussian-scene tooling. |
| [wilor-nano](packages/wilor-nano/) | Hand detection and 3D hand-pose estimation with Rerun logging. |

Build-only and vendored packages do not necessarily have standalone root
environments or demo tasks; their role is described in the second column.

## Conventions

- Root-managed runnable packages generally use `<name>` for production and
  `<name>-dev` for development. The `[environments]` table in `pixi.toml` is
  the source of truth.
- Development environments expose the canonical `lint`, `typecheck`,
  `deadcode`, and `tests` tasks:

  ```bash
  pixi run -e robocap-slam-dev --frozen lint
  pixi run -e robocap-slam-dev --frozen typecheck
  pixi run -e robocap-slam-dev --frozen deadcode
  pixi run -e robocap-slam-dev --frozen tests
  ```

- Prefer `pixi run --frozen` and `pixi install --frozen` while dependencies are
  unchanged. Omit `--frozen` only when intentionally updating the solve and
  lockfile.
- List the tasks available in an environment with
  `pixi task list -e <name>` or `pixi task list -e <name>-dev`.
