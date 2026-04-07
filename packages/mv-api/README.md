# mv-api

Multiview exo/ego pose estimation and calibration workflows for Rerun `.rrd` captures, packaged for the `examples-monorepo`.

This first pass brings in the RRD-centered surfaces from upstream `mv-api` and aligns them to this repo's package, dependency, and task conventions. The retained workflows are:

- full exo/ego RRD pipeline
- exo-only calibration
- batch exo calibration
- Gradio RRD app
- CLI smoke client for the Gradio queue flow

Deferred from this first pass:

- label UI
- face blur service
- video-only and hand-only experimental entrypoints
- upstream standalone Pixi environments

## Quick Start

```bash
pixi run -e mv-api mv-api-full-pipeline-app
```

Use another terminal to exercise the queue client:

```bash
pixi run -e mv-api mv-api-rrd-client -- --help
```

## Tasks

```bash
pixi task list -e mv-api
pixi task list -e mv-api-dev
```

Primary tasks:

- `mv-api-full-pipeline-app`: Launch the Gradio app for `.rrd` inputs
- `mv-api-rrd-client`: Submit a job to a running Gradio app
- `mv-api-exo-only-calib`: Run exo-only calibration directly
- `mv-api-batch-calib`: Batch-calibrate episode trees
- `mv-api-validate`: Validate retained imports, tools, and app construction

## Layout

```text
packages/mv-api/
├── src/mv_api/
│   ├── api/
│   │   ├── full_exoego_pipeline.py
│   │   ├── exo_only_calibration.py
│   │   └── batch_calibration.py
│   ├── gradio_ui/
│   │   └── full_pipeline_rrd_ui.py
│   ├── hand_keypoints.py
│   ├── multiview_pose_estimator.py
│   ├── robust_triangulate.py
│   └── coco133_layers.py
├── tools/
│   ├── app_full_pipeline_rrd.py
│   ├── run_rrd_client_example.py
│   ├── run_exo_only_calib.py
│   ├── batch_exo_calib_client.py
│   └── validate_mv_api.py
└── tests/
```

## Development

```bash
pixi install -e mv-api-dev
pixi run -e mv-api-dev lint
pixi run -e mv-api-dev tests
pixi run -e mv-api-dev mv-api-validate
```

## Notes

- This package uses the monorepo's shared `common`, `cuda`, and `dev` features rather than carrying upstream's standalone `pixi.toml`.
- `monopriors` and `wilor-nano` are resolved from local monorepo packages via `tool.uv.sources`.
- No example dataset is bundled in the first pass; the app and tools expect user-provided `.rrd` data.

See [docs/state-machine.md](docs/state-machine.md) for the workflow diagram and script inventory.
