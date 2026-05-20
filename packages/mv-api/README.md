# mv-api

Raw HoCap full exo/ego pipeline port for the examples monorepo.

The package scope is CLI/API only. Gradio and client UIs are intentionally out of scope for the first port.

This package is intended to be run from the Pixi workspace. Its model packages
(`monopriors`, `wilor-nano`, and `rtmlib`) are resolved by the root `pixi.toml`
path/git sources rather than standalone PyPI releases.

## Run

The full pipeline is exposed through the node app entrypoint:

```bash
pixi run -e mv-api mv-api-full-app --rr-config.save artifacts/node-validation/hocap-node-app.rrd --rr-config.headless --max-frames 1 --camera-source dataset
```

## Catalog Notes

For the lean macOS catalog registration and query examples, do not use the
catalog optimization flags. The optimized cache path is confusing for this
workflow and can make quick iteration look like it is hung while it prepares
the full Assembly101 set. Prefer mounting/registering the existing local RRDs
directly.
