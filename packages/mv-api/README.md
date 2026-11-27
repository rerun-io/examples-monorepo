# mv-api

Raw HoCap full exo/ego pipeline port for the examples monorepo.

The package scope is CLI/API only. Gradio and client UIs are intentionally out of scope for the first port.

This package is intended to be run from the Pixi workspace. Its model packages
(`monopriors`, `posekit` for WiLoR hands, and `rtmlib`) are resolved by the root `pixi.toml`
path/git sources rather than standalone PyPI releases.

## Run

The full pipeline is exposed through the node app entrypoint:

```bash
pixi run -e mv-api mv-api-full-app --rr-config.save artifacts/node-validation/hocap-node-app.rrd --rr-config.headless --max-frames 1 --camera-source dataset
```

## Catalog Notes

The catalog tools (`mv-api-catalog-prediction-layer` and the macOS
`mv-api-catalog-register`) connect to an **already-running** Rerun catalog via
`CatalogClient` — they no longer mount the catalog in-process. Before running
either, serve the catalog and register the base recordings with dataforge:

```bash
pixi run -e simplecv-catalog --frozen simplecv-catalog-serve       # tier 1: serve the catalog (rerun server)
pixi run -e dataforge --frozen dataforge-register --catalog-url rerun+http://127.0.0.1:9988 assembly101  # tier 2
```

Tier 2 registers dataforge's converted `assembly101` recordings (convert them first) on the
server the tools use; for another dataset, pass its subcommand here and its name to the mv-api tool.

Both tools default to `--catalog-url rerun+http://127.0.0.1:9988`. Prediction
layers are attached with `on_duplicate=REPLACE`, so re-running is idempotent.

The prediction dataloader samples each exo stream at its **native frame rate** via the
stock `RerunIterableDataset` + `FixedRateSampling` — sub-native decimation is avoided
because the sampling grid drops reference packets and can make AV1 decode fail or return
wrong pixels (RR-5087). See
[`docs/catalog_dataloader.md`](docs/catalog_dataloader.md).
