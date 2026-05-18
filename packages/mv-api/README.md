# mv-api

Raw HoCap full exo/ego pipeline port for the examples monorepo.

The initial package scope is CLI/API only. Gradio and legacy client UIs are intentionally out of scope for the first port.

This package is intended to be run from the Pixi workspace. Its model packages
(`monopriors`, `wilor-nano`, and `rtmlib`) are resolved by the root `pixi.toml`
path/git sources rather than standalone PyPI releases.
