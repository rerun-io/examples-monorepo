# canonical-viewer

One Rerun viewer that holds every change we make to the viewer. Native, headless and web are built from one list
of extensions, one set of Rerun patches and one Rerun release. Pages, Gradio apps, tools, agents and people all open
this build. Design: `/mnt/nas/ssd-8tb/scratch/canonical-viewer/DESIGN.md`.

`rerun-canonical` is the stock `rerun` command line (viewer, `--headless`, `mcp`, `rrd`, SDK spawn) with every
extension in each viewer it opens. It leaves out analytics, the OSS server (`rerun server`; the catalog server stays on
stock `rerun`) and `--serve-web`.

## Use

```bash
# From the repo root.
pixi run -e canonical-viewer --frozen canonical-viewer-build                 # prepares the patched Rerun checkout, then cargo
pixi run -e canonical-viewer --frozen rerun-canonical -- recording.rrd       # any `rerun` arguments follow
# The web viewer, in the stock /rerun-viewer/<version>/ layout (shell files copied from a stock web viewer of the release):
pixi run -e canonical-viewer-web --frozen canonical-viewer-web-build <stock-0.44.0-dir> <out-dir>
```

`rust/crates/web` is the web entry point: it installs the same hooks before JavaScript constructs the stock `WebHandle`,
so pages keep the stock `@rerun-io/web-viewer` API. gsplat's page flags still work: `force_no_subgroups=1`,
`profile=1`. The web build is 17.8 MB brotli (stock 0.44.0: 10.6 MB); depth video adds 0.3 MB of it.

## Extensions

| Extension | Crate | Adds | Goldens |
|---|---|---|---|
| `gsplat` | `packages/gsplat-rust-renderer/crates/gsplat-viewer` (`setup::EXTENSION`) | compute Gaussian-splat renderer; `Spatial3DView` visualizers for `GaussianSplats3D` (compute), `Hexels3D`, SOG, SOG LOD and animated SOG; device needs: `SUBGROUP` when offered (not with `GSPLAT_NO_SUBGROUPS=1`), raised compute limits; replaces the stock 3D view's `EyeControls3D` position and look-target fallbacks (default framing on splat and hexel bounds) | `packages/gsplat-rust-renderer/tests/test_viewer_pixels.py` on `rerun-canonical` |
| `depth-video` | `packages/depth-video-viewer` (`EXTENSION`) | `DepthVideoStream` (depth as AV1 10-bit monochrome lossless video, exact 10-bit codes) in `Spatial2DView` (colormapped image) and `Spatial3DView` (depth cloud under a `Pinhole`); no device needs; on the web the same decoder (`re_rav1d`, no asm, one thread, on the viewer's thread) gives the same exact codes | `packages/depth-video-viewer/tests/test_viewer_pixels.py` on `rerun-canonical`, including the scene with gsplat splats |

## Kinds of viewer change

| Kind | Where | Rule |
|---|---|---|
| Extension: views, visualizers, renderers, loaders, archetypes, device needs | a library crate beside its domain package, exporting `pub const EXTENSION: Extension` (`rust/crates/extension`) | no `main`, no `App::new`, no Rerun patch; one line in `EXTENSIONS` (`rust/crates/app/src/lib.rs`) + one golden screenshot |
| Patch to a Rerun crate | `patches/`, listed in `RERUN_PATCHES` (`canonical_viewer/apis/prepare_rerun.py`) | only hooks, build fixes and dependency swaps; a patch over about 100 lines is logic and becomes an upstream proposal instead |
| Default | here (native + web) | one place per surface |

## How it works

`patches/re_viewer-0.44.0-extension-hooks.patch` adds two hooks to `re_viewer`, at the two places every viewer
(native, headless, web) passes through: the device request in `wgpu_options()` (it receives the adapter and Rerun's
device descriptor) and the end of `App::with_commands` (every new `App`). `canonical_viewer::install_hooks` points them
at `EXTENSIONS`, and `rerun-canonical` then runs the stock `rerun::run`.

A third hook, in `re_viewer_context`, keeps Rerun from warning (a toast on every start) when an extension replaces a
stock fallback that it declares in `replaces_fallbacks`. Each declared (view, component) pair has one owner across all
extensions (unit test); any other replacement still warns.

Each extension's device needs apply in list order to Rerun's device descriptor. A change is kept only when it keeps
everything already requested and stays within the adapter's offer (`with_device_needs`). An extension that fails to
register logs an error, and the other extensions still load.

`tools/prepare_rerun.py` checks out the Rerun tag into `target/patch/rerun-<tag>/` and applies the patches, and prepares
each patched crates.io crate (`PATCHED_CRATES`: wgpu with the WebGPU subgroups patch, web backend only) into
`target/patch/<crate>-<version>/` (gsplat's web build keeps a staged copy of this code until PR 7 points
`recipes/gradio-rerun-src` at this build). The Cargo workspace (`rust/`) redirects every Rerun crate
there. The checkout sits outside `rust/` because, inside it, cargo passes re_renderer's sources as relative paths and
re_renderer's embedded shader lookup breaks.

## Rerun patches

Applied in order by `tools/prepare_rerun.py` (`RERUN_PATCHES`) to the Rerun tag's checkout:

| Patch | What | Drop when |
|---|---|---|
| `re_viewer-0.44.0-extension-hooks.patch` | the device and `App::new` hooks (native, headless, web) and the declared-fallback hook | upstream has extension hooks (Pablo wants no upstream PRs, issues or proposal drafts) |
| `rerun-0.44.0-av1-arm64.patch` | native AV1 decoding on Linux arm64 (rerun-io/rerun#7755): `re_rav1d` 0.1.4 from a pinned git revision, whose arm64 assembly tables are hidden (the workspace's `[patch.crates-io]` gives depth-video-viewer the same `re_rav1d`) | a Rerun release fixes #7755 and `re_rav1d` 0.1.4 is on crates.io |

`tools/prepare_rerun.py` also applies `re_rav1d-0.1.4-wasm-libc.patch` to the AV1 patch's `re_rav1d` revision
(`PATCHED_GIT_CRATES`): `libc` has none of the C types and errno values rav1d names on wasm32-unknown-unknown, so a
15-line shim crate stands in for it there. Drop it when rav1d builds for wasm32 upstream.

The AV1 patch comes byte-identical from ai-demos `.github/rerun-source-build/` (main `acaa65f`), which builds the same fix
into its `rerun-sdk` wheels. The stalled-proxy-reader patch from the same place is not in this set while a test decides it
(Pablo, 2026-11-27).

## Distribution

| Surface | How |
|---|---|
| Native, any machine | the conda package `rerun-canonical` (`recipe/recipe.yaml`), installed by the fleet beside stock `rerun`; stock `rerun` stays the catalog server and the SDK's own CLI |
| Python spawn | `RerunTyroConfig` (simplecv) spawns `rerun-canonical` when it is on PATH, else `rerun`; the SDK warns when the viewer's release differs from its own |
| Stills, videos, posters | the `rerun-viewer-validation` skill's scripts default to `rerun-canonical` when it is on PATH |
| Web pages | `canonical-viewer-web-build`, then agent-fleet `rerun_viewer_install <tag>-canonical-<sha7> --from-dir <out> --canonical`: embed.js opens it on every page without an installed `data-version` |

Build the package from the repo root on each platform (linux-64, linux-aarch64, osx-arm64). brush-src builds first:
the gsplat Cargo workspace names Brush crates, and rattler-build gives later recipes the earlier outputs.

```bash
rattler-build build --recipe recipes/brush-src/recipe.yaml --recipe packages/canonical-viewer/recipe/recipe.yaml \
  --output-dir <out> -c https://prefix.dev/conda-forge
```

The package version is the Rerun tag (`ci/tests/test_rerun_pins.py` checks it), so the SDK's version check still
applies; raise `build.number` for every extension or patch change at the same tag.

## Tests

- Rust unit tests (`canonical-viewer-rust-check`): the device-needs merge (features and limits only), unique extension
  names, one owner per replaced fallback.
- `tests/test_prepare_rerun.py`: patches apply in order; the checkout is reused until the commit or a patch changes; every
  patch names the release or crate version it applies to.
- `tests/test_build_web.py`: the npm module factory and its `deinit`.
- `tests/test_rerun_canonical.py` (integration): `--version` is the SDK's release; `mcp`, `rrd` and `--help` exist.
- `tests/test_stock_parity.py` (golden): a 2D + 3D recording with no extension data renders pixel-identical in
  `rerun-canonical` and this environment's stock `rerun`.
