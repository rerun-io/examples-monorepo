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
```

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

Each extension's device needs apply in list order to Rerun's device descriptor. A change is kept only when it keeps
everything already requested and stays within the adapter's offer (`with_device_needs`). An extension that fails to
register logs an error, and the other extensions still load.

`tools/prepare_rerun.py` checks out the Rerun tag into `target/patch/rerun-<tag>/` and applies the patches (gsplat's web
build keeps a staged copy of this code until it moves onto the canonical web crate). The Cargo workspace (`rust/`) redirects every Rerun crate
there. The checkout sits outside `rust/` because, inside it, cargo passes re_renderer's sources as relative paths and
re_renderer's embedded shader lookup breaks.

## Tests

- Rust unit tests (`canonical-viewer-rust-check`): the device-needs merge and unique extension names.
- `tests/test_prepare_rerun.py`: patches apply in order; the checkout is reused until the commit or a patch changes.
- `tests/test_rerun_canonical.py` (integration): `--version` is the SDK's release; `mcp`, `rrd` and `--help` exist.
- `tests/test_stock_parity.py` (golden): a 2D + 3D recording with no extension data renders pixel-identical in
  `rerun-canonical` and this environment's stock `rerun`.
