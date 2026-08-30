# Phase 2 — Port into the monorepo

Target package is normally `packages/monoprior` (`monopriors`). Branch names `<model>/<n>-<stage>`;
each PR bases on the previous one. Keep every PR reviewable on its own.

## PR 1 — `1-vendor`: upstream inference subset, as-is

- Copy only the inference modules to `monopriors/third_party/<model>/` plus the upstream `LICENSE`.
  Docstring in `__init__.py`: upstream URL, license (flag non-commercial), fork + SHA, file mapping
  (`ours.py <- upstream/path.py`), "Local changes: none yet".
- Make it importable: absolute imports, drop `sys.path` hacks; nothing else changes.
- Register the vendor dir as *unowned* in `pyrefly.toml` (`project-excludes`) and in the package
  `pyproject.toml` (ruff `extend-exclude`, vulture `exclude`, package-data for LICENSE).
- Deps: add to the package feature in root `pixi.toml` (conda first, PyPI second); keep `platforms`
  explicit. `pixi install -e <pkg>-dev` then commit `pixi.lock`.
- Copy the same upstream files a second time to `tests/reference_data/<model>/upstream_*.py` — pristine
  fixtures for the equivalence test in PR 3 (also excluded from lint/typecheck).

## PR 2 — `2-predictor`: the contract

- `monopriors/models/<family>/<model>.py`: `<Model>Predictor(Base<Family>Predictor)` with
  `__init__(device, model_size=..., checkpoint: Path | None = None, ...)`, checkpoint download from the pinned
  HF revision, strict `load_state_dict`, preprocessing mirroring upstream's demo exactly (padding, scaling,
  normalisation, `iters`/`test_mode`), and `__call__` returning the family's prediction dataclass
  (stereo: `StereoDepthPrediction(disparity, depth_meters, K_33, baseline_m)` via `disparity_to_metric_depth`).
- Register: `<FAMILY>_PREDICTORS` Literal + `get_<family>_predictor` in `models/<family>/__init__.py`.
  Existing demos/apps/catalog tools select by `predictor_name`; if one still hardcodes a class, switch it to
  the registry in this PR (that is how the tools gain the model).
- Tests: fast CPU test that builds the model from config and runs a tiny random pair (shape/dtype/finite);
  slow band (`pytestmark = [slow_cuda, requires_cuda]`) that downloads the checkpoint and checks the
  reference number on the ETH3D sample (validate.md gate 2). Default `pytest -q` must stay seconds.
- Demo: if the family has no `tools/demos/<family>.py`, add one (thin shim over `monopriors/apis/<family>.py`
  with `RerunTyroConfig`) plus a `pixi.toml` task with a `_download` dependency.

### Pickled-module checkpoints (`torch.load(..., weights_only=False)` on a whole `nn.Module`)

Do not import upstream at its original path and do not re-host converted files. Load with a custom
`pickle.Unpickler` whose `find_class` remaps upstream module paths (`core.foundation_stereo` →
`monopriors.third_party.<model>.foundation_stereo`, `Utils` → ...) onto the vendored modules, then take
`.state_dict()` and load it strictly into a freshly built module. Cover it with a test on the real file (slow).
Inspect the pickle first (`zipfile` + `re.findall(rb'core\.[A-Za-z_.]+', data.pkl)`): it also references the
config classes the module holds (e.g. `omegaconf.dictconfig`), which then become a dependency, and a checkpoint
serialised by older code may lack attributes the current `forward()` reads — measure each candidate default
against the reference sample instead of guessing (FFS: `args.normalize` missing; True → 0.48 % bad1, False → 45 %).

## PR 3 — `3-typed`: owned fork

- Apply python-conventions to the vendored subset: jaxtyping shapes on tensors, TypedDict/dataclass configs,
  Google docstrings, absolute imports, remove training-only paths, dead code, unused helpers.
- Module/parameter names and `state_dict` keys must not change.
- `tests/test_<model>_upstream_equivalence.py`: load the pristine fixtures as a synthetic package
  (`importlib.util.spec_from_file_location` under a fake package name), build upstream and ours from the same
  config, copy the state dict, assert bit-identical outputs on seeded random pairs for every variant and
  every optional path (e.g. triton kernel vs torch fallback, hierarchical vs plain).
- Flip the vendor dir to *owned*: remove it from `pyrefly.toml`/ruff/vulture excludes; add vulture
  `ignore_names` for framework-read config keys instead of reworking code.
- Update the `__init__.py` docstring: "Local changes vs upstream" + the re-sync recipe (copy upstream to
  fixtures, re-apply annotations, run the equivalence test).

## PR 4 — `4-app` (only if the family has no Gradio app)

Follow `posekit`/`stereo_depth_ui.py`: Radio-driven panels, `gradio_rerun` streaming viewer, ints typed
`float | int`, no `np.float32` defaults (not JSON-serialisable), `--host` default `127.0.0.1`, launched with the
sandbox off and exposed with `tailscale serve --bg --https=<port> http://127.0.0.1:<port-1>` (the embedded
viewer needs a secure context).

## PR 5 — `5-catalog` (only if the family has no catalog tool)

Follow `monopriors/apis/stereo_catalog.py` / `promptda_polycam`: read cameras + poses from the catalog with
`simplecv.rerun_dataloader` (`open_segment_decoder`, NVDEC), build simplecv camera dataclasses
(`Fisheye62Parameters`, `PinholeParameters`), rectify with `models/stereo_depth/rectify.py`, log the rig with
`log_rig_static`, relay video with `send_columns`, `EncodedDepthImage` (16-bit mm, `depth_range`) for depth,
`Open3DFuser` + `log_open3d_mesh` for incremental TSDF, `RerunTyroConfig` for the sink. Stream to the viewer;
do not register a layer unless asked.

## Stacking and merging

Open PRs bottom-up with `--base` on the previous branch. When merging: merge with a merge commit in order,
retarget each next PR to `main` (`gh pr edit N --base main`) right before merging it, delete branches after.
