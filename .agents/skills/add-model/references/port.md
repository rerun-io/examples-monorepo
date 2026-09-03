# Phase 2 — Port into the monorepo

Target package is normally `packages/monoprior` (`monopriors`). Branch names `<model>/<n>-<stage>`;
each PR bases on the previous one. Keep every PR reviewable on its own.

## Setup (once per port)

```bash
git worktree add /tmp/<model>/wt -b <model>/1-vendor <base-branch>
cd /tmp/<model>/wt && pixi install -e <pkg>-dev          # editable installs must point at THIS tree — do not symlink .pixi
pixi run -e <pkg>-dev python -c "import <module>; print(<module>.__file__)"   # must print /tmp/<model>/wt/...
ln -s <main-checkout>/packages/<pkg>/data/examples/<family> packages/<pkg>/data/examples/<family>   # gitignored samples
```
Run pytest from `packages/<pkg>` (root collection pulls in packages whose envs are absent). Use
`pixi run --frozen` after the first install; re-run `pixi install` only after changing deps.

## PR 1 — `1-vendor`: upstream inference subset, as-is

- Copy only the inference modules to `monopriors/third_party/<model>/` plus the upstream `LICENSE`.
  Docstring in `__init__.py`: upstream URL, license (flag non-commercial), fork + SHA, file mapping
  (`ours.py <- upstream/path.py`), "Local changes: import paths only" (absolute imports, no `sys.path` hacks, export/TRT-only classes dropped — nothing behavioural).
- Make it importable: absolute imports, drop `sys.path` hacks; nothing else changes.
- Exception forced by the beartype claw (it instruments `third_party/` too): annotations that lie (`mlp_ratio: int = 4` used as
  a float, `size: Tuple[int, int] = None`) get value-preserving fixes (`4.0`, `Optional[...]`) in PR 1 or PR 2, listed in the
  docstring as "beartype-compatible annotation fixes, values unchanged".
- Register the vendor dir as *unowned* in `pyrefly.toml` (`project-excludes`) and in the package
  `pyproject.toml` (ruff `extend-exclude`, vulture `exclude`, package-data for LICENSE).
- Deps: add to the package feature in root `pixi.toml` (conda first, PyPI second); keep `platforms`
  explicit. `pixi install -e <pkg>-dev` then commit `pixi.lock`.
- Copy the same upstream files a second time to `tests/reference_data/<model>/upstream_*.py` — pristine
  fixtures (excluded from lint/typecheck) — and add `tests/test_<model>_upstream_equivalence.py` already here
  (it passes trivially against the as-is copy); PR 3 must keep it passing.

## PR 2 — `2-predictor`: the contract

- `monopriors/models/<family>/<model>.py`: `<Model>Predictor(Base<Family>Predictor)` with
  `__init__(device, model_size=..., checkpoint: Path | None = None, ...)`, checkpoint download from the pinned
  HF revision, strict `load_state_dict`, preprocessing mirroring upstream's demo exactly (padding, scaling,
  normalisation, `iters`/`test_mode`), and `__call__` returning the family's prediction dataclass
  (stereo: `StereoDepthPrediction(disparity, depth_meters, K_33, baseline_m)` via `disparity_to_metric_depth`).
- Register model-specific options in a per-model config dataclass next to the
  predictor (`LiteAnyStereoConfig(model_size=...)`, `FastFoundationStereoConfig(valid_iters=...)`, each with
  `setup(device) -> Predictor`), combined into a tyro subcommand union exactly like
  `simplecv/configs/exoego_dataset_configs.py` (defaults dict → `tyro.extras.subcommand_type_from_defaults` →
  `tyro.conf.OmitSubcommandPrefixes`); tools take one `predictor:` union field, and existing demos/apps/catalog
  tools gain the model through that registry. Never put a model-specific flag
  (a `model_size`, an iteration count) flat on a tool's config — review feedback on FFS PR #162.
- A family's first model: tyro cannot build a subcommand union from a one-entry defaults dict — alias the union to that config
  at runtime (base class under `TYPE_CHECKING`) and switch to `subcommand_type_from_defaults` when the second model lands.
- Upstream private helpers that fix a frame convention (LAMP `MpsLoader._compute_T_gravityWorld_world`) are ported verbatim
  with a parity test against the pristine fixture — never re-derived.
- New package instead of a family (LAMP → `packages/lamp`, module `lamptrack`): its env composes `common` + `cuda` + the
  package feature (+ `posekit` for 2D people stages); a `<pkg>-catalog` lane mirrors `monoprior-catalog`; the two catalog rig
  readers are copied from `monopriors/apis/stereo_catalog.py` (no monoprior dependency) pending a shared simplecv reader.
- Tests: fast CPU test that builds the model from config and runs a tiny random pair (shape/dtype/finite);
  slow band (`pytestmark = [slow_cuda, requires_cuda]`) that downloads the checkpoint and checks the
  reference number on the ETH3D sample (validate.md gate 2). Default `pytest -q` must stay seconds.
- Demo: if the family has no `tools/demos/<family>.py`, add one (thin shim over `monopriors/apis/<family>.py`
  with `RerunTyroConfig`) plus a `pixi.toml` task with a `_download` dependency.

### Pickled-module checkpoints (`torch.load(..., weights_only=False)` on a whole `nn.Module`)

Do not import upstream at its original path and do not re-host converted files. Load with a custom
`pickle.Unpickler` subclass (wrapped in a module-like namespace for `torch.load(..., pickle_module=...)`) whose
`find_class` remaps upstream module paths (`core.foundation_stereo` →
`monopriors.third_party.<model>.foundation_stereo`, `Utils` → ...) onto the vendored modules, then take
`.state_dict()` and load it strictly into a freshly built module. Cover it with a test on the real file (slow).
Inspect the pickle first (`zipfile` + `re.findall(rb'core\.[A-Za-z_.]+', data.pkl)`): it also references the
config classes the module holds (e.g. `omegaconf.dictconfig`), which then become a dependency, and a checkpoint
serialised by older code may lack attributes the current `forward()` reads — measure each candidate default
against the reference sample instead of guessing (FFS: `args.normalize` missing; True → 0.48 % bad1, False → 45 %).
Compare the unpickled module's `state_dict` against a config-built module before promising "build from config +
strict load": a NAS-pruned / distilled release (FFS: 210 tensors only in the pickle, 54 shape changes, distillation
wrapper modules) cannot be described by its `cfg.yaml`. Then the pickle *is* the architecture spec: the loader
unpickles + deep-copies + strictly reloads, the typed fork types the classes without renaming modules, classes or
parameters (the remap and the state_dict keys depend on them), the fast equivalence test compares config-built
upstream vs owned nets, and a slow test unpickles the real file onto both packages and compares outputs.

## PR 3 — `3-typed`: owned fork

- Apply python-conventions to the vendored subset: jaxtyping shapes on tensors, TypedDict/dataclass configs,
  Google docstrings, absolute imports, remove training-only paths, dead code, unused helpers.
- Module/parameter names and `state_dict` keys must not change.
- `tests/test_<model>_upstream_equivalence.py`: load the pristine fixtures as a synthetic package
  (`importlib.util.spec_from_file_location` under a fake package name), build upstream and ours from the same
  config, copy the state dict, assert bit-identical outputs (fp32, no autocast) on seeded random pairs for every
  checkpoint/config variant and every code path that has both an upstream and an owned implementation
  (hierarchical vs plain, iteration counts). Alternative kernels (triton vs torch volume) are *parity* checks
  with a tolerance, not bit-identity, and are asserted separately.
- Flip the vendor dir to *owned*: remove it from `pyrefly.toml`/ruff/vulture excludes **and add it to the
  package's `PYREFLY_TARGET` list in root `pixi.toml`** (monoprior enumerates its typechecked paths; a path
  missing there is silently never checked); add vulture `ignore_names` for framework-read config keys instead of
  reworking code.
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
do not register a layer unless asked. Catalog registrations are in-memory: after a server restart the dataset is
gone (`LookupError: No dataset found with name 'robocap'`) — re-register (`/mnt/nas/datasets/robocap/rrd/reregister.py`)
before blaming the tool.

Catalog-tool rules added by the X-Lens/LAMP runs: fisheye rigs get rectified pinhole twins for depth (SKILL.md "Rerun geometry
rules"); `start_s` is absolute `video_time` seconds (say so in the field docstring); decoded+resized fisheye frames are logged
as `rr.Image(...).compress(...)` (the `send_columns` relay is for untouched pinhole streams); print ms per frameset with
views × resolution; ended tracks are cleared; the 60-s evidence rrd stays well under 500 MB.

## PR description (every PR)

Stacked-on line; what the PR adds in 3–6 bullets; **numbers** (reference metric vs paper/incumbent, warm ms/frame,
resolution); **gates** as run (lint/typecheck/deadcode/tests counts, slow tests); licenses when non-permissive;
for the last PR the evidence directory and the key screenshots (frusta, depth, mesh, same-frame comparison).

## Stacking and merging

Open PRs bottom-up with `--base` on the previous branch. A fix that belongs to a lower PR goes on that PR's branch
(cherry-pick), then replay everything above it (`git rebase <lower>` per branch, or `git rebase --onto` for
worktrees based on an old tip) and `git push --force-with-lease` — a fix on the top branch hides the defect in the
reviewed PR. When merging: merge with a merge commit in order,
retarget each next PR to `main` (`gh pr edit N --base main`) right before merging it, delete branches after.
