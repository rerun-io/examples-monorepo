# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## What This Is

A **Pixi workspace monorepo** of computer vision projects. Runnable Python projects live in `packages/<name>/` with their modules, CLI tools, and tests; the directory also contains build-only dependencies and vendored code. Root-managed dependencies, tasks, and environments live in `pixi.toml`. Runnable packages keep standard Python packaging metadata and package-specific tooling config such as `[tool.ruff]` and `[tool.vulture]` in `pyproject.toml`; Pixi-build packages such as `asmk`, `dpretrieval`, and `mast3r` have their own build manifests.

## Environments

Each root-managed runnable package has a prod env (`<name>`) and a dev env (`<name>-dev`, adds ruff, pytest, beartype, pyrefly, hypothesis, vulture). The dev env exposes the tasks `lint`, `typecheck`, `deadcode`, and `tests` (e.g. `pixi run -e <name>-dev tests`). In package directories that contain a `.envrc`, direnv auto-activates the `*-dev` env when you enter the directory.

## Commands

```bash
# With direnv active (cd into a package dir first):
ruff check .        # lint
pytest -q           # test
pyrefly check .     # typecheck

# From repo root (needed for tasks with depends-on chains):
pixi run -e monoprior --frozen monoprior-relative-depth   # runs download + demo
pixi run -e robocap-slam-dev --frozen tests
```

Prefer `pixi run --frozen` to skip re-solving deps. Only omit `--frozen` when you've modified dependencies. The dev tasks (`pixi run -e <name>-dev {lint,typecheck,deadcode,tests}`) are the canonical runners — `typecheck` applies the monorepo `pyrefly.toml` plus any per-package baseline, whereas a bare `pyrefly check .` skips the baseline and can surface known false-positives.

## Platforms & lockfile

The workspace `platforms` list defines the full platform vocabulary: the plain `linux-64`, `linux-aarch64`, and `osx-arm64` CPU/macOS subdirs plus the named `linux-64-cuda13` and `linux-aarch64-cuda13` platforms, which carry the CUDA 13.0 and glibc 2.35 virtual packages. A feature opts into a platform by listing its plain subdir or named workspace platform. An environment only solves a platform that every one of its features allows. For a `common`-composing package to run on macOS, `common` and `dev` also list `osx-arm64`, with an osx-scoped `pytorch-cpu` (simplecv imports torch at module load).

**Every linux-only feature MUST declare `platforms = ["linux-64", "linux-aarch64"]` explicitly.** Since pixi 0.71 (PR prefix-dev/pixi#6178), a feature that omits `platforms` defaults to the entire workspace list — including `osx-arm64` and the named CUDA platforms. An env's platforms are the intersection of its features' lists, so one omitting feature can demand unintended macOS or CUDA solves; if that demand is unsolvable (CUDA deps like `libcublas`), **every `pixi install -e <any-env>`/`pixi lock` in the whole workspace aborts** on the next lock write. Solvable missing demands get silently solved and added to the lock instead. When adding a feature, copy the `platforms` line from an existing linux-only feature (e.g. `mv-api`).

**Whole-workspace lock generation.** Resolution never needs package Python at build time: every in-repo package exposes static metadata, and anything that would need a build step is vendored or prebuilt instead. What limits a host is pixi's build dispatch, which instantiates every env that contains editable or git PyPI deps on the current machine and picks any of that env's platforms to do so. Every env has a linux-64 lane, so **a linux-64 host regenerates the whole-workspace lock**. A linux-aarch64 host can re-lock changes scoped to envs that have an aarch64 lane; a change to a shared feature (`cuda`, `common`, `dev`) also invalidates the linux-64-only envs and `pixi lock` fails with `build dispatch initialization failed: the environment '<env>' does not support 'linux-aarch64' on this machine`. Re-lock such changes on a linux-64 host: apply the change in a worktree there, run `CONDA_OVERRIDE_CUDA=13.0 pixi lock`, copy `pixi.lock` back, and confirm with `pixi lock --check`. macOS cannot regenerate the full lock for the same reason: linux-only envs cannot exist on osx.

## Architecture

**Beartype** is activated conditionally via `PIXI_DEV_MODE` in each runnable Python package's `__init__.py`:
```python
import os
if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package
    beartype_this_package()
```

**`tools/` scripts must be thin shims** — `beartype_this_package()` only instruments
code **inside** the package, so the Tyro `Config` dataclass and `main()` belong in the
package (e.g. `<module>/apis/<name>.py`), and the `tools/` script just wires them up.
Logic placed directly in a `tools/` script is **not** beartype-checked under dev.
```python
# tools/apps/<name>.py  — keep it to a few lines
import tyro
from <module>.apis.<name> import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
```

**Typical runnable package structure:**
```
packages/<name>/
  pyproject.toml    # [project], [build-system], [tool.ruff]
  <module>/
    __init__.py     # Beartype activation
    apis/           # High-level interfaces + Tyro Config/main (beartype-instrumented)
    gradio_ui/      # Gradio components (if applicable)
  tools/            # THIN CLI shims over <module>/apis/ (demos/ and apps/ subdirs)
  tests/
```

## Adding a model

Bringing an upstream research model in (fork + pixify, then a vendor → predictor → typed PR stack with
Rerun pixel evidence) follows the repo skill `add-model` (`.agents/skills/add-model/SKILL.md`, also visible to
Claude Code through `.claude/skills`). Both Codex and Claude Code load it when asked to "add/port <model>".

## Adding a new package

1. Create `packages/<name>/` with `pyproject.toml`, the source module, `tools/`, and `tests/` (structure above). If it imports another workspace package, list that name in `[project].dependencies` **and** pin it in `[tool.uv.sources]` (`name = { path = "../<dir>", editable = true }`); `packages/simplecv/tests/test_workspace_sources.py` checks both and fails if the lock ever resolves a workspace name from PyPI.
2. Add `[feature.<name>]` in the root `pixi.toml`: conda deps, pypi deps (editable install), `activation.env` with `PACKAGE_DIR = "packages/<name>"`, and tasks with `cwd = "packages/<name>"`. Declare `platforms` explicitly (see **Platforms & lockfile**).
3. Add `<name>` and `<name>-dev` entries in `[environments]`, both with `solve-group = "<name>"` and `no-default-feature = true`; `<name>-dev` adds the `dev` feature.
4. Copy a package `.envrc` (defaults `PIXI_ENV` to `<name>-dev`) and add `packages/<name>/data/` to `.gitignore`.
5. Register the package in `pyrefly.toml` in three places (see **Code Style**).
6. Run `pixi install -e <name>-dev` to verify the solve.

## Code Style

Prefer straightforward inline code over tiny one-off helpers. Only extract a
function when it has meaningful reuse, hides real complexity, names a domain
concept, or improves testability. Thin wrappers that only pass through
arguments or hide a single call should usually be inlined at the call site.

**Ruff** — line length 150, rules: E, F, UP, B, SIM, I. Ignored: E501, F722/F821 (jaxtyping), UP037/UP040, SIM901.

**pyrefly** config is monorepo-wide in root `pyrefly.toml`; do not add `[tool.pyrefly]` to per-package `pyproject.toml`. When you add a package, register it in `pyrefly.toml` in **three** places: `search-path` and `site-package-path` (omit these and imports of the new module resolve to `missing-import`), and `project-includes` (omit it and the package's files aren't typechecked at all). For unavoidable stub false-positives from compiled/untyped deps (e.g. `depthai`), add a per-package `pyrefly-baseline.json` and wire it via `PYREFLY_EXTRA_ARGS = "--baseline pyrefly-baseline.json"` in `[feature.<name>.activation.env]` (see `simplecv`, `live-rerun`).

## Serialization

pyserde is the door through which data enters Python. Anything that crosses a boundary
gets a `@serde` dataclass that states what the data is: a file on disk, an HTTP response,
a dataset's own JSON / YAML / pickle, a catalog row, a model's output dictionary. Fields
are checked on the way in; the rest of the code holds typed objects, never dicts.

- **Records yes, streams no.** A record you read whole goes through pyserde, arrays
  included: calibrations, hand models, keypoint rows, reports, whole-sequence pose
  tables. A stream you iterate does not: video frames, depth maps, masks, point clouds
  and trajectories live in their own containers (npz/npy, Parquet/Arrow, Rerun) and
  pyserde carries only the metadata that names them. The test is the role of the data,
  not its size.
- **Arrays carry jaxtyping with an explicit dtype** (`Float32[ndarray, "n 3"]`, not
  `Float[...]` or bare `ndarray`): that is what makes pyserde enforce dtype and shape on
  load; generic `Float`/`Int` do not fix the width and a bare `ndarray` comes back as
  float64.
- **Formats for files we own:** JSON when a program writes it, TOML when a person edits
  it. YAML and pickle only when a third party hands us that format (the MANO pickles are
  model-asset adapters, not a precedent).
- **Strictness is per schema.** Files we own: `deny_unknown_fields=True`, and
  `type_check=coerce` for hand-written TOML (`from_toml` does not widen `30` to `30.0`;
  machine-written JSON stays strict). Third-party formats we read partially: unknown
  fields allowed, their schema is not ours to police.
- **Validation lives at the door.** Types come from the class. Cross-field rules go in
  `__post_init__` (pyserde runs it on load). Rules that need context live in one loader
  that wraps `SerdeError` (and the parser's own error, e.g. `TOMLDecodeError`) into a
  `ValueError` naming the source. Callers do not re-validate. Never catch a bare
  `Exception` around a decode: beartype violations must propagate.
- **Mechanics.** `@serde(...)` above `@dataclass(frozen=True, slots=True)`. Properties
  do not serialise: a report with computed columns gets a flat report dataclass at the
  write boundary. In our JSON outputs an unscored number is `X | None` written as `null`,
  never NaN (orjson); inputs that legitimately carry non-finite cells keep their custom
  field decoder. `tomli-w` and `orjson` are declared beside every `pyserde` declaration
  (`common` and friends): without `tomli-w` even `from_toml` fails to import.
- **Rust owns its formats.** Documents a Rust extension produces or consumes (slam-rs
  `Calibration`, `VioConfig`; gsplat's scene files) are parsed by the Rust serde derive;
  Python asks the extension for typed accessors instead of parsing `to_json()` output.
  The one sanctioned text-level exception is `slam_rs.reference.profiled_config_text`,
  which overlays profile keys onto the Rust-owned config JSON as text so the resolved
  bytes stay hashable.
- **Rollout.** No big-bang: convert a hand-rolled `json.load`/`yaml.safe_load` + dict
  indexing site when you are already editing that file, plus one deliberate pass per
  package when it has an owner. The full guidance with examples is in the
  `python-conventions` skill.

## Rerun Tools

When adding or updating Tyro-facing Rerun CLIs, prefer the shared `RerunTyroConfig`
from `simplecv.rerun_log_utils` instead of hand-rolling viewer/save/connect
flags or creating a local `rr.RecordingStream`. Add it as a nested dataclass
field such as `rr_config: RerunTyroConfig`, let its `__post_init__` configure
spawn/connect/save/serve/headless behavior, and then use the normal global
`rr.*` logging calls unless a test or library boundary specifically requires an
explicit recording stream. This preserves the flexible viewer and save behavior
expected across SimpleCV tools. For a realtime tool that needs the live viewer
**and** a `.rrd` at once, set `rr_config.live` together with `--rr-config.save`:
`RerunTyroConfig` then fans out to both via `set_sinks` (the `live`/`port` fields).

### OSS catalog file-descriptor limits

Rerun 0.34.1 keeps one descriptor open per registered `.rrd`. One ARKitScenes
dataset uses 5,015 × 7 = 35,105, so two exceed 65,536 during the second `gt`
layer. `arkitscenes-download-serve` uses `ulimit -n 524288`; restart the server
to inherit it. This is a capacity workaround—the upstream fix is open-on-demand
files or an LRU descriptor pool.

### The shared catalog server (:51235) is in-memory — never kill Rerun by name

`rerun server` keeps every registration in memory: any exit, even SIGTERM, loses
the whole corpus and re-registering takes ~90 s warm / ~16 min cold. It has died to
two things, both from agent sessions on this host:

- **Name-based kills** (`pkill -x rerun`, `pkill rerun`, `killall rerun`,
  `pkill -f rerun`) hit every process called `rerun`. Kill your own viewer by pid or
  with a pattern that includes your port, e.g. `pkill -f "rerun --headless --port 9877"`,
  and check `ss -ltnp | grep 51235` before touching anything named `rerun`. (The
  server currently runs through a `rerun-catalog` symlink so its process name is not
  `rerun`; that hides the symptom, it does not excuse a name-based kill.)
- **paseo restarts.** `paseo.service` uses `KillMode=control-group`, so a restart —
  including the one systemd triggers when the kernel OOM-kills any process in that
  cgroup — SIGTERMs every agent session and everything they spawned.
  `setsid`/`nohup`/`disown` do not leave the cgroup, and neither does `tmux` when the
  tmux server is started from an agent shell (it inherits the cgroup; only a tmux
  started from a real SSH login lives in the session scope). From an agent, start
  long-lived servers as a transient user unit, which sits beside `paseo.service`
  instead of inside it:
  ```bash
  systemd-run --user --unit rerun-catalog-9988 --collect --property=WorkingDirectory=$PWD \
    .pixi/envs/<env>/lib/python3.12/site-packages/rerun_sdk/rerun_cli/rerun server --port 9988
  systemctl --user status rerun-catalog-9988      # journalctl --user -u rerun-catalog-9988 for logs
  systemctl --user stop rerun-catalog-9988        # never pkill by name
  ```
  Verify with `cat /proc/<pid>/cgroup` (must not contain `paseo.service`). The unit is
  transient: a reboot drops it and the in-memory corpus, so keep a re-registration
  script for the layers on disk. Recreating the env that holds the binary is fine while
  the server runs, but the next start needs that path to exist.

Real fixes, not yet done: a persistent catalog store, and a persistent user unit that
re-registers from disk on start (then `Restart=on-failure` becomes safe).

## Testing Rerun builds

**Rust follows the primary Python Rerun lane.** The `common` / `rerun-prerelease`
PyPI `rerun-sdk` pin is the source of truth; the Rust `re_*` crates
(`packages/gsplat-rust-renderer/Cargo.toml`) must match it exactly, or the viewer
silently loses protocol/tooling parity. To bump: Python first, then the Rust pins
(matching that release's egui family), then re-lock Pixi and Cargo.

The primary workspace lane runs **`rerun-sdk == 0.37.0`** with the `catalog`
extra through `common`. Catalog/stream environments add the `dataloader` extra
through `rerun-prerelease`. `gradio-rerun` pins an exact `rerun-sdk`, so bump
both pins in `[feature.common.pypi-dependencies]` together, never separately.

Two macOS lanes restate that exact pin rather than composing `common`, because
`common` is linux-only (`rerun-sdk[dataloader]` needs a torchvision that does
not build on `osx-arm64`): `[feature.mv-api-catalog-register-mac]` and
`[feature.slam-rs-osx]`. Both are tagged `# pin-bump: hold`; a bump has to touch
all four sites or the Mac silently runs a release behind, and no linux gate and
no `pixi lock --check` can see it.

To test an **unreleased** Rerun build, add a `find-links` at
`build.rerun.io/commit/<sha>/wheels/` to `[feature.rerun-prerelease.pypi-options]` (CI builds one
per commit, including PR branches — `curl` the index first to confirm your platform; PR commits
are usually linux-x86_64 only) and match `rerun-sdk == <ver>` to the wheel filename. Re-lock on
linux-64 (pixi 0.70.x) and move back to a public release once the fix ships.

## Gotchas

- **Never use pip** — all dependency management goes through Pixi
- **`hf download` not `huggingface-cli`** — conda's huggingface_hub provides `hf`, not `huggingface-cli`
- **gradio from PyPI, not conda** — conda's gradio package has missing transitive deps
- **No dependency may need a build step during resolution** — `pixi lock` must never need package Python. Vendor the inference subset (as `monopriors/third_party/` does, with the upstream rev and patch list in the package `__init__` docstring), ship a prebuilt wheel, or give the package static metadata and build any CUDA extension through an explicit task.
- **sam3d-body uses `tool/` (singular)** not `tools/` for its CLI scripts
- **Direnv fails after changing `pixi.toml`** — run `pixi install -e <name>-dev` to re-solve, then direnv picks up the updated lockfile. A shared-feature change re-solves the whole workspace; that works from any Linux host, while macOS is limited to `pixi lock --check` (see **Platforms & lockfile**).
- **Never use bare `except Exception` with beartype** — it silently swallows type violations. Always re-raise `BeartypeException`:
  ```python
  from beartype.roar import BeartypeException
  try:
      result = some_typed_function()
  except BeartypeException:
      raise
  except Exception:
      print("failed")
  ```
- **Use `0.0` not `0` for float annotations** — beartype strictly distinguishes `int` from `float`. `last_error: float = 0` will fail; use `last_error: float = 0.0`
- **`vulture` (the `deadcode` task) flags framework-used names** — Tyro/dataclass config fields, `pytestmark`, `__exit__`'s `*exc`, etc. Add them to `[tool.vulture] ignore_names` in the package `pyproject.toml` rather than reworking the code.
- **Static shape checking is not on yet** — pyrefly checks jaxtyping shapes once the `shape_extensions` package resolves (the `tensor-shapes` config key is a no-op now); it comes from the PyPI stub packages `pyrefly-torch-stubs` and `pyrefly-numpy-stubs`, versioned in lockstep with pyrefly. Neither is in the `dev` feature: the numpy stubs exist only on the 1.3 line and still lack `einsum` and batched `@`, so enabling them today produces hundreds of coverage errors per numpy-heavy package. Until then beartype at function boundaries is the only check on array shapes and dtypes, which is why names must not repeat them (python-conventions skill). Never add a stub fixtures directory to `search-path`.
- **Pixi collapses multiline `cmd = """..."""` into a single line**, replacing newlines with spaces. If a task has separate commands on different lines (e.g. `export`, `echo`, `python`), they become arguments to the first command and never execute. The task appears to succeed (exit 0) but produces no output. Always use `&&`-chained single-line commands or `\` line continuations instead.
- **Don't poll with `pgrep -f <pat>` when the polling command itself contains `<pat>`** — it matches its own shell and the `until ! pgrep ...` loop never exits (silently hangs forever). Prefer `run_in_background` on the real command (you're notified on its own exit), or wait on a file/sentinel.
- **Always pass `--rr-config.headless` to Rerun CLIs in shells without `DISPLAY`** — the `RerunTyroConfig` default calls `rr.spawn()`; when the viewer fails to start (winit "neither WAYLAND_DISPLAY nor DISPLAY is set"), the recording stream's channel fills and every `rr.log()` blocks forever. The run wedges silently (zombie viewer child, zero CPU) instead of erroring out.
- **beartype's PEP 526 checks re-create jaxtyping hints on every annotated assignment** — under `beartype_this_package()` each `x: Float64[ndarray, "n 3"] = ...` inside a hot loop evaluates its hint again, jaxtyping returns a new class each time, and beartype compiles and caches a new checker for it: exo-calib's Stage B ran 2× slower in the dev env and its memory grew without bound over a 4,000-frame capture. Packages with array-heavy inner loops pass `BeartypeConf(claw_is_pep526=False)` (function boundaries stay checked). beartype caches its transformed bytecode as `__pycache__/*.opt-beartype*.pyc` keyed by beartype version, not by conf, so after changing the conf delete those files or the old checks stay in.
