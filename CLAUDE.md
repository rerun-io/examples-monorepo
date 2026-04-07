# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A **Pixi workspace monorepo** of computer vision projects. Each package lives in `packages/<name>/` with its own Python module, CLI tools, and tests. All Pixi configuration (deps, tasks, environments) lives in the root `pixi.toml` — per-package `pyproject.toml` files only have standard Python packaging metadata.

## Environments

Each package has a prod env (`<name>`) and a dev env (`<name>-dev`, adds ruff, pytest, beartype, pyrefly). Direnv auto-activates the `*-dev` env when you `cd` into a package directory.

## Commands

```bash
# With direnv active (cd into a package dir first):
ruff check .        # lint
pytest -q           # test
pyrefly check .     # typecheck

# From repo root (needed for tasks with depends-on chains):
pixi run -e monoprior --frozen relative-depth   # runs download + demo
pixi run -e robocap-dev --frozen tests
```

Prefer `pixi run --frozen` to skip re-solving deps. Only omit `--frozen` when you've modified dependencies.

## Architecture

**Beartype** is activated conditionally via `PIXI_DEV_MODE` in each package's `__init__.py`:
```python
import os
if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package
    beartype_this_package()
```

**Package structure:**
```
packages/<name>/
  pyproject.toml    # [project], [build-system], [tool.ruff]
  <module>/
    __init__.py     # Beartype activation
    apis/           # High-level inference interfaces
    gradio_ui/      # Gradio components (if applicable)
  tools/            # CLI scripts (demos/ and apps/ subdirs)
  tests/
```

**Ruff** — line length 150, rules: E, F, UP, B, SIM, I. Ignored: E501, F722/F821 (jaxtyping), UP037/UP040, SIM901.

**pyrefly** config is monorepo-wide in root `pyrefly.toml`. Do not add `[tool.pyrefly]` to per-package `pyproject.toml`.

## Gotchas

- **Never use pip** — all dependency management goes through Pixi
- **`hf download` not `huggingface-cli`** — conda's huggingface_hub provides `hf`, not `huggingface-cli`
- **gradio from PyPI, not conda** — conda's gradio package has missing transitive deps
- **`moge` needs no-build-isolation** — it requires torch at build time (configured in `[pypi-options]`)
- **sam3d-body-rerun uses `tool/` (singular)** not `tools/` for its CLI scripts
- **Direnv fails after changing `pixi.toml`** — run `pixi install -e <name>-dev` to re-solve, then direnv picks up the updated lockfile
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
- **Pixi collapses multiline `cmd = """..."""` into a single line**, replacing newlines with spaces. If a task has separate commands on different lines (e.g. `export`, `echo`, `python`), they become arguments to the first command and never execute. The task appears to succeed (exit 0) but produces no output. Always use `&&`-chained single-line commands or `\` line continuations instead.
