---
name: add-model
description: >
  Bring an upstream research model (stereo, mono depth, normals, pose, ...) into this monorepo:
  fork + pixify the upstream repo with a Rerun demo, then port the minimum into a package as a
  typed, tested, reviewable PR stack that ends with pixel-verified Rerun output. Use when the user
  says "add/port/bring in <model>", "fork and pixify <repo>", or "make <method> a predictor".
  Works from Claude Code and Codex; nothing here depends on runtime-specific tools.
---

# Add a model to the monorepo

Two phases, five gates. Phase 1 is skippable when a pixified fork already exists.
Every phase reads one reference file; read it fully before starting that phase.

| Phase | Reference | Output |
|---|---|---|
| 0. Scope | (this file) | answers to the scope questions, written down |
| 1. Fork + pixify | [references/fork-and-pixify.md](references/fork-and-pixify.md) | public fork with frozen `main`, `pixi` branch, `pixi run demo`, `NOTES.md` |
| 2. Port | [references/port.md](references/port.md) | PR stack `1-vendor → 2-predictor → 3-typed (→ 4-app → 5-catalog)` |
| Gates | [references/validate.md](references/validate.md) | equivalence test, reference number, dev tasks green, viewer pixel evidence |

Worked examples with every gotcha hit so far: [references/example-liteanystereo.md](references/example-liteanystereo.md)
(the run this skill was distilled from) and [references/example-fast-foundationstereo.md](references/example-fast-foundationstereo.md)
(the first run *through* the skill: pickled NAS checkpoint, existing family).

## Rules that always bind

- **Pixi only.** Never pip/uv. Pins copy the monorepo (`python 3.12`, `cuda-version 13.0.*`,
  conda `pytorch-gpu`, `rerun-sdk` at the workspace pin, `timm`, `tyro`), so the port has no env surprises.
- **Do not hand-roll what exists.** Before writing any helper, look in `simplecv` first
  (camera dataclasses, `rescale_intri`, `Rig`/`log_rig_static`, `RerunTyroConfig`, `rerun_dataloader`,
  `Open3DFuser`, `log_open3d_mesh`, `rr.VideoCodec`, `scipy.spatial.transform.Rotation`).
  The port ends with an explicit hand-roll audit (validate.md, gate 4).
- **Python conventions**: beartype via `PIXI_DEV_MODE` claw (never `@beartype`), PEP 526 annotations
  everywhere, jaxtyping dtype+shape on every array, `TypeAlias` not PEP 695, tyro CLIs, `0.0` for float
  defaults, `einops` over reshape/permute chains, dataclass field docstrings, thin `tools/` shims.
  (Claude: `python-conventions`, `karpathy-guidelines`, `tdd` skills; Codex has the same rules.)
- **Minimal diff.** Vendor only the inference subset. No speculative flexibility. Upstream code is
  untouched in the fork; it is *owned* (typed, trimmed) only in the `3-typed` PR, with an equivalence test.
- **Rerun output is validated with pixels**, not logs (`rerun-viewer-validation` skill): screenshots of the
  demo, the app, and the catalog tool; `--rr-config.headless` + `--rr-config.save` in shells without `DISPLAY`.
- **Process hygiene.** Never `pkill -f`/`pgrep -f` a pattern that appears in the same command line
  (kills the tool's shell, exit 144); use `pkill -x`, a saved PID, `fuser -k <port>/tcp`, or tmux sessions.
  Never start ad-hoc HTTP servers for artifacts; Gradio apps go through `tailscale serve --https`.
- **Identities.** Forks live under the personal GitHub account (`GH_TOKEN=$(gh auth token --user pablovela5620) gh …`),
  monorepo PRs use the work account (`GH_TOKEN=$(gh auth token --user pablo-rerun) gh pr create …` — the active
  login is often the personal one and `gh pr create` then fails with "must be a collaborator"). HF mirrors under the
  personal HF account.
- **Durable execution.** Anything longer than a minute (env solves, catalog runs, Codex jobs) runs in a named tmux
  session with `python -u`, a log file, and a sentinel file on exit; background shells die with the agent session
  and leave half-written `.rrd`s that look complete. Poll the sentinel, never the process name.
- **Feedback loop.** Every delegated run (Codex or subagent) ends its report with a "skill discrepancies" list:
  anything unclear, missing, contradictory, or guessed. Fold each item into this skill before the next phase.

## Running as Codex (sandbox)

- The sandbox may hide the CUDA virtual package (`CONDA_OVERRIDE_CUDA=13.0` for solves) and make the default pixi
  cache read-only (`PIXI_CACHE_DIR=/tmp/<job>-pixi-cache`). Neither proves the GPU works: run the GPU gate outside
  the sandbox or report exactly which command the reviewer must run — never fake numbers.
- Work in a dedicated git worktree (see port.md "Setup"); never edit the reviewer's checkout.

## Phase 0 — Scope (write the answers into the fork's `NOTES.md` later)

1. **License.** MIT/Apache: mirror freely. NVIDIA Source Code License / research-only: allowed to vendor and
   mirror *with the license text*; flag non-commercial in the vendored `__init__.py` docstring, the fork's NOTES.md, and (own mirrors only) the HF card. Weights can carry a different license than the code (e.g. NVIDIA Open Model Agreement vs Source Code License) — record both.
2. **Weights.** Is there an official HF repo? Search the Hub (`curl 'https://huggingface.co/api/models?search=<name>'`)
   before mirroring anything. Pin the HF revision SHA (`curl .../api/models/<repo>` → `sha`) and record the file's SHA-256 in the loader test. Is the file a `state_dict` or a pickled module
   (`torch.load(..., weights_only=False)`)? Pickled modules need the unpickler remap recipe in port.md.
3. **Inference subset.** List the modules the demo actually imports. Everything else (training, export,
   ONNX/TRT, dataset loaders, visualisation utils) stays out.
4. **Hard deps.** Custom CUDA/Triton kernels (keep only with a pure-torch fallback + parity test), xformers,
   flash-attn, open3d. Check what the model package imports vs what the README says.
5. **Reference number.** Does upstream ship an eval script and a per-dataset number? If not, the monorepo's
   ETH3D `playground_1l` sample (HF `pablovela5620/monoprior-example`, `stereo/eth3d`) is the fallback: record
   the result as a *baseline*, not a reproduction.
6. **Upstream entry points.** Name the demo script (for `demo-upstream`), its preprocessing (scaling, padding divisor, normalisation, AMP dtype), the forward signature (`iters`, `test_mode`, output shape), and the minimum input size the network accepts (fixes the fast test's tiny pair).
7. **Contract.** Which existing predictor family does it join (`monopriors.models.stereo_depth`,
   `relative_depth`, `normals`, ...)? Joining an existing `Base*Predictor` + `get_*_predictor` registry is the
   goal; a new family needs its contract designed first (grill the user).
8. **Which PRs.** `1-vendor`, `2-predictor`, `3-typed` are always required. `4-app`/`5-catalog` only when the
   family has no app/catalog tool yet — an existing tool gains the new model through the registry.

Probe commands that answer 1–5 in minutes (run them, do not ask the user for these facts):

```bash
git clone -q --depth 1 https://github.com/<org>/<repo> /tmp/<model>-probe && cd /tmp/<model>-probe && git rev-parse HEAD
head -40 LICENSE*; grep -n -iE "commercial|research" LICENSE* | head           # code license
grep -rhoE "^(import|from) [a-zA-Z_0-9.]+" --include=*.py <model-dir> | sort | uniq -c | sort -rn   # real deps
curl -s 'https://huggingface.co/api/models?search=<name>' | python3 -c 'import sys,json;[print(m["id"]) for m in json.load(sys.stdin)]'
curl -s https://huggingface.co/api/models/<repo>/tree/main | python3 -c 'import sys,json;[print(f["path"],f.get("size")) for f in json.load(sys.stdin)]'
curl -s https://huggingface.co/api/models/<repo> | python3 -c 'import sys,json;print(json.load(sys.stdin)["sha"])'
python - <<'PY'   # is the .pth a state_dict or a pickled module, and which classes does it reference?
import zipfile,re; z=zipfile.ZipFile('<file>.pth'); d=z.read([n for n in z.namelist() if n.endswith('data.pkl')][0])
print(sorted(set(re.findall(rb'[A-Za-z_][A-Za-z_0-9]*\.[A-Za-z_0-9.]+', d)))[:60])
PY
```
When `forward()` reads an attribute the checkpoint lacks, A/B each candidate value against the reference sample's
ground truth (FFS `normalize`: True 0.48 % vs False 45 % bad1) — never pick by reading code alone.

Grill the user on anything ambiguous above (batch the questions, give recommendations) before phase 1.
