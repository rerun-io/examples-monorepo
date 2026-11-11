---
name: zerogpu-pixi-space
description: >
  Put a monorepo Gradio + Rerun app on a Hugging Face ZeroGPU Space, installed with pixi (no Docker,
  uv or pip): probe the GPU features first, pin our packages by commit, prebuild TensorRT engines,
  iterate in Dev Mode, size the GPU request from measured clicks, and prove it with screenshots of the
  Space page. Use when the user says "put <app> on a Space", "ZeroGPU", "HF Space", or "deploy the
  Gradio app to Hugging Face".
---

# A monorepo app on a ZeroGPU Space, the pixi way

The Space runs our code at a pinned monorepo commit inside a locked pixi env. The only file the
Gradio SDK sees is a standard-library `app.py` that installs pixi and execs `pixi run --frozen app`.
Worked example: the EgoHand hands app (`packages/posekit/src/posekit/hands_app_ui/`).

Templates: [references/app.py](references/app.py) (bootstrap, copy unchanged),
[references/pixi.toml](references/pixi.toml) (manifest skeleton),
[references/space_app.py](references/space_app.py) (the app module the `app` task runs).

## Facts that shape the design

- ZeroGPU works only with the **Gradio SDK**, not Docker. The main process never has a GPU; every
  `@spaces.GPU` call runs in a forked worker. Nothing may initialise CUDA in the main process.
- The GPU is an **RTX PRO 6000 Blackwell MIG slice** (default size: half the card, 48 GB, 94 SMs),
  compute capability 12.0. NVFP4/FP8 TensorRT work. TensorRT, NVDEC (torchcodec CUDA) and CUDA graphs
  work in the worker. **NVENC does not** (`libnvidia-encode` is not mounted; pixi cannot supply a
  driver library), so any video encode must fall back to CPU.
- ZeroGPU **drops an idle worker after a short time**. The next click forks a fresh worker from the
  main process and pays its setup again. A startup warm-up only helps clicks soon after boot.
- A worker forked from the main process **inherits its imports and RAM**: import heavy modules
  (and read CPU-side assets) in the main process before the first GPU call.
- **Quota:** a visitor is charged the requested `duration` × 1.5 up front. Logged-out visitors get
  about 2 min a day per IP, free accounts 5, PRO 40. A run is aborted about 2 s past its request.
- ZeroGPU allows **10 Spaces per user**. Check free slots before you plan; moving a Space to
  `cpu-basic` frees one (ask the owner which).
- A **private** Space renders inside the HF page through a signed iframe; page reloads in your head
  script and direct `*.hf.space` links do not work for private Spaces in a normal browser. Prefer
  public, or test private ones only on the HF page while logged in.
- `torch.compile` re-traces in every fresh worker (15 s on EgoHand, for 5 % fps). Use eager on the
  Space unless the compile artifacts ship with it.

## Steps

1. **Scope.** Decide with the user: public or private, which stacks/models, the GPU seconds per click
   (start from "a fresh worker must finish inside it"), and where engines come from. Count the free
   ZeroGPU slots (`HfApi().list_spaces(author=..., expand=["runtime"])`).
2. **Probe Space first.** A throwaway Space whose one `@spaces.GPU` generator exercises *every* GPU
   feature the real app uses: TensorRT build + run, NVDEC decode, NVENC encode, CUDA graph replay,
   torch.compile, Triton, warp kernels, plus `torch.cuda.get_device_capability()` in the main process
   and the worker pid on two quick calls (reuse). Add one long `@spaces.GPU` call at import time
   (startup) that writes a file a later worker reads. Twenty minutes here replaces guesses with facts.
3. **Manifest.** Copy [references/pixi.toml](references/pixi.toml). Take the dependency list from the
   app's monorepo env **feature by feature**, the whole `cuda` feature included (Triton needs
   `cuda.h`); do not trim CUDA dev packages by guess. Our packages are PyPI git deps and the upstream
   recipes are pixi-build git source deps, all pinned to one **pushed** monorepo commit. Copy
   `conda-pypi-map-ai-demos.json` with only the entries the app needs. Lock on linux-64 in a host-side
   terminal with `CONDA_OVERRIDE_CUDA=13.0`.
4. **App module.** Copy [references/space_app.py](references/space_app.py). `import spaces` first;
   wrap the app's run generator in one `@spaces.GPU(duration=...)`; pass Space settings (stacks,
   examples dir, frame cap) as `build()` parameters instead of patching module globals. If you must
   replace a module function, keep the original in a name first (a wrapper that calls the name it
   replaced recurses).
5. **Repo hygiene.** `.gitignore` and `upload_folder(ignore_patterns=...)` cover `.pixi/`,
   `.ruff_cache/`, `__pycache__/` from the first push. README front matter: `sdk: gradio`,
   `app_file: app.py`, `startup_duration_timeout: 1h`, `short_description` under 60 characters.
6. **Example clips.** `ffprobe` them: Rerun's `Mp4Reader` transcodes clips with B-frames, which costs
   CPU time inside the GPU reservation. Re-encode once with `-bf 0` (H.264, yuv420p, same size and fps).
7. **Engines off the Space.** Build TensorRT engines on a local sm_120 card (an RTX 5090 is the same
   GB202 chip) with the Space's own env and an **empty** engine cache, then upload the engines, their
   manifests and the ONNX files they are keyed on to a public HF model repo. The Space downloads them
   on the CPU before launch (`snapshot_download(..., local_dir=<engine cache>)`). Prove a download
   into an empty cache builds nothing (zero "building" log lines). TensorRT warns about the SM count
   and device model on every load; it ran without failure on EgoHand.
8. **Local proof, clean.** Run the exact Space code on a local GPU (`@spaces.GPU` is a no-op off
   ZeroGPU) with **every cache empty**: Triton, Inductor, engine, warp, HF. Set `PYTHONUNBUFFERED=1`
   and wait on the port, not a log line. One example through the browser; read the screenshot.
9. **Push, then Dev Mode at once.** `POST /api/spaces/<id>/dev-mode {"enabled": true}`. RUNNING in
   Dev Mode means the container is up, not the app: poll `/config`. Iterate over SSH
   (`hf spaces ssh <id>`; `tar | ssh`, `scp` fails): edit, kill the old app tree by pid, check
   `ps --forest` for orphans, restart. 22–48 s per try instead of a ~6 min rebuild. **Never print
   `env`** in the container: it holds the Space's signing secret.
10. **Profile a fresh worker** as soon as one run works: a separate `@spaces.GPU` script timing setup,
    first chunk, second chunk. Fix the largest cost first (torch.compile, asset downloads, mesh or
    model construction that could run on the CPU before the fork).
11. **Measure three clicks in a browser**, each labelled: after boot, warm, and after ≥5 min idle (a
    fresh worker). `gradio_client` batches generator updates, so it cannot time "first output".
    Size `duration` from the fresh-worker time plus margin; cap frames to fit.
12. **Test logged out**, on the public HF page, on the final commit only (each click spends
    `duration` × 1.5 of a ~2 min daily IP budget). Count how many clicks a visitor gets.
13. **Cold-boot proof.** Factory restart; the log must show the engine download and no build.
14. **Done means pixels.** A screenshot you have read, of the HF Space page (not just `*.hf.space`,
    not an HTTP 200), showing the viewer with the model output after 20–40 s. Turn Dev Mode off and
    check once more after the rebuild.
15. **Keep code and Space in step.** Commit UI changes before you look at the Space; move the pin,
    re-lock and push in one step. Do not hot-reload a shared local app while someone tests it.

## End state

Swap the git deps for conda packages on `https://prefix.dev/ai-demos` (rattler-build recipes for our
packages; see `recipes/README.md`). That is a `pixi.toml`-only change in the Space.
