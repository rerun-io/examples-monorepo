"""Convert ARKitScenes sequences to layered RRDs on Modal and upload them to HuggingFace.

Worker = download ONE sequence from Apple's CDN → run the package's own download/ingest
tools inside the image's pixi env (same lockfile as the local run) → land the seven
verified layer RRDs on the staging volume, layer-first → die. One ``drain_to_hf``
process batch-uploads staging to the HF repo. Idempotency is destination-existence
(is ``gt/<id>.rrd`` staged or on HF), so there is no shared state anywhere — kill and
relaunch freely.

Entrypoints (run from ``packages/arkitscenes-download``):

    # benchmark both encoder paths on the same sequences (uploads under bench/):
    pixi run -e arkitscenes-download modal run -m arkitscenes_download.modal_jobs.convert_sequences::benchmark

    # the full run (only after the benchmark verdict + explicit approval):
    pixi run -e arkitscenes-download modal run --detach \\
        -m arkitscenes_download.modal_jobs.convert_sequences::full_run --encoder gpu --confirm

The GPU function (L4) takes the ``av1_nvenc`` path of ``ingest.mov``'s encoder table;
the CPU function falls through to ``libsvtav1`` naturally — no code forks.
"""

from __future__ import annotations

import modal

from arkitscenes_download.modal_jobs import (
    ENV_BIN,
    HF_REPO_ID,
    PACKAGE_DIR,
    STAGING_MOUNT,
    arkitscenes_image,
    hf_credentials,
    staging_volume,
)

app = modal.App("arkitscenes-rrd-convert", image=arkitscenes_image)

HOUR = 60 * 60

# Mirrors pipeline.ASSETS (the local-NAS orchestrator, deliberately untouched here).
ASSETS: tuple[str, ...] = (
    "mov",
    "annotation",
    "mesh",
    "lowres_wide.traj",
    "confidence",
    "lowres_depth",
    "lowres_wide_intrinsics",
    "ultrawide_intrinsics",
    "highres_depth",
)

# Sequences validated end-to-end by the local pipeline — known-good inputs, so
# benchmark differences are attributable to the compute, not the data.
BENCH_IDS: tuple[str, ...] = ("40753679", "40753686", "40776203", "40776204", "40777060")

# Workers never UPLOAD to HF (staging-volume architecture) — the skip check below
# still makes one HF metadata call per unstaged sequence, which stays far under the
# resolve-bucket limits. Concurrency is bounded only by GPU budget:
# 32 workers ≈ 2.5 h for the full corpus at ~70 s/sequence.
MAX_CONTAINERS = 32


def _run_tool(script: str, args: list[str]) -> None:
    """Run one of the package's tools/ shims inside the pixi env, loudly on failure."""
    import os
    import subprocess

    # No pixi activation in the container: put the env's binaries (ffmpeg, rerun,
    # curl) on PATH ourselves, the way `pixi run` would.
    env = {**os.environ, "PATH": f"{ENV_BIN}:{os.environ['PATH']}", "FFMPEG_PATH": f"{ENV_BIN}/ffmpeg"}
    result = subprocess.run(
        [f"{ENV_BIN}/python", f"tools/apps/{script}", *args],
        cwd=PACKAGE_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{script} {args[:2]} failed (rc={result.returncode}):\n{result.stdout[-1500:]}\n{result.stderr[-3000:]}")


def _convert_and_upload(video_id: str, prefix: str, overwrite: bool) -> dict:
    """Download one sequence, ingest it, and land the layer RRDs.

    Destination is the staging volume (the drain uploads to HF); a sequence is
    skipped when it already sits in staging OR already made it to the HF repo.
    Returns timing/size metrics.
    """
    import shutil
    import tempfile
    import time
    from pathlib import Path

    from huggingface_hub import HfApi

    # Production (prefix="") stages layer-first, matching the HF repo layout since the
    # 2026-07-23 migration: <layer>/<video_id>.rrd. Bench legs (prefix="bench/*") keep
    # the old sequence-subdir shape — they're scratch and never registered.
    layer_first = not prefix
    dest = video_id if layer_first else f"{prefix}/{video_id}"
    gt_relpath = f"gt/{video_id}.rrd" if layer_first else f"{dest}/gt.rrd"
    if not overwrite:
        if (Path(STAGING_MOUNT) / gt_relpath).exists():
            print(f"skip (staged): {dest}")
            return {"video_id": video_id, "skipped": True}
        if HfApi().file_exists(HF_REPO_ID, gt_relpath, repo_type="dataset"):
            print(f"skip (on HF): {dest}")
            return {"video_id": video_id, "skipped": True}

    metrics: dict = {"video_id": video_id, "skipped": False}

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"

        t0 = time.perf_counter()
        _run_tool(
            "download.py",
            ["--download-dir", str(data_dir), "--video-ids", video_id, "--assets", *ASSETS, "--no-include-point-clouds"],
        )
        raw_dirs = list((data_dir / "raw").glob(f"*/{video_id}"))
        metrics["raw_bytes"] = sum(p.stat().st_size for d in raw_dirs for p in d.rglob("*") if p.is_file())
        metrics["download_s"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        _run_tool(
            "ingest_sequence.py",
            ["--video-id", video_id, "--data-dir", str(data_dir), "--output", str(data_dir / "rrd")],
        )
        out_dir = data_dir / "rrd" / video_id
        layer_files = sorted(out_dir.glob("*.rrd"))
        if len(layer_files) != 7:
            raise RuntimeError(f"{video_id}: expected 7 layers, got {[f.name for f in layer_files]}")
        metrics["rrd_bytes"] = sum(f.stat().st_size for f in layer_files)
        metrics["ingest_s"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        if layer_first:
            # Per-file tmp+rename, gt LAST: the skip probe keys on gt, so a partially
            # landed sequence never looks complete.
            ordered = sorted(layer_files, key=lambda f: f.stem == "gt")
            for f in ordered:
                dest_file = Path(STAGING_MOUNT) / f.stem / f"{video_id}.rrd"
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                tmp_file = dest_file.with_suffix(".rrd.tmp")
                shutil.copyfile(f, tmp_file)
                tmp_file.rename(dest_file)
        else:
            # Atomic-ish land on the staging volume: temp dir, then rename into place.
            staged = Path(STAGING_MOUNT) / dest
            tmp_staged = staged.with_name(staged.name + ".tmp")
            shutil.rmtree(tmp_staged, ignore_errors=True)
            shutil.copytree(out_dir, tmp_staged)
            shutil.rmtree(staged, ignore_errors=True)
            tmp_staged.rename(staged)
        staging_volume.commit()
        metrics["upload_s"] = time.perf_counter() - t2
        metrics["total_s"] = time.perf_counter() - t0

        shutil.rmtree(data_dir, ignore_errors=True)
    print(f"done {dest}: {metrics}")
    return metrics


@app.function(
    gpu="L4",
    cpu=8,
    memory=16384,
    timeout=2 * HOUR,
    secrets=[hf_credentials],
    max_containers=MAX_CONTAINERS,
    volumes={STAGING_MOUNT: staging_volume},
)
def convert_gpu(video_id: str, prefix: str = "", overwrite: bool = False) -> dict:
    """L4 worker: `av1_nvenc` wins the encoder probe."""
    return _convert_and_upload(video_id, prefix, overwrite)


@app.function(
    cpu=16,
    memory=16384,
    timeout=2 * HOUR,
    secrets=[hf_credentials],
    max_containers=MAX_CONTAINERS,
    volumes={STAGING_MOUNT: staging_volume},
)
def convert_cpu(video_id: str, prefix: str = "", overwrite: bool = False) -> dict:
    """CPU worker: nvenc probe fails, `libsvtav1` fallback engages."""
    return _convert_and_upload(video_id, prefix, overwrite)


@app.function(
    cpu=4,
    memory=8192,
    timeout=24 * HOUR,
    secrets=[hf_credentials],
    max_containers=1,
    volumes={STAGING_MOUNT: staging_volume},
)
def drain_to_hf(idle_exit_passes: int = 6) -> None:
    """Single uploader: batch-push the staging volume to the HF repo, forever-ish.

    `upload_large_folder` batches many files per commit and handles rate limits +
    resume via its own cache (kept on the volume). Loops with `reload()` so files
    landed after a pass get picked up; exits after `idle_exit_passes` consecutive
    passes that found nothing new to do (i.e. converters have long finished).
    """
    import time
    from pathlib import Path

    from huggingface_hub import HfApi

    api = HfApi()
    idle = 0
    while idle < idle_exit_passes:
        staging_volume.reload()
        before = sum(1 for _ in Path(STAGING_MOUNT).glob("gt/*.rrd"))
        if before == 0:
            time.sleep(60)
            continue
        api.upload_large_folder(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            folder_path=STAGING_MOUNT,
            # Mirrors ingest.layers.LAYER_NAMES — deliberately not imported: the slim
            # container interpreter can't import the ingest package (rerun/rich-heavy).
            allow_patterns=["base/**", "calibration/**", "depth/**", "gt/**", "imu/**", "video_ultrawide/**", "video_wide/**", "bench/**"],
            print_report=False,
        )
        staging_volume.commit()  # persist upload_large_folder's resume cache
        staging_volume.reload()
        after = sum(1 for _ in Path(STAGING_MOUNT).glob("gt/*.rrd"))
        idle = idle + 1 if after == before else 0
        print(f"drain pass done: {after} sequences in staging, idle_passes={idle}")
        time.sleep(120)


def _summarize(label: str, results: list[dict]) -> None:
    done = [r for r in results if r and not r.get("skipped")]
    if not done:
        print(f"{label}: no completed conversions")
        return
    total_gb = sum(r["rrd_bytes"] for r in done) / 1024**3
    print(f"\n== {label} ({len(done)} sequences)")
    print(f"{'video_id':>10} {'download_s':>11} {'ingest_s':>9} {'upload_s':>9} {'total_s':>8} {'rrd_MB':>7}")
    for r in done:
        print(
            f"{r['video_id']:>10} {r['download_s']:>11.1f} {r['ingest_s']:>9.1f}"
            f" {r['upload_s']:>9.1f} {r['total_s']:>8.1f} {r['rrd_bytes'] / 1024**2:>7.0f}"
        )
    mean_total = sum(r["total_s"] for r in done) / len(done)
    mean_ingest = sum(r["ingest_s"] for r in done) / len(done)
    print(f"mean: total {mean_total:.1f}s  ingest {mean_ingest:.1f}s  rrd out {total_gb:.2f} GiB")


@app.local_entrypoint()
def benchmark(ids: str = "") -> None:
    """Run BOTH encoder paths over the same sequences; uploads land under bench/<leg>/."""
    id_list = [i for i in ids.split(",") if i] or list(BENCH_IDS)
    print(f"benchmarking {len(id_list)} sequences on gpu(L4/nvenc) and cpu(16-core/svt-av1) → {HF_REPO_ID}/bench/")
    gpu_results = list(convert_gpu.starmap([(vid, "bench/gpu", True) for vid in id_list]))
    cpu_results = list(convert_cpu.starmap([(vid, "bench/cpu", True) for vid in id_list]))
    _summarize("GPU L4 / av1_nvenc", gpu_results)
    _summarize("CPU 16-core / libsvtav1", cpu_results)
    # Assumed on-demand rates; the dashboard is the source of truth.
    l4_rate, cpu_core_rate, mem_gib_rate = 0.80 / 3600, 0.135 / 3600, 0.024 / 3600
    for label, results, rate in (
        ("GPU", gpu_results, l4_rate + 8 * cpu_core_rate + 16 * mem_gib_rate),
        ("CPU", cpu_results, 16 * cpu_core_rate + 16 * mem_gib_rate),
    ):
        done = [r for r in results if r and not r.get("skipped")]
        if done:
            per_seq = sum(r["total_s"] for r in done) / len(done) * rate
            print(f"{label}: ~${per_seq:.3f}/seq → ~${per_seq * 5047:.0f} for 5,047 (assumed rates)")


@app.local_entrypoint()
def convert(ids: str, encoder: str = "gpu", prefix: str = "bench/gpu", overwrite: bool = False) -> None:
    """Convert an explicit id list on one encoder leg (e.g. extra benchmark samples)."""
    fn = {"gpu": convert_gpu, "cpu": convert_cpu}[encoder]
    id_list = [i for i in ids.split(",") if i]
    results = list(fn.starmap([(vid, prefix, overwrite) for vid in id_list]))
    _summarize(f"{encoder} / {prefix}", results)


@app.local_entrypoint()
def full_run(encoder: str = "gpu", limit: int = 0, overwrite: bool = False, confirm: bool = False) -> None:
    """Fan the full conversion out (detached). Requires --confirm — approval-gated."""
    if not confirm:
        raise SystemExit("Refusing: the full run is approval-gated. Pass --confirm once green-lit.")
    import tempfile
    from pathlib import Path

    from arkitscenes_download.download_dataset import load_metadata

    fn = {"gpu": convert_gpu, "cpu": convert_cpu}[encoder]
    with tempfile.TemporaryDirectory() as tmp:
        ids = sorted(load_metadata(Path(tmp)))
    if limit > 0:
        ids = ids[:limit]
    print(f"spawning {len(ids)} {encoder} workers → staging volume → {HF_REPO_ID} layer-first (existing sequences skip)")
    # spawn_map takes parallel iterators (map-style), unlike starmap's tuple rows.
    fn.spawn_map(ids, [""] * len(ids), [overwrite] * len(ids))
    drain_to_hf.spawn()
    print("spawned converters + one drain; watch the Modal dashboard. Safe to re-run — completed sequences skip.")
