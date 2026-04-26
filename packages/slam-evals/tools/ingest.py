#!/usr/bin/env python3
"""Ingest sequences from a VSLAM-LAB benchmark root into per-stream layer RRD files.

Each sequence becomes a directory ``<out>/<dataset>/<seq>/`` with one ``.rrd``
file per source data stream (calibration, groundtruth, rgb_<i>, depth_<i>,
imu_<i>). Layer files share the sequence's ``recording_id`` so the catalog
composes them into a single segment.

NVENC has a small per-process session limit; with ``--workers > 1`` the GPU
encoder occasionally rejects ``avcodec_open2(hevc_nvenc)`` with
``ExternalError [Errno 542398533]``. We detect those failures by name and
retry the affected sequence serially at the end of the run, so the corpus
finishes 109/109 without manual cleanup.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import tyro
from beartype.roar import BeartypeException
from tqdm import tqdm

from slam_evals.data import discover_sequences
from slam_evals.data.discovery import filter_sequences
from slam_evals.data.types import Modality, Sequence
from slam_evals.ingest import applicable_layers, ingest_sequence

# Substring on the formatted exception that identifies a transient NVENC
# session-contention failure. The raw error type is av.error.ExternalError
# (errno 542398533, AVERROR_EXTERNAL); the encoder name in the message is
# what makes the match unambiguous.
_NVENC_RETRY_MARKERS: tuple[str, ...] = (
    "avcodec_open2(hevc_nvenc)",
    "avcodec_open2(h264_nvenc)",
    "avcodec_open2(av1_nvenc)",
)


@dataclass
class IngestConfig:
    benchmark_root: Path = field(default_factory=lambda: Path("/home/pablo/0Dev/work/VSLAM-LAB-Benchmark"))
    """VSLAM-LAB-style benchmark root. Walked + classified at every invocation."""

    out: Path = field(default_factory=lambda: Path("data/slam-evals/rrd"))
    """Output directory. Each sequence writes layer files to ``<out>/<dataset>/<name>/<layer>.rrd``."""

    only: tuple[str, ...] = ()
    """Restrict to specific slugs or sequence names (e.g. ``EUROC/MH_01_easy`` or ``MH_01_easy``)."""

    datasets: tuple[str, ...] = ()
    """Restrict to specific datasets (e.g. ``EUROC KITTI``)."""

    modalities: tuple[Modality, ...] = ()
    """Restrict to specific modalities."""

    layers: tuple[str, ...] = ()
    """Restrict to specific layer names (e.g. ``rgb_0`` to only re-encode camera 0). Default: all applicable to each sequence."""

    force: bool = False
    """Re-ingest layer files that already exist."""

    limit: int | None = None
    """Cap the number of sequences processed (post-filter)."""

    workers: int = 4
    """Process pool size. NVENC is GPU-bound so over-subscribing past ~4 doesn't help; ``1`` runs serially."""


def _load_sequences(cfg: IngestConfig) -> list[Sequence]:
    return discover_sequences(cfg.benchmark_root.expanduser().resolve())


def _select_layers_for(sequence: Sequence, *, requested: tuple[str, ...]) -> set[str] | None:
    """Resolve ``--layers`` against what's applicable for this sequence.

    Returns ``None`` to mean "all applicable" (the orchestrator's default
    behavior). Returns an empty set if the user asked for layer names that
    don't apply to this sequence — caller should skip the sequence.
    """
    if not requested:
        return None
    applicable = set(applicable_layers(sequence))
    return applicable & set(requested)


def _ingest_one(seq: Sequence, out_dir: Path, layers: set[str] | None) -> tuple[str, float, int]:
    """Worker entrypoint: ingest a single sequence; return (slug, wall_seconds, num_layer_files_written)."""
    t0 = time.monotonic()
    written = ingest_sequence(seq, out_dir, layers=layers)
    return seq.slug, time.monotonic() - t0, len(written)


def _is_nvenc_session_flake(err_message: str) -> bool:
    return any(marker in err_message for marker in _NVENC_RETRY_MARKERS)


def _seq_dir(out_dir: Path, seq: Sequence) -> Path:
    return out_dir / seq.dataset / seq.name


def _existing_layers(seq_dir: Path) -> set[str]:
    if not seq_dir.is_dir():
        return set()
    return {p.stem for p in seq_dir.glob("*.rrd")}


def _run_serial(
    todo: list[tuple[Sequence, set[str] | None]],
    *,
    out_dir: Path,
    desc: str,
) -> tuple[list[tuple[str, str, float, int]], list[tuple[str, str]]]:
    """Run ``todo`` sequentially. Returns (oks, fails). Each ok = (slug, modality, dur, num_layers)."""
    oks: list[tuple[str, str, float, int]] = []
    fails: list[tuple[str, str]] = []
    for seq, layers in tqdm(todo, desc=desc):
        try:
            slug, dur, num_written = _ingest_one(seq, out_dir, layers)
        except BeartypeException:
            raise
        except Exception as exc:  # noqa: BLE001
            fails.append((seq.slug, f"{type(exc).__name__}: {exc}"))
            tqdm.write(f"FAIL {seq.slug}: {exc}")
            continue
        oks.append((slug, str(seq.modality), dur, num_written))
        tqdm.write(f"ok   {slug} [{seq.modality}] {num_written} layers in {dur:.1f}s -> {_seq_dir(out_dir, seq)}")
    return oks, fails


def main(cfg: IngestConfig) -> None:
    sequences = _load_sequences(cfg)
    sequences = filter_sequences(
        sequences,
        only=cfg.only or None,
        datasets=cfg.datasets or None,
        modalities=cfg.modalities or None,
    )
    if cfg.limit is not None:
        sequences = sequences[: cfg.limit]

    print(f"Ingesting {len(sequences)} sequences -> {cfg.out}  (workers={cfg.workers})")
    if cfg.layers:
        print(f"  --layers filter: {' '.join(cfg.layers)}")
    cfg.out.mkdir(parents=True, exist_ok=True)

    # Build todo list. Each entry is (sequence, layers_to_emit). We resolve
    # the per-sequence layer set here so workers don't need to redo the
    # filtering logic, and so we can skip sequences whose layers already
    # exist on disk (unless --force).
    todo: list[tuple[Sequence, set[str] | None]] = []
    skipped = 0
    no_op = 0
    for seq in sequences:
        seq_dir = _seq_dir(cfg.out, seq)
        seq_dir.mkdir(parents=True, exist_ok=True)

        target_layers = _select_layers_for(seq, requested=cfg.layers)
        if target_layers is not None and not target_layers:
            # User asked for layers that don't apply to this sequence (e.g.
            # --layers depth_0 on a mono sequence). Skip silently.
            no_op += 1
            continue

        if not cfg.force:
            existing = _existing_layers(seq_dir)
            if target_layers is None:
                # Default mode: skip if every applicable layer file already exists.
                if set(applicable_layers(seq)).issubset(existing):
                    skipped += 1
                    continue
            else:
                # Subset mode: skip if every requested layer file exists.
                if target_layers.issubset(existing):
                    skipped += 1
                    continue

        todo.append((seq, target_layers))

    succeeded = skipped
    failed: list[tuple[str, str]] = []
    nvenc_retries: list[tuple[Sequence, set[str] | None]] = []

    if cfg.workers <= 1 or len(todo) <= 1:
        oks, fails = _run_serial(todo, out_dir=cfg.out, desc="sequences")
        succeeded += len(oks)
        failed.extend(fails)
    else:
        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            futures = {pool.submit(_ingest_one, seq, cfg.out, layers): (seq, layers) for seq, layers in todo}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="sequences"):
                seq, layers = futures[fut]
                try:
                    slug, dur, num_written = fut.result()
                except BeartypeException:
                    raise
                except Exception as exc:  # noqa: BLE001
                    msg = f"{type(exc).__name__}: {exc}"
                    if _is_nvenc_session_flake(msg):
                        nvenc_retries.append((seq, layers))
                        tqdm.write(f"NVENC flake (will retry serially): {seq.slug}")
                        continue
                    failed.append((seq.slug, msg))
                    tqdm.write(f"FAIL {seq.slug}: {exc}")
                    continue
                tqdm.write(f"ok   {slug} [{seq.modality}] {num_written} layers in {dur:.1f}s -> {_seq_dir(cfg.out, seq)}")
                succeeded += 1

    if nvenc_retries:
        print(f"\nRetrying {len(nvenc_retries)} NVENC-flaked sequence(s) serially…")
        oks, fails = _run_serial(nvenc_retries, out_dir=cfg.out, desc="retries")
        succeeded += len(oks)
        failed.extend(fails)

    summary = f"\nDone: {succeeded}/{len(sequences)} succeeded ({skipped} skipped"
    if no_op:
        summary += f", {no_op} no-op"
    summary += f"), {len(failed)} failed."
    print(summary)
    if failed:
        print("Failures:")
        for slug, err in failed:
            print(f"  {slug}: {err}")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_cyan")
    main(tyro.cli(IngestConfig, description="Ingest VSLAM-LAB sequences into per-stream Rerun layer files."))
