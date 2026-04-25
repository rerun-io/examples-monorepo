#!/usr/bin/env python3
"""Ingest sequences from a manifest into per-sequence RRD files.

NVENC has a small per-process session limit, so when ``--workers > 1``
the GPU encoder occasionally rejects ``avcodec_open2(hevc_nvenc)`` with
``ExternalError [Errno 542398533]``. We catch those specific failures
and retry them serially (workers=1) at the end of the run, so the corpus
finishes 109/109 without manual cleanup.
"""

from __future__ import annotations

import json
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
from slam_evals.ingest import ingest_sequence

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
    manifest: Path | None = None
    """Manifest JSON produced by ``discover.py``. If unset, ``benchmark_root`` is scanned on the fly."""

    benchmark_root: Path = field(default_factory=lambda: Path("/home/pablo/0Dev/work/VSLAM-LAB-Benchmark"))
    """Used only when ``manifest`` is unset."""

    out: Path = field(default_factory=lambda: Path("data/slam-evals/rrd"))
    """Output directory. Each sequence writes to ``<out>/<dataset>/<name>.rrd``."""

    only: tuple[str, ...] = ()
    """Restrict to specific slugs or sequence names (e.g. ``EUROC/MH_01_easy`` or ``MH_01_easy``)."""

    datasets: tuple[str, ...] = ()
    """Restrict to specific datasets (e.g. ``EUROC KITTI``)."""

    modalities: tuple[Modality, ...] = ()
    """Restrict to specific modalities."""

    force: bool = False
    """Re-ingest sequences whose RRD already exists."""

    limit: int | None = None
    """Cap the number of sequences processed (post-filter)."""

    workers: int = 4
    """Process pool size. NVENC is GPU-bound so over-subscribing past ~4 doesn't help; ``1`` runs serially."""


def _load_sequences(cfg: IngestConfig) -> list[Sequence]:
    if cfg.manifest is not None:
        raw = json.loads(cfg.manifest.read_text())
        return [
            Sequence(
                dataset=s["dataset"],
                name=s["name"],
                root=Path(s["root"]),
                modality=Modality(s["modality"]),
                has_calibration=bool(s.get("has_calibration", False)),
            )
            for s in raw["sequences"]
        ]
    return discover_sequences(cfg.benchmark_root.expanduser().resolve())


def _ingest_one(seq: Sequence, out_rrd: Path) -> tuple[str, float]:
    """Worker entrypoint: ingest a single sequence; return (slug, wall_seconds)."""
    t0 = time.monotonic()
    ingest_sequence(seq, out_rrd)
    return seq.slug, time.monotonic() - t0


def _is_nvenc_session_flake(err_message: str) -> bool:
    return any(marker in err_message for marker in _NVENC_RETRY_MARKERS)


def _run_serial(
    todo: list[tuple[Sequence, Path]],
    *,
    desc: str,
) -> tuple[list[tuple[str, str, float]], list[tuple[str, str]]]:
    """Run ``todo`` sequentially. Returns (oks, fails). Each ok = (slug, modality, dur)."""
    oks: list[tuple[str, str, float]] = []
    fails: list[tuple[str, str]] = []
    for seq, out_rrd in tqdm(todo, desc=desc):
        try:
            slug, dur = _ingest_one(seq, out_rrd)
        except BeartypeException:
            raise
        except Exception as exc:  # noqa: BLE001
            fails.append((seq.slug, f"{type(exc).__name__}: {exc}"))
            tqdm.write(f"FAIL {seq.slug}: {exc}")
            continue
        oks.append((slug, str(seq.modality), dur))
        tqdm.write(f"ok   {slug} [{seq.modality}] in {dur:.1f}s -> {out_rrd}")
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
    cfg.out.mkdir(parents=True, exist_ok=True)

    # Skip sequences whose RRD already exists (unless --force).
    todo: list[tuple[Sequence, Path]] = []
    skipped = 0
    for seq in sequences:
        out_rrd = cfg.out / seq.dataset / f"{seq.name}.rrd"
        if out_rrd.exists() and not cfg.force:
            skipped += 1
            continue
        out_rrd.parent.mkdir(parents=True, exist_ok=True)
        todo.append((seq, out_rrd))

    succeeded = skipped
    failed: list[tuple[str, str]] = []
    nvenc_retries: list[tuple[Sequence, Path]] = []

    if cfg.workers <= 1 or len(todo) <= 1:
        oks, fails = _run_serial(todo, desc="sequences")
        succeeded += len(oks)
        failed.extend(fails)
    else:
        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            futures = {pool.submit(_ingest_one, seq, out_rrd): (seq, out_rrd) for seq, out_rrd in todo}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="sequences"):
                seq, out_rrd = futures[fut]
                try:
                    slug, dur = fut.result()
                except BeartypeException:
                    raise
                except Exception as exc:  # noqa: BLE001
                    msg = f"{type(exc).__name__}: {exc}"
                    if _is_nvenc_session_flake(msg):
                        # Don't count as failure yet — queue for serial retry.
                        nvenc_retries.append((seq, out_rrd))
                        tqdm.write(f"NVENC flake (will retry serially): {seq.slug}")
                        continue
                    failed.append((seq.slug, msg))
                    tqdm.write(f"FAIL {seq.slug}: {exc}")
                    continue
                tqdm.write(f"ok   {slug} [{seq.modality}] in {dur:.1f}s -> {out_rrd}")
                succeeded += 1

    if nvenc_retries:
        print(f"\nRetrying {len(nvenc_retries)} NVENC-flaked sequence(s) serially…")
        oks, fails = _run_serial(nvenc_retries, desc="retries")
        succeeded += len(oks)
        failed.extend(fails)

    print(f"\nDone: {succeeded}/{len(sequences)} succeeded ({skipped} skipped), {len(failed)} failed.")
    if failed:
        print("Failures:")
        for slug, err in failed:
            print(f"  {slug}: {err}")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_cyan")
    main(tyro.cli(IngestConfig, description="Ingest VSLAM-LAB sequences into Rerun RRDs."))
