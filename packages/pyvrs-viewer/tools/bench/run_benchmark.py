"""Benchmark pyvrs-viewer across multiple Hot3D VRS files.

Downloads 5 Quest + 5 Aria VRS files and runs both AV1 encode and
EncodedImage (JPEG passthrough) modes, collecting timing and size metrics.

Usage:
    pixi run -e pyvrs-viewer -- python tools/bench/run_benchmark.py
"""

import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from pyvrs import SyncVRSReader
from simplecv.rerun_log_utils import RerunTyroConfig

from pyvrs_viewer.video_encoder import VideoCodecChoice
from pyvrs_viewer.vrs_to_rerun import VrsToRerunConfig, vrs_to_rerun

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

REPO_ROOT: Path = Path(__file__).resolve().parents[4]
PKG_ROOT: Path = Path(__file__).resolve().parents[2]
QUEST_JSON: Path = REPO_ROOT / ".vibe-attachments" / "e0f8ef29-0798-4e19-8ac2-d58289db420f_hot3dquest_download_urls.json"
ARIA_JSON: Path = REPO_ROOT / ".vibe-attachments" / "c641c45d-c468-4e35-912d-2cb92fdc893d_hot3daria_download_urls.json"
DATA_DIR: Path = PKG_ROOT / "data" / "benchmark"
MAX_FILES: int = 5


@dataclass
class BenchmarkResult:
    """Results for one VRS file."""

    sequence_id: str
    device: str
    vrs_size_mb: float
    n_records: int
    av1_time_sec: float = 0.0
    av1_rrd_size_mb: float = 0.0
    jpeg_time_sec: float = 0.0
    jpeg_rrd_size_mb: float = 0.0
    error: str = ""


def _download_vrs(url: str, dest: Path) -> None:
    """Download a VRS file with curl if not already on disk."""
    if dest.exists():
        logger.info("  Already exists: %s", dest.name)
        return
    logger.info("  Downloading %s (%.0f MB)...", dest.name, 0)
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-L", "-o", str(dest), url], check=True)


def _run_one(vrs_path: Path, encode_video: bool) -> tuple[float, float]:
    """Run vrs_to_rerun and return (time_sec, rrd_size_mb)."""
    with tempfile.NamedTemporaryFile(suffix=".rrd", delete=False) as f:
        rrd_path: Path = Path(f.name)

    try:
        config: VrsToRerunConfig = VrsToRerunConfig(
            vrs_path=vrs_path,
            rr_config=RerunTyroConfig(save=rrd_path, headless=True),
            encode_video=encode_video,
            video_codec=VideoCodecChoice.AV1,
        )
        t0: float = time.perf_counter()
        vrs_to_rerun(config)
        elapsed: float = time.perf_counter() - t0
        rrd_size_mb: float = rrd_path.stat().st_size / 1024 / 1024
        return elapsed, rrd_size_mb
    finally:
        rrd_path.unlink(missing_ok=True)


def _collect_vrs_files(json_path: Path, device: str) -> list[tuple[str, str, Path]]:
    """Return list of (sequence_id, download_url, local_path) for first MAX_FILES sequences."""
    with open(json_path) as f:
        data: dict = json.load(f)

    results: list[tuple[str, str, Path]] = []
    for seq_id in list(data["sequences"].keys())[:MAX_FILES]:
        vrs_info: dict = data["sequences"][seq_id].get("main_vrs", {})
        url: str = vrs_info.get("download_url", "")
        local_path: Path = DATA_DIR / device / f"{seq_id}.vrs"
        results.append((seq_id, url, local_path))
    return results


def _copy_existing_files() -> None:
    """Copy existing VRS files from data/ to benchmark directories."""
    existing_quest: Path = PKG_ROOT / "data" / "hot3d_quest.vrs"
    existing_aria: Path = PKG_ROOT / "data" / "hot3d_aria.vrs"
    dest_quest: Path = DATA_DIR / "quest" / "P0002_1464cbdc.vrs"
    dest_aria: Path = DATA_DIR / "aria" / "P0001_10a27bf7.vrs"

    if existing_quest.exists() and not dest_quest.exists():
        logger.info("Linking existing Quest VRS → %s", dest_quest.name)
        os.link(str(existing_quest), str(dest_quest))
    if existing_aria.exists() and not dest_aria.exists():
        logger.info("Linking existing Aria VRS → %s", dest_aria.name)
        os.link(str(existing_aria), str(dest_aria))


def main() -> None:
    logger.info("=== pyvrs-viewer Benchmark ===")

    # Organize existing files
    _copy_existing_files()

    # Collect file lists
    quest_files: list[tuple[str, str, Path]] = _collect_vrs_files(QUEST_JSON, "quest")
    aria_files: list[tuple[str, str, Path]] = _collect_vrs_files(ARIA_JSON, "aria")
    all_files: list[tuple[str, str, Path, str]] = [(s, u, p, "quest") for s, u, p in quest_files] + [(s, u, p, "aria") for s, u, p in aria_files]

    # Download missing files
    logger.info("Downloading %d VRS files...", len(all_files))
    for seq_id, url, local_path, device in all_files:
        _download_vrs(url, local_path)

    # Run benchmarks
    results: list[BenchmarkResult] = []
    for seq_id, _url, local_path, device in all_files:
        vrs_size_mb: float = local_path.stat().st_size / 1024 / 1024
        reader: SyncVRSReader = SyncVRSReader(str(local_path))
        n_records: int = reader.n_records

        logger.info("")
        logger.info("─── %s (%s, %.0f MB, %d records) ───", seq_id, device, vrs_size_mb, n_records)

        result: BenchmarkResult = BenchmarkResult(
            sequence_id=seq_id, device=device, vrs_size_mb=vrs_size_mb, n_records=n_records
        )

        try:
            # AV1 encode mode
            logger.info("  AV1 encode...")
            result.av1_time_sec, result.av1_rrd_size_mb = _run_one(local_path, encode_video=True)
            logger.info("  → %.1fs, %.0f MB RRD (%.1fx compression)", result.av1_time_sec, result.av1_rrd_size_mb, vrs_size_mb / result.av1_rrd_size_mb if result.av1_rrd_size_mb > 0 else 0)
        except Exception as e:
            result.error += f"AV1: {e}; "
            logger.exception("  AV1 FAILED: %s", e)

        try:
            # EncodedImage (JPEG passthrough) mode
            logger.info("  JPEG passthrough...")
            result.jpeg_time_sec, result.jpeg_rrd_size_mb = _run_one(local_path, encode_video=False)
            logger.info("  → %.1fs, %.0f MB RRD", result.jpeg_time_sec, result.jpeg_rrd_size_mb)
        except Exception as e:
            result.error += f"JPEG: {e}; "
            logger.exception("  JPEG FAILED: %s", e)

        results.append(result)

    # Print summary table
    print("\n" + "=" * 120)
    print("BENCHMARK RESULTS")
    print("=" * 120)
    header: str = f"{'Sequence':<20} {'Device':<6} {'VRS(MB)':>8} {'Records':>8} │ {'AV1 Time':>9} {'AV1 RRD':>9} {'Ratio':>6} │ {'JPEG Time':>10} {'JPEG RRD':>10} {'Ratio':>6} │ {'Errors'}"
    print(header)
    print("─" * 120)
    for r in results:
        av1_ratio: str = f"{r.vrs_size_mb / r.av1_rrd_size_mb:.0f}x" if r.av1_rrd_size_mb > 0 else "N/A"
        jpeg_ratio: str = f"{r.vrs_size_mb / r.jpeg_rrd_size_mb:.1f}x" if r.jpeg_rrd_size_mb > 0 else "N/A"
        print(
            f"{r.sequence_id:<20} {r.device:<6} {r.vrs_size_mb:>8.0f} {r.n_records:>8} │ "
            f"{r.av1_time_sec:>8.1f}s {r.av1_rrd_size_mb:>8.0f}M {av1_ratio:>6} │ "
            f"{r.jpeg_time_sec:>9.1f}s {r.jpeg_rrd_size_mb:>9.0f}M {jpeg_ratio:>6} │ "
            f"{r.error or 'OK'}"
        )
    print("─" * 120)

    # Save markdown results
    results_md: Path = DATA_DIR / "results.md"
    with open(results_md, "w") as f:
        f.write("# pyvrs-viewer Benchmark Results\n\n")
        f.write("| Sequence | Device | VRS (MB) | Records | AV1 Time (s) | AV1 RRD (MB) | AV1 Ratio | JPEG Time (s) | JPEG RRD (MB) | JPEG Ratio | Status |\n")
        f.write("|----------|--------|----------|---------|--------------|--------------|-----------|---------------|---------------|------------|--------|\n")
        for r in results:
            av1_ratio_str: str = f"{r.vrs_size_mb / r.av1_rrd_size_mb:.0f}x" if r.av1_rrd_size_mb > 0 else "N/A"
            jpeg_ratio_str: str = f"{r.vrs_size_mb / r.jpeg_rrd_size_mb:.1f}x" if r.jpeg_rrd_size_mb > 0 else "N/A"
            f.write(
                f"| {r.sequence_id} | {r.device} | {r.vrs_size_mb:.0f} | {r.n_records} | "
                f"{r.av1_time_sec:.1f} | {r.av1_rrd_size_mb:.0f} | {av1_ratio_str} | "
                f"{r.jpeg_time_sec:.1f} | {r.jpeg_rrd_size_mb:.0f} | {jpeg_ratio_str} | "
                f"{r.error or 'OK'} |\n"
            )
    logger.info("Results saved to %s", results_md)


if __name__ == "__main__":
    main()
