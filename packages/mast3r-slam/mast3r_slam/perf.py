from __future__ import annotations

import json
import math
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch


def _maybe_sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def timed_section(
    metrics: dict[str, float | int | str],
    key: str,
    *,
    sync_cuda: bool = False,
) -> Any:
    if sync_cuda:
        _maybe_sync_cuda()
    start = time.perf_counter()
    try:
        yield
    finally:
        if sync_cuda:
            _maybe_sync_cuda()
        metrics[key] = (time.perf_counter() - start) * 1000.0


class BenchmarkRecorder:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict[str, Any]] = []

    def append(self, row: dict[str, Any]) -> None:
        self._rows.append(row)

    def flush(self) -> None:
        with self.output_path.open("w", encoding="utf-8") as f:
            for row in self._rows:
                f.write(json.dumps(row, sort_keys=True))
                f.write("\n")


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _max(values: list[float]) -> float:
    return max(values) if values else 0.0


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if math.isclose(denom, 0.0):
        return 0.0
    numer = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=False))
    return numer / denom


def summarize_benchmark(
    frontend_rows: list[dict[str, Any]],
    backend_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "frontend_frames": len(frontend_rows),
        "backend_tasks": len(backend_rows),
        "frontend": {},
        "backend": {},
        "diagnosis": [],
    }

    if frontend_rows:
        total_ms = [float(row.get("frame_total_ms", 0.0)) for row in frontend_rows]
        log_ms = [float(row.get("logging_total_ms", 0.0)) for row in frontend_rows]
        track_ms = [float(row.get("tracking_total_ms", 0.0)) for row in frontend_rows]
        frame_idx = [float(row.get("frame_idx", i)) for i, row in enumerate(frontend_rows)]
        n_kf = [float(row.get("n_keyframes", 0)) for row in frontend_rows]

        summary["frontend"] = {
            "avg_frame_ms": _mean(total_ms),
            "avg_fps": 1000.0 / _mean(total_ms) if _mean(total_ms) > 0 else 0.0,
            "max_frame_ms": _max(total_ms),
            "avg_logging_ms": _mean(log_ms),
            "avg_tracking_ms": _mean(track_ms),
            "frame_ms_slope_per_frame": _linear_slope(frame_idx, total_ms),
            "logging_ms_slope_per_frame": _linear_slope(frame_idx, log_ms),
            "logging_ms_slope_per_keyframe": _linear_slope(n_kf, log_ms),
            "tracking_ms_slope_per_frame": _linear_slope(frame_idx, track_ms),
        }

        if summary["frontend"]["logging_ms_slope_per_keyframe"] > 0.05:
            summary["diagnosis"].append(
                "Frontend logging cost increases with keyframe count; visualization overhead is a likely source of FPS decay."
            )
        if summary["frontend"]["tracking_ms_slope_per_frame"] > 0.05:
            summary["diagnosis"].append(
                "Tracking cost itself grows over time; inspect matcher warm starts, optimizer iteration counts, or cached state growth."
            )

    if backend_rows:
        task_ms = [float(row.get("task_total_ms", 0.0)) for row in backend_rows]
        add_factor_ms = [float(row.get("add_factors_total_ms", 0.0)) for row in backend_rows]
        solve_ms = [float(row.get("global_opt_total_ms", 0.0)) for row in backend_rows]
        idxs = [float(row.get("task_keyframe_idx", i)) for i, row in enumerate(backend_rows)]

        summary["backend"] = {
            "avg_task_ms": _mean(task_ms),
            "max_task_ms": _max(task_ms),
            "avg_add_factors_ms": _mean(add_factor_ms),
            "avg_global_opt_ms": _mean(solve_ms),
            "task_ms_slope_per_keyframe": _linear_slope(idxs, task_ms),
        }

        if summary["backend"]["avg_add_factors_ms"] > summary["backend"]["avg_global_opt_ms"]:
            summary["diagnosis"].append(
                "Backend pair construction dominates backend time; decoder plus dense matching remains the main backend bottleneck."
            )
        if summary["backend"]["task_ms_slope_per_keyframe"] > 1.0:
            summary["diagnosis"].append(
                "Backend work grows noticeably with map size; retrieval and global optimization are not staying constant-time."
            )

    return summary


def write_summary_markdown(summary: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = ["# MASt3R-SLAM Benchmark Summary", ""]

    frontend = summary.get("frontend", {})
    if frontend:
        lines.extend(
            [
                "## Frontend",
                "",
                f"- Frames: {summary['frontend_frames']}",
                f"- Average frame time: {frontend['avg_frame_ms']:.2f} ms",
                f"- Average FPS: {frontend['avg_fps']:.2f}",
                f"- Average tracking time: {frontend['avg_tracking_ms']:.2f} ms",
                f"- Average logging time: {frontend['avg_logging_ms']:.2f} ms",
                f"- Frame-time slope: {frontend['frame_ms_slope_per_frame']:.4f} ms/frame",
                f"- Logging slope vs keyframes: {frontend['logging_ms_slope_per_keyframe']:.4f} ms/keyframe",
                "",
            ]
        )

    backend = summary.get("backend", {})
    if backend:
        lines.extend(
            [
                "## Backend",
                "",
                f"- Tasks: {summary['backend_tasks']}",
                f"- Average task time: {backend['avg_task_ms']:.2f} ms",
                f"- Average add-factors time: {backend['avg_add_factors_ms']:.2f} ms",
                f"- Average global-opt time: {backend['avg_global_opt_ms']:.2f} ms",
                f"- Task-time slope: {backend['task_ms_slope_per_keyframe']:.4f} ms/keyframe",
                "",
            ]
        )

    diagnosis = summary.get("diagnosis", [])
    if diagnosis:
        lines.append("## Diagnosis")
        lines.append("")
        for item in diagnosis:
            lines.append(f"- {item}")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
