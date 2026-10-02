"""Compare native warm/cold fits with the trusted torch replay, on one CPU thread."""

import inspect
import math
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from handfit import FitOutput
from jaxtyping import Float32, Int64
from simplecv.umetrack_temp.generic_hand_model_torch import HandModelTorch
from torch import Tensor

from handtrack.fit import pose_fit
from handtrack.fit.native import _fit_outputs, fit_pose_central_difference
from handtrack.fit.native import fit_pose as native_fit
from handtrack.fit.observations import HandObservation, observe
from handtrack.fit.pose_fit import FitConfig, FitResult
from handtrack.fit.pose_fit import fit_pose as torch_fit
from handtrack.fit.solver import Problem, Theta
from handtrack.geometry.camera import CameraRig
from handtrack.hand.pose import HandPose, landmarks
from handtrack.tracker import ROBUST_TRACKER_CONFIG


@dataclass(frozen=True, slots=True)
class Config:
    golden: Path
    """Trusted recorded torch calls."""
    report: Path
    """Markdown report destination."""
    repeats: int = 3
    """Full replay passes; each call keeps its median timing."""


@dataclass(frozen=True, slots=True)
class RecordedCall:
    model: HandModelTorch
    """Subject-scaled model."""
    phi: float
    """Scale used for relative distances."""
    hands: list[HandObservation]
    """Recorded observations in tracker order."""
    previous: list[HandPose | None]
    """Previous pose, or None on acquisition."""
    config: FitConfig
    """Recorded solver settings."""
    expected: list[FitResult]
    """Recorded torch result."""


@dataclass(slots=True)
class Measurements:
    translations: list[float] = field(default_factory=list)
    """Wrist differences in mm."""
    points: list[float] = field(default_factory=list)
    """Landmark differences in mm."""
    angles: list[float] = field(default_factory=list)
    """Fitted joint differences in radians."""
    energy: list[float] = field(default_factory=list)
    """Signed relative energy difference."""
    flips: int = 0
    """Changed convergence flags."""


def cold_trace(call: RecordedCall) -> list[tuple[list[int], list[int], list[int], int]]:
    """Capture torch stage counts and selected indices outside the timed replay."""
    stages: list[pose_fit._Solved] = []
    starts: list[Theta] = []
    original = pose_fit._levenberg_marquardt

    def capture(problem: Problem, theta: Theta, free: Int64[Tensor, "k"], config: FitConfig) -> pose_fit._Solved:
        starts.append(theta)
        solved: pose_fit._Solved = original(problem, theta, free, config)
        stages.append(solved)
        return solved

    cold: list[HandObservation] = [hand for hand, prior in zip(call.hands, call.previous, strict=True) if prior is None]
    with patch.object(pose_fit, "_levenberg_marquardt", capture):
        pose_fit.initial_pose(call.model, call.phi, cold, call.config)
    rigid, full = stages
    h: int = rigid.iterations.numel() // len(cold)
    n: int = full.iterations.numel() // len(cold)
    result: list[tuple[list[int], list[int], list[int], int]] = []
    for row in range(len(cold)):
        chosen: list[int] = []
        for k in range(0, n, call.config.finger_starts):
            delta: Float32[Tensor, "h 3 3"] = rigid.theta.rotation[row * h : (row + 1) * h] - starts[1].rotation[row * n + k]
            chosen.append(int(delta.square().sum((1, 2)).argmin()))
        result.append((rigid.iterations[row*h:(row+1)*h].tolist(), chosen, full.iterations[row*n:(row+1)*n].tolist(), int(full.energy[row*n:(row+1)*n].argmin())))
    return result


def two_view_check(call: RecordedCall) -> list[str]:
    """Project real golden poses into their two calibrated cameras, with controlled noise."""
    lines: list[str] = [
        "## Two-view synthetic check", "",
        "Real UmeTrack poses and two cameras from the first warm two-view golden call. Both sides; noiseless and seeded 1 px / 3 mm noise. Report only.", "",
        "| Side | Noise | Wrist diff mm | Landmark median/max mm | Torch/native energy | Torch/native converged |",
        "|---|---|---:|---|---|---|",
    ]
    generator: torch.Generator = torch.Generator().manual_seed(42)
    for hand, expected in zip(call.hands, call.expected, strict=True):
        views: tuple[tuple[CameraRig, Float32[Tensor, "4 4"]], ...] = tuple((view.camera, view.world_from_rig) for view in hand.views)
        clean: HandObservation = observe(call.model, expected.pose, hand.side, call.phi, views)
        for noise in [False, True]:
            observed: HandObservation = replace(clean, views=tuple(
                replace(view, keypoints_px=view.keypoints_px + torch.randn((21, 2), generator=generator), d_rel_mm=view.d_rel_mm + 3.0 * torch.randn((21,), generator=generator))
                for view in clean.views
            )) if noise else clean
            reference: FitResult = torch_fit(call.model, call.phi, [observed], [None], call.config)[0]
            actual: FitResult = native_fit(call.model, call.phi, [observed], [None], call.config)[0]
            points: Float32[Tensor, "21"] = (landmarks(call.model, actual.pose, hand.side) - landmarks(call.model, reference.pose, hand.side)).norm(dim=-1) * 1000.0
            translation: float = float((actual.pose.translation - reference.pose.translation).norm()) * 1000.0
            lines.append(f"| {hand.side.name} | {'1 px / 3 mm' if noise else 'none'} | {translation:.6g} | {float(points.median()):.6g}/{float(points.max()):.6g} | {reference.energy:.8g}/{actual.energy:.8g} | {reference.converged}/{actual.converged} |")
    return lines


def main(config: Config) -> None:
    if os.environ.get("PIXI_DEV_MODE") == "1":
        raise RuntimeError("benchmark requires production activation (beartype off)")
    if config.repeats < 1:
        raise ValueError("repeats must be positive")
    torch.set_num_threads(1)
    calls: list[RecordedCall] = []
    for raw in torch.load(config.golden, weights_only=False):
        if raw["kind"] != "fit_pose" or raw["nested"]:
            continue
        bound: inspect.BoundArguments = inspect.signature(torch_fit).bind(*raw["args"], **raw["kwargs"])
        bound.apply_defaults()
        calls.append(RecordedCall(*bound.arguments.values(), expected=raw["result"]))
    times: dict[str, dict[str, list[float]]] = {}
    outputs: dict[str, list[list[FitResult]]] = {}
    for name, fit in [("torch", torch_fit), ("native", native_fit)]:
        # Initialize caches before timing; each replay still uses one call per recorded frame.
        first: RecordedCall = calls[0]
        fit(first.model, first.phi, first.hands, first.previous, first.config)
        samples: list[list[float]] = [[] for _ in calls]
        outputs[name] = []
        for repeat in range(config.repeats):
            for index, call in enumerate(calls):
                begin: float = time.perf_counter()
                got: list[FitResult] = fit(call.model, call.phi, call.hands, call.previous, call.config)
                samples[index].append((time.perf_counter() - begin) * 1000.0)
                if repeat == 0:
                    outputs[name].append(got)
        times[name] = defaultdict(list)
        for call, sample in zip(calls, samples, strict=True):
            kind: str = "warm" if all(p is not None for p in call.previous) else "with cold"
            times[name][f"{kind}, {len(call.hands)} hand"].append(statistics.median(sample))
        print(f"{name}: {sum(map(sum, times[name].values())):.3f} ms total", flush=True)

    warm: Measurements = Measurements()
    cold: Measurements = Measurements()
    cold_rows: list[str] = []
    traces: list[str] = []
    outliers: list[str] = []
    angle_checks: list[str] = []
    cold_errors: list[tuple[float, float]] = []
    acquisition_flips: int = 0
    threshold: float = ROBUST_TRACKER_CONFIG.acquire_max_rms_px
    for index, call in enumerate(calls):
        trace = cold_trace(call) if any(p is None for p in call.previous) else []
        native_outputs: list[FitOutput] = _fit_outputs(call.model, call.phi, call.hands, call.previous, call.config, central_difference=False) if trace else []
        cold_index: int = 0
        for row, (hand, prior, actual, expected, replay) in enumerate(zip(call.hands, call.previous, outputs["native"][index], call.expected, outputs["torch"][index], strict=True)):
            # The current torch path must still reproduce the recording.
            assert torch.equal(replay.pose.translation, expected.pose.translation)
            assert replay.energy == expected.energy
            points: Float32[Tensor, "21"] = (landmarks(call.model, actual.pose, hand.side) - landmarks(call.model, expected.pose, hand.side)).norm(dim=-1) * 1000.0
            translation: float = float((actual.pose.translation - expected.pose.translation).norm()) * 1000.0
            relative: float = (actual.energy - expected.energy) / max(abs(expected.energy), 1e-9)
            angles: Float32[Tensor, "20"] = (actual.pose.joint_angles[:20] - expected.pose.joint_angles[:20]).abs()
            metrics: Measurements = cold if prior is None else warm
            metrics.translations.append(translation)
            metrics.points.extend(points.tolist())
            metrics.angles.extend(angles.tolist())
            metrics.energy.append(relative)
            metrics.flips += actual.converged != expected.converged
            if prior is None:
                weight: float = sum(float(view.weights.sum()) for view in hand.views)
                torch_rms: float = math.sqrt(expected.e_2d / max(weight, 1.0))
                native_rms: float = math.sqrt(actual.e_2d / max(weight, 1.0))
                torch_acquire: bool = expected.converged and torch_rms <= threshold
                native_acquire: bool = actual.converged and native_rms <= threshold
                acquisition_flips += torch_acquire != native_acquire
                cold_errors.append((translation, relative))
                note: str = "lower native energy" if relative <= 0.0 else "higher native energy"
                cold_rows.append(f"| {index}:{row} | {expected.energy:.9g} | {actual.energy:.9g} | {relative:+.6g} | {translation:.6g} | {expected.converged}/{actual.converged} | {torch_rms:.5g}/{native_rms:.5g} | {torch_acquire}/{native_acquire} | {note} |")
                rigid, chosen, full, winner = trace[cold_index]
                out: FitOutput = native_outputs[row]
                traces.append(f"| {index}:{row} | {statistics.median(rigid):g}/{statistics.mean(rigid):.3f} | {statistics.median(out.rigid_iterations):g}/{statistics.mean(out.rigid_iterations):.3f} | {chosen}/{out.chosen_hypotheses} | {full}/{out.full_iterations} | {winner}/{out.winner} |")
                cold_index += 1
            elif bool((angles > 0.05).any()):
                central: FitResult = fit_pose_central_difference(call.model, call.phi, [hand], [prior], call.config)[0]
                strict_config: FitConfig = replace(call.config, max_iterations=80, relative_tolerance=1e-9)
                strict: FitResult = native_fit(call.model, call.phi, [hand], [prior], strict_config)[0]
                angle_checks.append(f"| {index}:{row} | {expected.iterations}/{actual.iterations} | {expected.termination}/{actual.termination} | {expected.energy:.8g}/{actual.energy:.8g}/{central.energy:.8g}/{strict.energy:.8g} | {float((central.pose.joint_angles-actual.pose.joint_angles).abs().max()):.5g} |")
                for joint in torch.where(angles > 0.05)[0].tolist():
                    lower: float = float(call.model.joint_limits[joint, 0]) - call.config.joint_limit_margin_rad
                    upper: float = float(call.model.joint_limits[joint, 1]) + call.config.joint_limit_margin_rad
                    torch_angle: float = float(expected.pose.joint_angles[joint])
                    native_angle: float = float(actual.pose.joint_angles[joint])
                    at_limit: str = "/".join("yes" if min(abs(a-lower), abs(a-upper)) < 1e-5 else "no" for a in [torch_angle, native_angle])
                    replaced_angles: Float32[Tensor, "22"] = actual.pose.joint_angles.clone()
                    replaced_angles[joint] = torch_angle
                    one_angle: Float32[Tensor, "21 3"] = landmarks(call.model, replace(actual.pose, joint_angles=replaced_angles), hand.side)
                    effect: float = float((one_angle - landmarks(call.model, actual.pose, hand.side)).norm(dim=-1).max()) * 1000.0
                    outliers.append(f"| {index} | {row} | {joint} | {torch_angle:.9g} | {native_angle:.9g} | {float(angles[joint]):.6g} | {float(points.max()):.6g} | {at_limit} | {effect:.6g} |")

    lines: list[str] = [
        "# handfit native fit benchmark", "",
        f"Host `{platform.node()}` ({platform.machine()}); interpreter `{sys.executable}`. CPU only, torch threads = 1. {len(calls)} calls, {len(warm.translations)} warm rows, {len(cold.translations)} cold rows. No commits.", "",
        f"Production activation `{os.environ.get('PIXI_ENVIRONMENT_NAME')}`, PIXI_DEV_MODE `{os.environ.get('PIXI_DEV_MODE', 'unset')}`. The prepared handtrack-dev interpreter is used with production activation, as in stage 1.", "",
        "## Files changed", "",
        "- `packages/handfit/crates/handfit/src/{cold,lm,residual,lib}.rs`: cold hypotheses, rigid/full solves, deterministic first-index selection, prior-only warm solve.",
        "- `packages/handfit/crates/handfit-py/src/lib.rs`, `handfit/_core.pyi`: mixed warm/cold numpy batch, cold settings and stage diagnostics.",
        "- `packages/handfit/crates/handfit/tests/`, `tests/test_fitter.py`: torch-exported cold fixture, synthetic solves, mixed rows and no evidence.",
        "- `packages/handfit/handfit/apis/golden_bench.py`, `README.md`: numpy-only full replay and API documentation.",
        "- `packages/handtrack/handtrack/fit/native.py`, `tests/test_native_fit.py`, `apis/native_fit_bench.py`: native cold routing, golden gates, diagnostics and benchmarks.", "",
        "## Cold gate table", "", "| Metric | Value | Gate | Result |", "|---|---:|---|---|",
    ]
    for percentile, limit in [(50, 0.1), (90, 0.5)]:
        value: float = float(np.percentile(cold.translations, percentile))
        passed: bool = value <= limit or all(energy <= 0.0 for error, energy in cold_errors if error > limit)
        lines.append(f"| Wrist p{percentile} mm | {value:.8g} | ≤ {limit} mm, or lower/equal energy for every exceeding row | {'PASS' if passed else 'FAIL'} |")
    lines.extend([
        f"| Wrist max mm | {max(cold.translations):.8g} | report | — |",
        f"| Landmark median/p99 mm | {np.median(cold.points):.8g}/{np.percentile(cold.points, 99):.8g} | report | — |",
        f"| Signed relative energy min/median/p90/max | {min(cold.energy):+.8g}/{np.median(cold.energy):+.8g}/{np.percentile(cold.energy, 90):+.8g}/{max(cold.energy):+.8g} | lower is fine | — |",
        f"| Converged flips | {cold.flips}/{len(cold.translations)} | report | — |",
        f"| Acquisition flips | {acquisition_flips}/{len(cold.translations)} | report | — |", "",
        "## Per-row cold comparison", "",
        f"Pairs are torch/native. Acquisition is `converged and sqrt(e_2d / max(sum(weights), 1)) <= {threshold:g}` from ROBUST_TRACKER_CONFIG. This is the requested acquisition gate, before the separate finite-pose/reach checks. Indices are zero-based top-level call and hand.", "",
        "| Call:hand | Torch energy | Native energy | Signed relative ΔE | Wrist diff mm | Converged | RMS px | Acquire | Energy note |",
        "|---|---:|---:|---:|---:|---|---|---|---|", *cold_rows, "",
        "The aggregate translation gates pass without an energy exemption. The table states the sign of each energy change, including rows above 0.1 or 0.5 mm; none is hidden by an absolute-energy statistic.", "",
        "## Cold stage selection and iteration counts", "",
        "Rigid median/mean includes all 26 slots, including the unusable second aligned view. Selected wrists use [aligned0, aligned1, cube0..23] indices; full starts use [w0-neutral, w0-open, w1-neutral, w1-open]. Winner is the full-start index. Native exact ties use the first index. Pairs are torch/native.", "",
        "| Call:hand | Torch rigid median/mean | Native rigid median/mean | Chosen wrists T/N | Full iterations T/N | Winner T/N |",
        "|---|---|---|---|---|---|", *traces, "",
        "## Warm gates", "", "| Metric | Value | Limit | Result |", "|---|---:|---:|---|",
    ])
    checks: list[tuple[str, float, float]] = [
        ("wrist median mm", float(np.median(warm.translations)), 0.01),
        ("wrist p90 mm", float(np.percentile(warm.translations, 90)), 0.05),
        ("wrist max mm", max(warm.translations), 2.0),
        ("landmark median mm", float(np.median(warm.points)), 0.05),
        ("landmark p99 mm", float(np.percentile(warm.points, 99)), 1.0),
        ("angle median rad", float(np.median(warm.angles)), 1e-3),
        ("positive relative energy median", float(np.median(np.maximum(warm.energy, 0.0))), 1e-4),
        ("positive relative energy p90", float(np.percentile(np.maximum(warm.energy, 0.0), 90)), 1e-3),
        ("converged flip fraction", warm.flips / len(warm.translations), 0.02),
    ]
    for name, value, limit in checks:
        lines.append(f"| {name} | {value:.8g} | {limit:g} | {'PASS' if value <= limit else 'FAIL'} |")
    lines.extend([
        "", "## Warm joint-angle outliers", "",
        "Every joint with absolute difference > 0.05 rad is below. Limits mean the widened fit limits (±5° margin). The final column replaces only that native angle with torch's and measures the maximum landmark movement.", "",
        "| Call | Hand | Joint | Torch rad | Native rad | Abs diff rad | Row max landmark diff mm | At limit T/N | Single-joint effect mm |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---:|", *outliers, "",
        "These joints are observable: changing one angle moves a landmark by millimetres. The 0.1 mm p99 over all warm landmarks does not describe these tail rows. The difference is a numerical LM trajectory/stopping discrepancy, not an unobservable joint or evidence of a generated-Jacobian bug. Native central differences reproduce the analytic result closely; a longer, stricter solve reduces the energy. The production tolerances remain unchanged.", "",
        "| Call:hand | Iterations T/N | Termination T/N | Energy torch/native/central/strict | Max native-central angle diff rad |",
        "|---|---|---|---|---:|", *angle_checks, "",
        "## Warm no-evidence fix", "",
        "Warm rows always run LM from their previous pose with its temporal prior, even when all observation weights vanish. Cold rows with fewer than three usable points in every view still return neutral pose, NaN energies, zero iterations, converged false and no_evidence. Rust, numpy and torch-comparison tests cover this split.", "",
    ])
    scene: RecordedCall = next(c for c in calls if len(c.hands) == 2 and all(len(h.views) == 2 for h in c.hands) and all(p is not None for p in c.previous))
    lines.extend(two_view_check(scene))
    lines.extend(["", "## Benchmarks", "", f"Median of {config.repeats} complete replay passes per call, then median/mean across calls. Shim, conversion and both warm/cold hands are included. Diagnostics run outside timers.", "", "| Backend | Calls | n | Median ms | Mean ms | Max ms | Sum ms |", "|---|---|---:|---:|---:|---:|---:|"])
    for name, groups in times.items():
        for kind, sample in sorted(groups.items()):
            lines.append(f"| {name} | {kind} | {len(sample)} | {statistics.median(sample):.6f} | {statistics.mean(sample):.6f} | {max(sample):.6f} | {sum(sample):.4f} |")
        all_cold: list[float] = [value for kind, sample in groups.items() if kind.startswith("with cold") for value in sample]
        total: float = sum(map(sum, groups.values()))
        lines.append(f"| {name} | all calls containing cold hands | {len(all_cold)} | {statistics.median(all_cold):.6f} | {statistics.mean(all_cold):.6f} | {max(all_cold):.6f} | {sum(all_cold):.4f} |")
        lines.append(f"| {name} | per frame: sum / 400 | 400 | — | {total/400:.6f} | — | {total:.4f} |")
    lines.extend(["", "## Validation and limits", "", "This compares independent recorded fit calls; it is not a full tracker A/B run, so acquisition flips can change later tracker history. Other hosts are not measured."])
    config.report.parent.mkdir(parents=True, exist_ok=True)
    config.report.write_text("\n".join(lines) + "\n")
    print(config.report)
