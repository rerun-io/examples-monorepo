"""The shared NVS scorer (contract run-layout.md): PSNR, SSIM and LPIPS-VGG on 16-bit PNG renders.

Same math as upstream's direct-float evaluator (`hexels/benchmark_metrics.py` L.343-416): clamp once
to [0, 1]; PSNR from the float64 SSE over all RGB elements of a view; SSIM = 2DGS `utils/loss_utils.ssim`
(11x11 Gaussian window, sigma 1.5, C1 = 0.01^2, C2 = 0.03^2, zero padding, mean over the map);
LPIPS = `lpips.LPIPS(net="vgg")` on inputs mapped to [-1, 1]. The reported PSNR is the mean of the
per-view PSNRs (upstream `mean_psnr_db`).
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Float32, Float64
from numpy import ndarray
from serde import serde
from serde.json import from_json, to_json

from hexels_bench.png16 import read_png16, to_unit_float

Scorer: TypeAlias = Literal["shared-png16", "upstream-float"]
SSIM_WINDOW: int = 11
SSIM_SIGMA: float = 1.5


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ViewMetrics:
    """Scores of one held-out view."""

    name: str
    """Image stem."""
    psnr: float
    """PSNR in dB."""
    ssim: float
    """SSIM (2DGS window)."""
    lpips_vgg: float | None
    """LPIPS-VGG, null when not scored."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class MetricsFile:
    """`metrics.json` of one run directory (contract run-layout.md)."""

    method: str
    """Run method, e.g. hexels-v3-30k-pub-cuda."""
    dataset: str
    """mipnerf360, dtu, tnt or imagegs."""
    scene: str
    """Scene name."""
    views: int
    """Number of scored views."""
    psnr: float
    """Mean per-view PSNR in dB."""
    ssim: float
    """Mean SSIM."""
    lpips_vgg: float | None
    """Mean LPIPS-VGG, null when not scored."""
    fps: float | None
    """Render frames per second (paper protocol), null when not measured."""
    deploy_mb: float | None
    """Primitive count x FP32 bytes per primitive / 1e6."""
    ply_mb: float | None
    """Size of model.ply in MB (1e6 bytes)."""
    scorer: Scorer
    """Which scorer produced the numbers."""
    per_view: list[ViewMetrics]
    """Per-view scores, in render order."""


def psnr_from_sse(sse: float, elements: int) -> float:
    if not math.isfinite(sse) or sse <= 0.0:
        raise ValueError(f"SSE must be positive and finite, got {sse}")
    return -10.0 * math.log10(sse / elements)


def view_sse(prediction: Float32[ndarray, "h w 3"], target: Float32[ndarray, "h w 3"]) -> float:
    difference: Float64[ndarray, "h w 3"] = np.clip(prediction, 0.0, 1.0).astype(np.float64) - np.clip(target, 0.0, 1.0).astype(np.float64)
    return float(np.sum(np.square(difference), dtype=np.float64))


def _gaussian_window(device: torch.device) -> Float[torch.Tensor, "3 1 11 11"]:
    offsets: Float64[ndarray, "11"] = np.arange(SSIM_WINDOW, dtype=np.float64) - SSIM_WINDOW // 2
    gauss: Float32[ndarray, "11"] = np.array([math.exp(-(x**2) / (2 * SSIM_SIGMA**2)) for x in offsets], dtype=np.float32)
    gauss = gauss / gauss.sum()
    window: torch.Tensor = torch.from_numpy(np.outer(gauss, gauss).astype(np.float32))
    return window.expand(3, 1, SSIM_WINDOW, SSIM_WINDOW).contiguous().to(device=device)


def ssim(prediction: Float[torch.Tensor, "3 h w"], target: Float[torch.Tensor, "3 h w"]) -> float:
    """2DGS `ssim(img1, img2)` with size_average=True."""
    window: Float[torch.Tensor, "3 1 11 11"] = _gaussian_window(prediction.device)
    padding: int = SSIM_WINDOW // 2
    mu1: torch.Tensor = F.conv2d(prediction, window, padding=padding, groups=3)
    mu2: torch.Tensor = F.conv2d(target, window, padding=padding, groups=3)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq: torch.Tensor = F.conv2d(prediction * prediction, window, padding=padding, groups=3) - mu1_sq
    sigma2_sq: torch.Tensor = F.conv2d(target * target, window, padding=padding, groups=3) - mu2_sq
    sigma12: torch.Tensor = F.conv2d(prediction * target, window, padding=padding, groups=3) - mu1_mu2
    c1: float = 0.01**2
    c2: float = 0.03**2
    ssim_map: torch.Tensor = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_map.mean().item())


def score_views(
    render_dir: Path,
    gt_dir: Path,
    names: list[str],
    device: torch.device,
    with_lpips: bool = True,
) -> list[ViewMetrics]:
    """Score `render_dir/<name>.png` against `gt_dir/<name>.png` for every name."""
    lpips_fn = None
    if with_lpips:
        import lpips

        lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(device=device).eval()
    rows: list[ViewMetrics] = []
    with torch.no_grad():
        for name in names:
            prediction: Float32[ndarray, "h w 3"] = to_unit_float(read_png16(render_dir / f"{name}.png"))
            target: Float32[ndarray, "h w 3"] = to_unit_float(read_png16(gt_dir / f"{name}.png"))
            if prediction.shape != target.shape:
                raise ValueError(f"{name}: render {prediction.shape} vs target {target.shape}")
            pred_chw: torch.Tensor = torch.from_numpy(np.ascontiguousarray(prediction.transpose(2, 0, 1))).to(device=device)
            target_chw: torch.Tensor = torch.from_numpy(np.ascontiguousarray(target.transpose(2, 0, 1))).to(device=device)
            lpips_value: float | None = None
            if lpips_fn is not None:
                lpips_value = float(lpips_fn(pred_chw[None] * 2.0 - 1.0, target_chw[None] * 2.0 - 1.0).item())
            rows.append(
                ViewMetrics(
                    name=name,
                    psnr=psnr_from_sse(view_sse(prediction, target), prediction.size),
                    ssim=ssim(pred_chw, target_chw),
                    lpips_vgg=lpips_value,
                )
            )
    return rows


def summarize(
    rows: list[ViewMetrics],
    *,
    method: str,
    dataset: str,
    scene: str,
    scorer: Scorer,
    fps: float | None,
    deploy_mb: float | None,
    ply_mb: float | None,
) -> MetricsFile:
    if not rows:
        raise ValueError("no views to summarize")
    lpips_values: list[float] = [row.lpips_vgg for row in rows if row.lpips_vgg is not None]
    return MetricsFile(
        method=method,
        dataset=dataset,
        scene=scene,
        views=len(rows),
        psnr=float(np.mean([row.psnr for row in rows])),
        ssim=float(np.mean([row.ssim for row in rows])),
        lpips_vgg=float(np.mean(lpips_values)) if len(lpips_values) == len(rows) else None,
        fps=fps,
        deploy_mb=deploy_mb,
        ply_mb=ply_mb,
        scorer=scorer,
        per_view=rows,
    )


def write_metrics_file(path: Path, metrics: MetricsFile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(metrics) + "\n")


def load_metrics_file(path: Path) -> MetricsFile:
    return from_json(MetricsFile, path.read_text())


def ply_vertex_count(path: Path) -> int:
    """Vertex count from any PLY header (3DGS and hexel PLY alike)."""
    with path.open("rb") as file:
        for raw in file:
            line: str = raw.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex "):
                return int(line.split()[2])
            if line == "end_header":
                break
    raise ValueError(f"{path}: no vertex element")


BYTES_PER_PRIMITIVE: dict[str, int] = {"hexels-v1": 80, "hexels-v2": 116, "hexels-v3": 260, "brush": 236}
"""FP32 learned bytes per primitive (run-layout.md): Hexels V1/V2/V3 and 3DGS with SH3 (59 floats)."""


def bytes_per_primitive(method: str) -> int | None:
    for prefix, size in BYTES_PER_PRIMITIVE.items():
        if method.startswith(prefix):
            return size
    return None
