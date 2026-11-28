"""Fixed model and preprocessing settings used by the released method.

Only reader-useful choices live in the CLI. These values describe the trained
model and therefore stay together here instead of being exposed as knobs.
"""

from __future__ import annotations

import os

from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceConfig:
    num_frames: int = 121
    height: int = 1280
    width: int = 704
    prompt: str = "视频中的人在做动作"
    auto_downsample_fps: tuple[tuple[int, int], ...] = (
        (24, 1),
        (24000, 1001),
        (25, 1),
        (30, 1),
        (30000, 1001),
    )
    temporal_sampling_policy: str = "nearest_source_pts_on_zero_based_cfr_clock"
    skeleton_h264_crf: int = 17
    target_h264_crf: int = 18
    h264_preset: str = "medium"
    skeleton_h264_preset: str = "veryfast"
    dit_fp8_w8a8: bool = os.environ.get("FDANYONE_DIT_FP8", "1") != "0"
    """Row-wise FP8 W8A8 (``torch._scaled_mm``) on the interior DiT projections (blocks 3-26), as the 5090 port's
    turbo. Activations are quantized in row chunks of at most 1 GiB of temporaries, so a six-view group fits beside
    the resident DiT (the FP8 weights free 4.5 GiB). On by default; ``FDANYONE_DIT_FP8=0`` turns it off. RTX 5090, six-view example (busy host, load ~10):
    denoise 112 -> 86 s, peak 29.2 -> 24.5 GB reserved, mean per-view SSIM vs the BF16 build 0.917."""
    dit_fused_glue: bool = os.environ.get("FDANYONE_DIT_FUSED", "1") != "0"
    """Fused Triton kernels (``fdanyone.model.fused``) for the DiT's memory-bound glue: RMSNorm + RoPE, LayerNorm +
    modulation (written frame-major for the multiview attention), gated residuals, and one FP8 row quantization per
    projection input (GELU folded into the FFN's). Same BF16 rounding points as the eager code; the FP8 path is
    bit-exact. RTX 5090, six-view group: one forward 20 -> 11.6 s. ``FDANYONE_DIT_FUSED=0`` runs the eager code."""
    dit_fp8_edges: bool = os.environ.get("FDANYONE_DIT_FP8_EDGES", "0") == "1"
    """Also run the boundary blocks 0-2 and 27-29 (BF16 otherwise) with FP8 W8A8 projections. Off by default;
    ``FDANYONE_DIT_FP8_EDGES=1`` turns it on. RTX 5090 step-0 capture: about 0.6 s per forward faster, predicted velocity
    0.068 rel L2 from the shipped FP8 policy and 0.079 from BF16 (the shipped policy: 0.084 from BF16)."""
    dit_nvfp4_projections: tuple[str, ...] = tuple(filter(None, os.environ.get("FDANYONE_DIT_NVFP4", "").split(",")))
    """Projection groups of the FP8 blocks that run NVFP4 W4A4 instead (cuBLASLt block-scaled FP4 GEMM, Triton
    activation quantization): any of ``ffn``, ``self_attn``, ``self_attn_mvs``, ``cross_attn``. Off by default;
    ``FDANYONE_DIT_NVFP4=ffn`` turns it on for the FFNs."""
    dit_nvfp4_blocks: tuple[int, int] = tuple(int(b) for b in os.environ.get("FDANYONE_DIT_NVFP4_BLOCKS", "3-26").split("-"))  # type: ignore[assignment]
    """Inclusive block range of ``dit_nvfp4_projections`` (inside the FP8 range, blocks 3-26)."""
    dit_nvfp4_steps: tuple[int, ...] = tuple(int(k) for k in os.environ.get("FDANYONE_DIT_NVFP4_STEPS", ""))
    """Denoising step indices (digits, e.g. ``3`` for the last of four) at which ``dit_nvfp4_projections`` run NVFP4;
    the other steps run their FP8 twins. Empty (default) = every step. Late steps amplify the NVFP4 error least."""
    dit_regional_compile: bool = False
    """``torch.compile`` each DiT block in place (static shapes). Off by default: on an RTX 5090 a cold
    max-autotune compile took ~690 s, and with upstream's six-view groups the compiled blocks ran out of the 32 GB
    (the 5090 port compiled four-view groups with warm caches)."""
    dit_compile_mode: str = "max-autotune-no-cudagraphs"
    """Inductor mode of the regional compile."""
    target_decoder: str = "taew2_2"
    """Decoder of the generated target views: ``taew2_2`` (TAEW2.2 tiny decoder) or ``wan`` (the full Wan 2.2 VAE)."""
    skeleton_max_dimension: int = 2048
    tiled_vae: bool = False
    vae_tile_size: tuple[int, int] = (52, 30)
    vae_tile_stride: tuple[int, int] = (26, 15)


@dataclass(frozen=True)
class DenoisingProfile:
    """Scheduler and target-context routing used by one inference mode."""

    name: str
    num_inference_steps: int
    denoising_strength: float
    scheduler_shift: float
    tcr_stride: int
    freeze_tcr_after_one_cycle: bool
    kept_steps: tuple[int, ...] | None = None
    """Scheduler steps to run (indices into the num_inference_steps trajectory); None runs all."""

    def __post_init__(self) -> None:
        if self.num_inference_steps <= 0:
            raise ValueError("Denoising steps must be positive.")
        if not 0.0 < self.denoising_strength <= 1.0:
            raise ValueError("Denoising strength must be in (0, 1].")
        if self.scheduler_shift <= 0.0:
            raise ValueError("Scheduler shift must be positive.")
        if self.tcr_stride <= 0:
            raise ValueError("TCR stride must be positive.")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "num_inference_steps": self.num_inference_steps,
            "denoising_strength": self.denoising_strength,
            "scheduler_shift": self.scheduler_shift,
            "tcr_stride": self.tcr_stride,
            "freeze_tcr_after_one_cycle": self.freeze_tcr_after_one_cycle,
            "kept_steps": None if self.kept_steps is None else list(self.kept_steps),
        }


@dataclass(frozen=True)
class CameraConfig:
    count: int = 24
    pitch_degrees: float = 15.0

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("Camera count must be positive.")


@dataclass(frozen=True)
class ForegroundConfig:
    """Pinned standard BiRefNet inference contract."""

    image_size: tuple[int, int] = (1024, 1024)
    batch_size: int = 4


@dataclass(frozen=True)
class FramingConfig:
    """Clip-wide camera solve matching the current GVHMR demo."""

    reference_radius: float = 3.0
    reference_target_height: float = 1.0
    reference_focal_normalized: float = 1664.0 / 1280.0
    height_target_ratio: float = 0.80
    height_percentile: float = 95.0
    width_target_ratio: float = 0.90
    width_percentile: float = 80.0
    min_radius: float = 1.5
    max_radius: float = 8.0
    input_min_confidence: float = 0.55
    max_focal_normalized: float = 4.0
    cutoff_target_ratio: float = 0.99
    cutoff_percentile: float = 80.0


@dataclass(frozen=True)
class CropConfig:
    """Source-mask crop; generated cameras use a plain center aspect crop."""

    margin_top: float = 0.04
    margin_right: float = 0.04
    margin_bottom: float = 0.04
    margin_left: float = 0.04
    allow_upscale: bool = True
    mask_threshold: float = 0.05

    @property
    def margins(self) -> tuple[float, float, float, float]:
        return (self.margin_top, self.margin_right, self.margin_bottom, self.margin_left)


@dataclass(frozen=True)
class SkeletonConfig:
    draw_body_reference_px: float = 640.0


INFERENCE = InferenceConfig()
BASE24 = DenoisingProfile(
    name="base24",
    num_inference_steps=24,
    denoising_strength=1.0,
    scheduler_shift=5.0,
    tcr_stride=1,
    freeze_tcr_after_one_cycle=False,
)
def _turbo_kept_steps() -> tuple[int, ...] | None:
    """``FDANYONE_TURBO_KEPT_STEPS=0,2,3`` runs a subset of the four trained Turbo timesteps (fewer DiT passes)."""

    value = os.environ.get("FDANYONE_TURBO_KEPT_STEPS", "").strip()
    return tuple(int(item) for item in value.split(",")) if value else None


RANK64_DELTA4 = DenoisingProfile(
    name="rank64_delta4",
    num_inference_steps=4,
    denoising_strength=1.0,
    scheduler_shift=5.0,
    tcr_stride=2,
    freeze_tcr_after_one_cycle=True,
    kept_steps=_turbo_kept_steps(),
)
CAMERA = CameraConfig()
FOREGROUND = ForegroundConfig()
FRAMING = FramingConfig()
CROP = CropConfig()
SKELETON = SkeletonConfig()
