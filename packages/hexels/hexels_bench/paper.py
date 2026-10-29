"""Published numbers: `paper-numbers.json` beside this module (machine-written from the paper, arXiv 2610.37668v1,
and the HF jhu-ccvl/Hexels catalog; its `source` section names the revisions). Read partially: the sections below."""

from dataclasses import dataclass
from pathlib import Path

from serde import serde

PAPER_NUMBERS: Path = Path(__file__).with_name("paper-numbers.json")


@serde
@dataclass(frozen=True, slots=True)
class PaperSource:
    """Where the numbers come from (`source`, read partially)."""

    hardware_paper: str
    """GPU the paper reports its numbers on, as the paper states it."""


@serde
@dataclass(frozen=True, slots=True)
class PublishedNvs:
    """One published 30k NVS checkpoint (`hf_nvs_per_scene`, the authors' numbers)."""

    views: int
    """Held-out views."""
    psnr: float
    """PSNR, dB (upstream direct-float evaluator)."""
    ssim: float
    """SSIM."""
    lpips_vgg: float
    """LPIPS-VGG."""
    primitives: int
    """Hexels in the checkpoint."""
    learned_model_bytes: int
    """FP32 learned parameter bytes."""
    train_seconds_summary: float | None
    """Training seconds from the checkpoint's summary.json; null for grown models."""


@serde
@dataclass(frozen=True, slots=True)
class Table1Row:
    """One method of the paper's Table 1 (2D, all 45 Image-GS images)."""

    method: str
    """Image-GS, Hexels (constant) or Hexels."""
    psnr: float
    """PSNR, dB."""
    ssim: float
    """SSIM."""
    lpips_alex: float
    """LPIPS-Alex."""


@serde
@dataclass(frozen=True, slots=True)
class PublishedImage:
    """One published 2D fit of one image (`hf_images_2d_per_image`)."""

    psnr_hxm: float
    """PSNR of the packed .hxm file, dB."""
    ssim_hxm: float
    """SSIM of the packed file."""
    lpips_alex_hxm: float
    """LPIPS-Alex of the packed file."""
    psnr_uint16: float
    """PSNR of the uniform 16-bit file, dB."""
    median_fps_uint16_a5000: float
    """Decode FPS of the 16-bit file on the authors' RTX A5000."""


@serde
@dataclass(frozen=True, slots=True)
class PublishedSurface:
    """The paper's surface checkpoint of one scene, scored by its own evaluation (`hf_surface_main_per_scene`)."""

    cd_mm: float | None = None
    """DTU Chamfer distance, mm."""
    f1_pct: float | None = None
    """Tanks and Temples F1, percent."""


@serde
@dataclass(frozen=True, slots=True)
class PaperNumbers:
    """The sections of paper-numbers.json the tables read; keys are upstream dataset names."""

    source: PaperSource
    """Paper revision, HF revision and the paper's hardware."""
    hf_nvs_per_scene: dict[str, dict[str, dict[str, PublishedNvs]]]
    """Published NVS checkpoints per dataset, scene and `<variant>-<population>` (e.g. v1-fixed)."""
    sfm_points: dict[str, dict[str, int]]
    """SfM points (the primitive cap) per dataset and scene."""
    table1_2d: list[Table1Row]
    """Paper Table 1."""
    hf_images_2d_per_image: dict[str, dict[str, PublishedImage]]
    """Published 2D fits per image stem and appearance (spatial-rgb, flat-rgb)."""
    hf_surface_main_per_scene: dict[str, dict[str, PublishedSurface]]
    """Published surface checkpoints per dataset and scene."""
