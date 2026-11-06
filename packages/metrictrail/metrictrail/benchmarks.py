"""Upstream MetricTrail benchmark settings as typed presets.

Source: ViPE @ ead2e00, ``configs/slam/README_metrictrail.md`` (the settings table and the commands below it).
Each preset turns into the Hydra overrides of one upstream ``python run.py pipeline=metrictrail ...`` command.
KITTI stereo is not here: it runs through the separate stereo runner, not ``pipeline=metrictrail``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

BenchmarkName: TypeAlias = Literal["tum", "wild_slam", "7scenes", "kitti", "kitti_calibrated", "oxford", "vbr", "lamaria"]
SlamPreset: TypeAlias = Literal["metrictrail_local", "metrictrail_gba", "metrictrail_gba_refine"]
RasterPolicy: TypeAlias = Literal["unchanged", "cover_crop", "resize"]

UPSTREAM_SEED: int = 1234
"""Upstream's ``pipeline.slam.seed`` default; DPVO seeds each sequence with it plus the directory name."""


@dataclass(frozen=True, slots=True)
class BenchmarkPreset:
    """One row of upstream's benchmark table."""

    name: BenchmarkName
    """Benchmark key, as upstream names its output directory ``vipe_results/<name>``."""
    slam_preset: SlamPreset
    """DPVO backend preset, ``slam@pipeline.slam=<preset>``."""
    raster_policy: RasterPolicy
    """``pipeline.init.input_raster.policy``: ``unchanged`` keeps prepared frames as they are."""
    raster_hw: tuple[int, int] | None = None
    """Read-time target ``(height, width)`` for ``cover_crop`` / ``resize``; None for ``unchanged``."""
    tracking_stride: int = 1
    """``pipeline.init.tracking_stride``: DPVO tracks every k-th frame (7-Scenes: 5); SharedK calibrates on all."""
    max_frames: int | None = None
    """``pipeline.init.input_raster.max_frames`` override (LaMaria raises it); None keeps upstream's 1500."""
    focal_refine: bool = False
    """``pipeline.focal_refine.enabled`` (RoMa stage B; VBR only)."""
    known_intrinsics: bool = False
    """Known K instead of SharedK (``kitti_calibrated``): the command passes ``pipeline.optional.intrinsics``."""
    seed_null: bool = False
    """Upstream sets ``pipeline.slam.seed=null`` (LaMaria)."""

    def __post_init__(self) -> None:
        if (self.raster_policy == "unchanged") != (self.raster_hw is None):
            raise ValueError(f"{self.name}: raster_hw must be set exactly when the raster policy resizes ({self.raster_policy})")
        if self.tracking_stride < 1:
            raise ValueError(f"{self.name}: tracking_stride must be >= 1, got {self.tracking_stride}")

    def run_overrides(self, frames_dir: Path, output_dir: Path, intrinsics_path: Path | None = None, seed: int | None = None) -> list[str]:
        """Hydra overrides of upstream's ``run.py`` command for one sequence.

        Args:
            frames_dir: Directory of one sequence's RGB frames; its name names the outputs and seeds DPVO.
            output_dir: ``pipeline.output.path``.
            intrinsics_path: Known-K file, required exactly for ``known_intrinsics`` presets.
            seed: Explicit ``pipeline.slam.seed`` (repeated runs); None keeps the preset's seed.

        Returns:
            The arguments after ``python run.py``, in upstream's order.
        """
        if self.known_intrinsics != (intrinsics_path is not None):
            raise ValueError(f"{self.name}: intrinsics_path is required exactly when the preset uses known intrinsics")
        overrides: list[str] = [
            "pipeline=metrictrail",
            f"slam@pipeline.slam={self.slam_preset}",
            "streams=frame_dir_stream",
            f"streams.base_path={frames_dir}",
            f"pipeline.init.input_raster.policy={self.raster_policy}",
        ]
        if self.raster_hw is not None:
            overrides += [f"pipeline.init.input_raster.height={self.raster_hw[0]}", f"pipeline.init.input_raster.width={self.raster_hw[1]}"]
        if self.max_frames is not None:
            overrides.append(f"pipeline.init.input_raster.max_frames={self.max_frames}")
        if self.tracking_stride != 1:
            overrides.append(f"pipeline.init.tracking_stride={self.tracking_stride}")
        if intrinsics_path is not None:
            overrides.append(f"pipeline.optional.intrinsics={{path: {intrinsics_path}}}")
        else:
            overrides.append("pipeline.init.calibration.allow_unstable=true")
        if self.focal_refine:
            overrides.append("pipeline.focal_refine.enabled=true")
        if seed is not None:
            overrides.append(f"pipeline.slam.seed={seed}")
        elif self.seed_null:
            overrides.append("pipeline.slam.seed=null")
        overrides.append(f"pipeline.output.path={output_dir}")
        return overrides


BENCHMARKS: dict[BenchmarkName, BenchmarkPreset] = {
    "tum": BenchmarkPreset("tum", "metrictrail_gba_refine", "unchanged"),
    "wild_slam": BenchmarkPreset("wild_slam", "metrictrail_gba", "cover_crop", raster_hw=(288, 512)),
    "7scenes": BenchmarkPreset("7scenes", "metrictrail_gba", "resize", raster_hw=(434, 574), tracking_stride=5),
    "kitti": BenchmarkPreset("kitti", "metrictrail_local", "unchanged"),
    "kitti_calibrated": BenchmarkPreset("kitti_calibrated", "metrictrail_local", "unchanged", known_intrinsics=True),
    "oxford": BenchmarkPreset("oxford", "metrictrail_local", "unchanged"),
    "vbr": BenchmarkPreset("vbr", "metrictrail_local", "unchanged", focal_refine=True),
    "lamaria": BenchmarkPreset("lamaria", "metrictrail_local", "cover_crop", raster_hw=(392, 504), max_frames=100000, seed_null=True),
}
"""Upstream's benchmark table, keyed by benchmark name."""
