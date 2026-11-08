"""MASt3R-SLAM hyper-parameters: one frozen dataclass per group, and the named presets the CLI and the app offer.

``SlamConfig()`` is upstream MASt3R-SLAM's ``config/base.yaml``; every other preset changes a few fields of a parent preset, as
the YAML files did with ``inherit``.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeAlias

import tyro

FilteringModeName: TypeAlias = Literal["recent", "first", "best_score", "weighted_pointmap", "weighted_spherical", "indep_conf"]
FilteringScoreName: TypeAlias = Literal["median", "mean"]


@dataclass(frozen=True)
class DatasetConfig:
    """How frames are read from the input."""

    subsample: int = 1
    """Keep every n-th frame."""
    img_downsample: int = 1
    """Stride over the decoder's pointmaps and the frame image (1 = full resolution)."""
    center_principle_point: bool = True
    """Center the principal point of the undistorted camera matrix (calibrated datasets only)."""


@dataclass(frozen=True)
class MatchingConfig:
    """Dense matching by iterative projection, then descriptor refinement."""

    max_iter: int = 10
    """Iterations of the projection solve."""
    lambda_init: float = 1e-8
    """Initial Levenberg-Marquardt damping of the projection solve."""
    convergence_thresh: float = 1e-6
    """Step size below which the projection solve stops."""
    dist_thresh: float = 1e-1
    """Largest 3D distance between matched points; a farther match counts as occluded."""
    radius: int = 3
    """Descriptor refinement window radius in pixels (0 = no refinement)."""
    dilation_max: int = 5
    """First dilation of the refinement window; it decreases to 1."""


@dataclass(frozen=True)
class TrackingConfig:
    """Frame-to-keyframe tracking (Gauss-Newton on ray + distance residuals, or pixel + depth when calibrated)."""

    min_match_frac: float = 0.05
    """Fraction of valid matches below which tracking is lost and relocalization starts."""
    max_iters: int = 50
    """Gauss-Newton iterations per frame."""
    C_conf: float = 0.0
    """Smallest pointmap confidence a match needs."""
    Q_conf: float = 1.5
    """Smallest descriptor confidence a match needs."""
    rel_error: float = 1e-3
    """Relative cost change below which Gauss-Newton stops."""
    delta_norm: float = 1e-3
    """Step norm below which Gauss-Newton stops."""
    huber: float = 1.345
    """Huber threshold on the whitened residuals."""
    match_frac_thresh: float = 0.333
    """Fraction of matches to the keyframe below which the frame becomes a new keyframe."""
    sigma_ray: float = 0.003
    """Standard deviation of the ray direction residual."""
    sigma_dist: float = 1e1
    """Standard deviation of the ray distance residual."""
    sigma_pixel: float = 1.0
    """Standard deviation of the pixel residual (calibrated)."""
    sigma_depth: float = 1e1
    """Standard deviation of the log-depth residual (calibrated)."""
    pixel_border: int = -10
    """Pixels outside the image a projection may land and stay valid, when negative (calibrated)."""
    depth_eps: float = 1e-6
    """Smallest depth a point needs to count (calibrated)."""
    filtering_mode: FilteringModeName = "weighted_pointmap"
    """How a keyframe's canonical pointmap fuses new observations (see ``FilteringMode``)."""
    filtering_score: FilteringScoreName = "median"
    """Score of one observation; only ``filtering_mode=best_score`` uses it."""


@dataclass(frozen=True)
class LocalOptConfig:
    """Backend global optimisation over the keyframe factor graph."""

    pin: int = 1
    """Keyframes held fixed at the start of the graph (gauge freedom)."""
    C_conf: float = 0.0
    """Smallest pointmap confidence a match needs."""
    Q_conf: float = 1.5
    """Smallest descriptor confidence a match needs."""
    min_match_frac: float = 0.1
    """Fraction of valid matches an edge needs to enter the graph."""
    pixel_border: int = -10
    """Pixels outside the image a projection may land and stay valid, when negative (calibrated)."""
    depth_eps: float = 1e-6
    """Smallest depth a point needs to count (calibrated)."""
    max_iters: int = 10
    """Gauss-Newton iterations per solve."""
    sigma_ray: float = 0.003
    """Standard deviation of the ray direction residual."""
    sigma_dist: float = 1e1
    """Standard deviation of the ray distance residual."""
    sigma_pixel: float = 1.0
    """Standard deviation of the pixel residual (calibrated)."""
    sigma_depth: float = 1e1
    """Standard deviation of the log-depth residual (calibrated)."""
    delta_norm: float = 1e-8
    """Step norm below which Gauss-Newton stops."""


@dataclass(frozen=True)
class RetrievalConfig:
    """Image retrieval of loop-closure candidates."""

    k: int = 3
    """Candidates returned per query."""
    min_thresh: float = 5e-3
    """Smallest retrieval score a candidate needs."""


@dataclass(frozen=True)
class RelocConfig:
    """Relocalization after tracking is lost."""

    min_match_frac: float = 0.3
    """Fraction of valid matches a relocalization edge needs."""
    strict: bool = True
    """Every retrieved candidate must match (otherwise one is enough)."""


@dataclass(frozen=True)
class SlamConfig:
    """MASt3R-SLAM hyper-parameters. ``SlamConfig()`` is upstream MASt3R-SLAM's ``config/base.yaml``."""

    use_calib: bool = False
    """Use the dataset's camera calibration (pixel + depth residuals); the input must carry one."""
    single_thread: bool = False
    """Run frontend and backend in lockstep: the frontend waits for each keyframe and relocalization."""
    max_backend_backlog: int = 2
    """Keyframes the frontend may queue ahead of the backend before it waits. A new keyframe starts from the tracked
    pose of the one before it; if the backend lags far behind, global optimisation starts from a stale chain and can
    settle in a wrong minimum."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    local_opt: LocalOptConfig = field(default_factory=LocalOptConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reloc: RelocConfig = field(default_factory=RelocConfig)

    def __post_init__(self) -> None:
        if self.dataset.subsample < 1:
            raise ValueError(f"dataset.subsample must be >= 1, got {self.dataset.subsample}")
        if self.dataset.img_downsample < 1:
            raise ValueError(f"dataset.img_downsample must be >= 1, got {self.dataset.img_downsample}")
        if self.max_backend_backlog < 0:
            raise ValueError(f"max_backend_backlog must be >= 0, got {self.max_backend_backlog}")

    def with_subsample(self, subsample: int) -> "SlamConfig":
        """This config, keeping every ``subsample``-th frame."""
        return dataclasses.replace(self, dataset=dataclasses.replace(self.dataset, subsample=subsample))

    @classmethod
    def base(cls) -> "SlamConfig":
        """The paper's defaults (upstream ``config/base.yaml``)."""
        return cls()

    @classmethod
    def fast(cls) -> "SlamConfig":
        """``base`` on every 5th frame."""
        return cls.base().with_subsample(5)

    @classmethod
    def test(cls) -> "SlamConfig":
        """``fast``, single-threaded: deterministic, for regression tests."""
        return dataclasses.replace(cls.fast(), single_thread=True)

    @classmethod
    def calib(cls) -> "SlamConfig":
        """``base`` with the dataset's calibration, on every 2nd frame."""
        return dataclasses.replace(cls.base(), use_calib=True).with_subsample(2)

    @classmethod
    def eval_calib(cls) -> "SlamConfig":
        """``calib``, single-threaded: the paper's calibrated benchmarks."""
        return dataclasses.replace(cls.calib(), single_thread=True)

    @classmethod
    def eval_no_calib(cls) -> "SlamConfig":
        """``base`` single-threaded on every 2nd frame: the paper's uncalibrated benchmarks."""
        return dataclasses.replace(cls.base(), single_thread=True).with_subsample(2)

    @classmethod
    def eth3d(cls) -> "SlamConfig":
        """``eval_calib`` on every frame, principal point as calibrated, any retrieved candidate may relocalize."""
        parent: SlamConfig = cls.eval_calib()
        return dataclasses.replace(
            parent,
            dataset=dataclasses.replace(parent.dataset, subsample=1, center_principle_point=False),
            reloc=dataclasses.replace(parent.reloc, strict=False),
        )


SlamPresetName: TypeAlias = Literal["base", "fast", "test", "calib", "eval_calib", "eval_no_calib", "eth3d"]
SLAM_PRESETS: dict[SlamPresetName, tuple[str, SlamConfig]] = {
    "base": ("Paper defaults, every frame.", SlamConfig.base()),
    "fast": ("Paper defaults on every 5th frame.", SlamConfig.fast()),
    "test": ("fast, single-threaded and deterministic (regression tests).", SlamConfig.test()),
    "calib": ("Uses the dataset's calibration, every 2nd frame.", SlamConfig.calib()),
    "eval_calib": ("calib, single-threaded (paper benchmarks).", SlamConfig.eval_calib()),
    "eval_no_calib": ("Uncalibrated, single-threaded, every 2nd frame (paper benchmarks).", SlamConfig.eval_no_calib()),
    "eth3d": ("eval_calib for ETH3D: every frame, uncentered principal point, lenient relocalization.", SlamConfig.eth3d()),
}
"""Preset name -> (one-line description, config). The CLI's preset subcommands and the app's dropdown both read it."""

if TYPE_CHECKING:
    SlamPreset = SlamConfig
else:
    SlamPreset = tyro.extras.subcommand_type_from_defaults(
        {name: cfg for name, (_, cfg) in SLAM_PRESETS.items()},
        {name: description for name, (description, _) in SLAM_PRESETS.items()},
        prefix_names=False,
    )
"""``SlamConfig`` for code; for tyro, one subcommand per preset whose defaults are that preset's values."""


def flat_config(cfg: SlamConfig) -> dict[str, bool | int | float | str]:
    """``cfg`` as one level of dotted keys (``tracking.max_iters``), for a recording property."""
    flat: dict[str, bool | int | float | str] = {}
    for name, value in dataclasses.asdict(cfg).items():
        if isinstance(value, dict):
            flat.update({f"{name}.{key}": leaf for key, leaf in value.items()})
        else:
            flat[name] = value
    return flat

