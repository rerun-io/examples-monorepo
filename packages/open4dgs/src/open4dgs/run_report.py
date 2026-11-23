"""The report of one training run (``open4dgs-train``): printed as JSON, stored flat as the run recording's ``summary``
property, and summed up in one headline line."""

import dataclasses
from dataclasses import dataclass

from serde import serde

from open4dgs.compensation import BrightnessSplit
from open4dgs.evaluate import BASELINES, HEADLINE
from open4dgs.trigs_adapter import CameraCompensation

Means = dict[str, float]
"""Metric name -> mean over the frames and cameras of a group (``evaluate.mean_metrics``, ``mean_geometry``)."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RunReport:
    """Every number a run reports. Optional groups are None when the run did not compute them."""

    holdout: Means
    holdout_init: Means
    """The init cloud rendered without training."""
    holdout_floors: dict[str, Means]
    """Per floor render (``evaluate.BASELINES``)."""
    train: Means
    """The training cameras (empty with ``--no-eval-train-cameras``)."""
    geometry_holdout: Means
    brightness: dict[str, BrightnessSplit]
    """Held-out cameras, plus ``<camera>:odd`` / ``<camera>:odd_fit`` for the test-time fit."""
    brightness_train: dict[str, BrightnessSplit]
    learned: dict[str, CameraCompensation]
    holdout_fitted: dict[str, CameraCompensation]
    it_per_s: float
    train_s: float
    peak_gpu_gb: float
    init_rows: int
    final_gaussians: int
    loss_rgb_first: float | None
    loss_rgb_last: float | None
    prior_scalars: dict[str, float]
    """``prior_depth_first``, ``prior_depth_last``, ``prior_normal_*`` when the prior was on."""
    timings: dict[str, float]
    """``read_clip_s``, ``masks_s``, ``init_s``, ``prior_s``, ``eval_s``, ``compensation_eval_s``."""
    train_appearance: Means | None = None
    """The training cameras with their learned affine (``--train.learn-appearance``)."""
    holdout_odd_identity: Means | None = None
    holdout_odd_fit: Means | None = None

    def _groups(self) -> dict[str, Means]:
        """The metric groups by their summary name (``holdout_floor_<kind>`` for the floors)."""
        groups: dict[str, Means | None] = {
            "holdout": self.holdout,
            "holdout_init": self.holdout_init,
            **{f"holdout_floor_{kind}": self.holdout_floors[kind] for kind in BASELINES},
            "train": self.train,
            "train_appearance": self.train_appearance,
            "holdout_odd_identity": self.holdout_odd_identity,
            "holdout_odd_fit": self.holdout_odd_fit,
            "geometry_holdout": self.geometry_holdout,
        }
        return {name: means for name, means in groups.items() if means is not None}

    def _scalars(self) -> dict[str, float | int]:
        scalars: dict[str, float | int | None] = {
            "it_per_s": self.it_per_s,
            "train_s": self.train_s,
            "peak_gpu_gb": self.peak_gpu_gb,
            "init_rows": self.init_rows,
            "final_gaussians": self.final_gaussians,
            "loss_rgb_first": self.loss_rgb_first,
            "loss_rgb_last": self.loss_rgb_last,
            **self.prior_scalars,
            **self.timings,
        }
        return {name: value for name, value in scalars.items() if value is not None}

    def summary(self) -> dict[str, object]:
        """The printed JSON: one key per metric group, the per-camera splits and compensations, then the scalars."""
        return {
            **self._groups(),
            "brightness": {name: dataclasses.asdict(split) for name, split in self.brightness.items()},
            "brightness_train": {name: dataclasses.asdict(split) for name, split in self.brightness_train.items()},
            "learned": {name: dataclasses.asdict(comp) for name, comp in self.learned.items()},
            "holdout_fitted": {name: dataclasses.asdict(comp) for name, comp in self.holdout_fitted.items()},
            **self._scalars(),
        }

    def summary_properties(self) -> dict[str, float]:
        """The ``summary`` property: ``<group>_<metric>``, ``brightness_<camera>_<field>`` (held-out cameras) and the scalars."""
        flat: dict[str, float] = {f"{group}_{k}": float(v) for group, means in self._groups().items() for k, v in means.items()}
        flat.update(
            {f"brightness_{name.replace('/', '_')}_{k}": float(v) for name, split in self.brightness.items() for k, v in dataclasses.asdict(split).items()}
        )
        flat.update({k: float(v) for k, v in self._scalars().items()})
        return flat

    def headline(self) -> str:
        """One line: held-out crop_raw PSNR/SSIM/LPIPS, with the init and both floors beside it, plus fg PSNR and alpha IoU."""
        groups: dict[str, Means] = self._groups()

        def triple(group: str) -> str:
            values: Means = groups.get(group, {})
            if not values:
                return "n/a"
            return "/".join(f"{values[name]:.3f}" if name != HEADLINE[0] else f"{values[name]:.2f}" for name in HEADLINE) + f" fg {values['fg_psnr']:.2f} iou {values['alpha_iou']:.3f}"

        names = ("holdout", "holdout_init", *(f"holdout_floor_{kind}" for kind in BASELINES), "train")
        return "[open4dgs] crop_raw PSNR/SSIM/LPIPS " + "; ".join(f"{name} {triple(name)}" for name in names)
