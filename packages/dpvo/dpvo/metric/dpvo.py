"""MetricDPVO: DPVO with a per-frame metric-depth prior and the MetricTrail global-BA backend.

The port of ViPE's MetricTrail DPVO fork (``vipe/metrictrail/dpvo/dpvo.py``)
onto the shared core: everything it adds plugs into the hooks of
:class:`~dpvo.dpvo.DPVO`. With the configs of the presets ``metrictrail_local``,
``metrictrail_gba`` and ``metrictrail_gba_refine`` (:meth:`MetricDPVOConfig.from_yacs`)
it runs ViPE's operations in ViPE's order.
"""

from collections.abc import Sequence

import torch
from jaxtyping import Bool, Float, Float32, Float64, UInt8
from numpy import ndarray
from torch import Tensor

from dpvo.dpvo import DPVO
from dpvo.hooks import DPVOObserver
from dpvo.metric.backend import CulledInterp, FinalRemeasure
from dpvo.metric.config import MetricDPVOConfig
from dpvo.metric.depth_prior import METRIC_DISPS, METRIC_WEIGHTS, MetricDepthPrior
from dpvo.metric.graph import MetricGraphPolicy
from dpvo.net import VONet
from dpvo.utils import DEFAULT_DEVICE


class MetricDPVO(DPVO):
    """DPVO whose patch depths and BA prior come from per-frame metric depth.

    Args:
        cfg: Tracker config.
        network: Checkpoint path or a ready :class:`~dpvo.net.VONet`.
        ht: Image height in pixels (divisible by 4).
        wd: Image width in pixels (divisible by 4).
        device: Device of the network and every buffer.
        observers: Extra run callbacks, called after MetricDPVO's own.
    """

    def __init__(
        self,
        cfg: MetricDPVOConfig,
        network: str | VONet,
        ht: int = 480,
        wd: int = 640,
        *,
        device: torch.device = DEFAULT_DEVICE,
        observers: Sequence[DPVOObserver] = (),
    ) -> None:
        self.metric_cfg: MetricDPVOConfig = cfg
        self.metric_prior: MetricDepthPrior = MetricDepthPrior(cfg)
        self.metric_graph: MetricGraphPolicy = MetricGraphPolicy(cfg)
        own: list[DPVOObserver] = [self.metric_prior]
        if cfg.remeasure_final != "off":
            own.append(FinalRemeasure(cfg))
        if cfg.culled_interp:
            own.append(CulledInterp())
        super().__init__(
            cfg, network, ht, wd, device=device, depth_prior=self.metric_prior, graph_policy=self.metric_graph, observers=[*own, *observers]
        )
        self.pg.add_frame_attr(METRIC_DISPS, (self.M,), torch.float32)
        self.pg.add_frame_attr(METRIC_WEIGHTS, (self.M,), torch.float32)

    def __call__(  # pyrefly: ignore[bad-override]
        self,
        tstamp: int,
        image: UInt8[Tensor, "3 ht wd"],
        intrinsics: Float[Tensor, "4"],
        metric_depth: Float32[Tensor, "ht wd"] | None = None,
        metric_mask: Bool[Tensor, "ht wd"] | None = None,
    ) -> None:
        """Track a frame with its metric z-depth (the parameter names are read by ViPE's slam.py).

        Args:
            tstamp: Caller timestamp (ordinal frame index).
            image: BGR image on the tracker's device.
            intrinsics: ``fx fy cx cy`` in pixels.
            metric_depth: z-depth in metres; required.
            metric_mask: Pixels of ``metric_depth`` that may be used, or ``None`` for all.

        Raises:
            ValueError: The frame has no metric depth or no valid prior; no state has changed.
        """
        super().__call__(tstamp, image, intrinsics, depth=metric_depth, depth_mask=metric_mask)

    def terminate(self) -> tuple[Float32[ndarray, "n_frames 7"], Float64[ndarray, "n_frames"]]:
        """Finish as DPVO does, then release the stored factors (with ``metric_global_ba``)."""
        result: tuple[Float32[ndarray, "n_frames 7"], Float64[ndarray, "n_frames"]] = super().terminate()
        if self.metric_cfg.metric_global_ba:
            self.release_inactive_factors()
        return result

    def metric_prior_diagnostics(self) -> dict[str, int | bool]:
        """Counters that ViPE's depth producer and slam record check after terminate."""
        result: dict[str, int | bool] = {
            "keyframes_seen": self.metric_prior.keyframes_seen,
            "regularization_enabled": self.metric_prior.regularization_enabled,
            "successful_ba_calls_with_metric_prior": self.metric_prior.ba_calls_with_prior,
            "inactive_factors": int(self.pg.ii_inac.numel()),
        }
        if self.metric_cfg.metric_global_ba:
            result.update(
                {
                    "metric_global_ba": True,
                    "global_ba_calls": self.metric_prior.global_ba_calls,
                    "global_ba_calls_with_metric_prior": self.metric_prior.global_ba_calls_with_prior,
                    "loop_edges_added": self.metric_graph.loop_edges_added,
                    "normalize_called": False,
                }
            )
        return result
