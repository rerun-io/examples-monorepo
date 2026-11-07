# [vipe-src] Replaces ViPE's vendored DPVO tracker (see __init__.py).
"""``DPVO``: :class:`dpvo.metric.MetricDPVO` behind the constructor that ``vipe.metrictrail.slam`` calls.

``slam.py`` builds ``DPVO(cfg, network, ht=..., wd=..., viz=False)`` with a yacs
``CfgNode`` (uppercase keys, merged over :mod:`.config`); this class maps it
through :meth:`dpvo.metric.MetricDPVOConfig.from_yacs`. ``__call__``,
``terminate`` and ``metric_prior_diagnostics`` are MetricDPVO's.
"""

from collections.abc import Mapping

from dpvo.metric import MetricDPVO, MetricDPVOConfig
from dpvo.net import VONet


class DPVO(MetricDPVO):
    """MetricDPVO built from a ViPE DPVO yacs config."""

    def __init__(self, cfg: Mapping[str, object], network: str | VONet, ht: int = 480, wd: int = 640, viz: bool = False) -> None:
        """Build the tracker on the default device (CUDA).

        Args:
            cfg: The 43-key DPVO yacs block (a ``CfgNode``).
            network: Checkpoint path or a ready ``VONet``.
            ht: Image height in pixels.
            wd: Image width in pixels.
            viz: Must be False: there is no DPVO viewer.

        Raises:
            ValueError: ``viz`` is set, or ``cfg`` is not a valid MetricDPVO config.
        """
        if viz:
            raise ValueError("MetricTrail's DPVO has no viewer (viz=True)")
        super().__init__(MetricDPVOConfig.from_yacs(cfg), network, ht, wd)
