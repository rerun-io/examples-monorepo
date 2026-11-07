"""MetricDPVO: DPVO with a per-frame metric-depth prior and ViPE MetricTrail's global-BA backend.

- :class:`MetricDPVO` -- the tracker (:mod:`dpvo.metric.dpvo`).
- :class:`MetricDPVOConfig` -- its config; ``from_yacs``/``to_yacs`` map ViPE's uppercase keys.
- :data:`YACS_DEFAULTS` -- ViPE's DPVO yacs defaults, all 43 keys.
"""

from dpvo.metric.config import YACS_DEFAULTS as YACS_DEFAULTS
from dpvo.metric.config import MetricDPVOConfig as MetricDPVOConfig
from dpvo.metric.dpvo import MetricDPVO as MetricDPVO
