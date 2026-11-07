# [vipe-src] Replaces ViPE's vendored DPVO config (see __init__.py).
"""``cfg``: ViPE's DPVO yacs defaults as a ``CfgNode``, built from :data:`dpvo.metric.YACS_DEFAULTS` (all 43 keys).

``vipe.metrictrail.slam.resolve_dpvo_config`` checks a preset's key set against
it, clones it and merges the preset over it, exactly as with the vendored config.
"""

from dpvo.metric import YACS_DEFAULTS
from vipe.metrictrail.dpvo_deps import CfgNode

cfg = CfgNode(dict(YACS_DEFAULTS))
