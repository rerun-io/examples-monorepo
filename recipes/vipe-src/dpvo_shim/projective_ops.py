# [vipe-src] Replaces ViPE's vendored DPVO projective ops (see __init__.py).
"""Re-exports :func:`dpvo.projective_ops.point_cloud`, the one op ``slam.py`` reads (the keyframe patch map)."""

from dpvo.projective_ops import point_cloud as point_cloud
