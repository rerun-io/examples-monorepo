# [vipe-src] Replaces ViPE's vendored DPVO tree (recipes/vipe-src/build.sh deletes it and copies this directory here).
"""MetricTrail's DPVO, re-exported from the external ``dpvo`` package (packages/dpvo, ``dpvo.metric``).

The modules keep the paths that ``vipe.metrictrail.slam`` and ``dino_fusion`` import:
``dpvo`` (the tracker), ``config`` (the yacs defaults), ``net``, ``altcorr``,
``projective_ops`` and ``lietorch``. The consuming environment supplies ``dpvo``
(an editable path dependency, like ``lietorch_mojo``), and with it the CUDA kernels.
"""
