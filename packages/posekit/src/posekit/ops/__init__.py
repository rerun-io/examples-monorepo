"""Shared GPU (torch) pre/postprocessing ops used by every posekit model.

These replace the per-package copies of bbox->center/scale math, affine crop
generation, and heatmap/SimCC decoding that previously lived in sapiens2-pose,
sapiens-coco133-pose, mamma, and wilor-nano.
"""
