"""Anatomical correspondences for SOMA's 25-joint hand order."""

SOMA_HAND25_TO_COCO21: tuple[int, ...] = (0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24)
"""Wrist then thumb/index/middle/ring/little; omit non-thumb metacarpals."""
