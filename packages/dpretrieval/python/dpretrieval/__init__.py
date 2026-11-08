"""ORB image retrieval for DPV-SLAM's classical loop closure, on kornia-rs (`dpretrieval._core`).

`DPRetrieval` detects ORB features with OpenCV (`cv2.ORB_create()`), keeps a kornia-bow bag of words of each frame
over ORB-SLAM3's `ORBvoc.txt` in an inverted-file database (kornia-staging-bow) and, for a query frame, returns the
best match at least `rad` frames away with its cross-checked ORB correspondences (kornia-imgproc).
"""

import cv2
import numpy as np
from jaxtyping import UInt8
from numpy import ndarray

from dpretrieval._core import Database

__all__ = ["DPRetrieval"]

MatchList = list[tuple[float, float, float, float, float]]
"""(train_x, train_y, query_x, query_y, hamming distance) per match."""

MAX_RESULTS: int = 4
"""Database results considered per query, as the C++ module."""


class DPRetrieval:
    """ORB retrieval over the frames inserted so far (frame index = insertion order)."""

    def __init__(self, vocab_path: str, rad: int) -> None:
        """Load the ORB vocabulary (`ORBvoc.txt`); queries ignore frames closer than `rad` to the query frame."""
        print(f"Loading the vocabulary {vocab_path}")
        self._database: Database = Database(vocab_path)
        self._rad: int = rad
        self._orb = cv2.ORB_create()

    def insert_image(self, array: UInt8[ndarray, "h w 3"]) -> None:
        """Detect ORB features in an (H, W, 3) uint8 image and add the frame."""
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("invalid image shape")
        keypoints, descriptors = self._orb.detectAndCompute(np.ascontiguousarray(array), None)
        points = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 2)
        self._database.add(points, descriptors if descriptors is not None else np.zeros((0, 32), dtype=np.uint8))

    def match_pair(self, ti: int, qi: int) -> MatchList:
        """Cross-checked brute-force Hamming matches of frame `qi`'s ORB features against frame `ti`'s."""
        return self._database.match_pair(ti, qi)

    def query(self, i: int) -> tuple[float, int, MatchList]:
        """The best-scoring frame at least `rad` frames from frame `i`, as (score, frame, matches); (-1, -1, []) if none."""
        if i < 0 or i >= len(self._database):
            raise ValueError("index invalid")
        candidates = ((score, j) for j, score in self._database.query(i, MAX_RESULTS) if abs(j - i) >= self._rad)
        best = max(candidates, key=lambda candidate: candidate[0], default=None)  # equal scores: the better-ranked one
        if best is None:
            return -1.0, -1, []
        score, j = best
        return score, j, self.match_pair(i, j)
