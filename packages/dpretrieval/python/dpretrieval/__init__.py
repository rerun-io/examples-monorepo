"""ORB image retrieval for DPV-SLAM's classical loop closure: a DBoW2 vocabulary-tree database (in Rust).

The API of the former pybind11 module (DBoW2 in C++). `DPRetrieval` detects ORB features with OpenCV
(`cv2.ORB_create()`, the defaults of the C++ `cv::ORB::create()`), keeps a DBoW2 database of their TF-IDF bags
of words (`dpretrieval._core`, packages/dpretrieval/crates/dbow) and, for a query frame, returns the best
match at least `rad` frames away with its cross-checked ORB correspondences.
"""

import cv2
import numpy as np
from jaxtyping import Float32, UInt8
from numpy import ndarray

from dpretrieval._core import Database, cross_check_match

__all__ = ["DPRetrieval"]

MatchList = list[tuple[float, float, float, float, float]]
"""(train_x, train_y, query_x, query_y, hamming distance) per match."""

MAX_RESULTS: int = 4
"""Database results considered per query, as the C++ module."""


class DPRetrieval:
    """A DBoW2 database of ORB features over the frames inserted so far (frame index = insertion order)."""

    def __init__(self, vocab_path: str, rad: int) -> None:
        """Load the ORB vocabulary (`ORBvoc.txt`); queries ignore frames closer than `rad` to the query frame."""
        print(f"Loading the vocabulary {vocab_path}")
        self._database: Database = Database(vocab_path)
        self._rad: int = rad
        self._orb = cv2.ORB_create()
        self._points: list[Float32[ndarray, "n 2"]] = []
        self._descriptors: list[UInt8[ndarray, "n 32"]] = []

    def insert_image(self, array: UInt8[ndarray, "h w 3"]) -> None:
        """Detect ORB features in an (H, W, 3) uint8 image and add the frame to the database."""
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("invalid image shape")
        keypoints, descriptors = self._orb.detectAndCompute(np.ascontiguousarray(array), None)
        if descriptors is None:
            descriptors = np.zeros((0, 32), dtype=np.uint8)
        self._points.append(np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 2))
        self._descriptors.append(descriptors)
        self._database.add(descriptors)

    def match_pair(self, ti: int, qi: int) -> MatchList:
        """Cross-checked brute-force Hamming matches of frame `qi`'s ORB features against frame `ti`'s."""
        train_pts, query_pts = self._points[ti], self._points[qi]
        return [
            (float(train_pts[t, 0]), float(train_pts[t, 1]), float(query_pts[q, 0]), float(query_pts[q, 1]), float(d))
            for q, t, d in cross_check_match(self._descriptors[qi], self._descriptors[ti])
        ]

    def query(self, i: int) -> tuple[float, int, MatchList]:
        """The best-scoring frame at least `rad` frames from frame `i`, as (score, frame, matches); (-1, -1, []) if none."""
        if i < 0 or i >= len(self._descriptors):
            raise ValueError("index invalid")
        # The C++ module held the best score in a float and compared each double score against it.
        best_score, best_j, best_matches = np.float32(-1.0), -1, []
        for j, score in self._database.query(i, MAX_RESULTS):
            if abs(j - i) >= self._rad and score > float(best_score):
                best_score, best_j, best_matches = np.float32(score), j, self.match_pair(i, j)
        return float(best_score), best_j, best_matches
