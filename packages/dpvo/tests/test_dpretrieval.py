"""dpretrieval (DBoW2 + OpenCV's cross-checked matcher, in Rust) on a synthetic vocabulary and synthetic images.

The port was checked against the former C++ module (pybind11 + DBoW2) on 2 x 400 frames of two apartment videos
with ORBvoc.txt: every query's (score, frame, matches) was identical. The KITTI 00 loop-closure test
(test_classic_loop_closure.py) runs it end to end.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2", reason="dpretrieval detects ORB features with OpenCV")
dpretrieval = pytest.importorskip("dpretrieval", reason="classic loop closure needs dpretrieval")


def _vocabulary(path: Path) -> Path:
    """A two-level tree: 8 random root children, 4 random leaves under each, idf weight 1."""
    rng = np.random.default_rng(0)
    lines = ["8 2 0 0"]
    for _ in range(8):
        lines.append("0 0 " + " ".join(map(str, rng.integers(0, 256, 32))) + " 0")
    for parent in range(1, 9):
        for _ in range(4):
            lines.append(f"{parent} 1 " + " ".join(map(str, rng.integers(0, 256, 32))) + " 1")
    path.write_text("\n".join(lines) + "\n")
    return path


def _image(seed: int) -> np.ndarray:
    """A textured image ORB finds corners in: random blocks, upsampled."""
    blocks = np.random.default_rng(seed).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    return np.ascontiguousarray(np.repeat(np.repeat(blocks, 10, axis=0), 10, axis=1))


def test_query_finds_the_repeated_frame_beyond_the_radius(tmp_path: Path) -> None:
    db = dpretrieval.DPRetrieval(str(_vocabulary(tmp_path / "voc.txt")), 3)
    for seed in range(4):
        db.insert_image(_image(seed))
    assert db.query(1) == (-1.0, -1, [])  # frames 0-3 are all closer than rad to frame 1
    db.insert_image(_image(0))
    score, j, matches = db.query(4)
    assert (score, j) == (1.0, 0)  # the same image: identical bags of words
    assert matches and all(m[4] == 0.0 for m in matches)  # cross-checked ORB matches of an image with itself


def test_bad_input_raises(tmp_path: Path) -> None:
    db = dpretrieval.DPRetrieval(str(_vocabulary(tmp_path / "voc.txt")), 3)
    with pytest.raises(ValueError, match="image shape"):
        db.insert_image(np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="index invalid"):
        db.query(0)
    with pytest.raises(ValueError, match="DBoW2 text vocabulary"):
        dpretrieval.DPRetrieval(str(_write(tmp_path / "bad.txt", "99 99 0 0\n")), 3)


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path
