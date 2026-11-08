"""dpretrieval (OpenCV ORB, kornia-bow bags of words, kornia-staging-bow database) on a synthetic vocabulary.

Against the DBoW2 port it replaces, on 2 x 400 frames of two apartment videos with ORBvoc.txt: the same frame for
every query (15 loops), the same 14 candidates above dpvo's loop_retr_thresh and the same 3,833 matches; scores
differ by at most 1e-6 relative (kornia-bow keeps idf weights in float32). The KITTI 00 loop-closure test
(test_classic_loop_closure.py) runs it end to end.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2", reason="dpretrieval detects ORB features with OpenCV")
dpretrieval = pytest.importorskip("dpretrieval", reason="classic loop closure needs dpretrieval")


def _vocabulary(path: Path) -> Path:
    """A two-level tree (branching 10, as ORBvoc.txt): 10 random root children, 4 random leaves under each, idf 1."""
    rng = np.random.default_rng(0)
    lines = ["10 2 0 0"]
    for _ in range(10):
        lines.append("0 0 " + " ".join(map(str, rng.integers(0, 256, 32))) + " 0")
    for parent in range(1, 11):
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
    assert j == 0 and score == pytest.approx(1.0, abs=1e-6)  # the same image: identical bags of words
    assert matches and all(m[4] == 0.0 for m in matches)  # cross-checked ORB matches of an image with itself


def test_bad_input_raises(tmp_path: Path) -> None:
    db = dpretrieval.DPRetrieval(str(_vocabulary(tmp_path / "voc.txt")), 3)
    with pytest.raises(ValueError, match="image shape"):
        db.insert_image(np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="index invalid"):
        db.query(0)
    with pytest.raises(ValueError, match="cannot load the ORB vocabulary"):
        dpretrieval.DPRetrieval(str(_write(tmp_path / "bad.txt", "99 99 0 0\n")), 3)


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path
