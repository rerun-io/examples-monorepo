"""SAM 2's connected components (Mojo port of sam2/csrc's CUDA extension) and the predictor's hole-fill flag.

The CUDA kernel labels each 8-connected component with its smallest 2x2-block index + 1 (union-find always
links the larger root to the smaller), so labels are deterministic and the reference is exact: OpenCV's
components, relabelled by that rule, with their areas.
"""

import numpy as np
import pytest
import torch
from numpy import ndarray

cv2 = pytest.importorskip("cv2", reason="the reference labelling is OpenCV's")


def _reference(mask: ndarray) -> tuple[ndarray, ndarray]:
    _, _, height, width = mask.shape
    labels: ndarray = np.zeros(mask.shape, np.int32)
    counts: ndarray = np.zeros(mask.shape, np.int32)
    rows, cols = np.mgrid[0:height, 0:width]
    block: ndarray = (rows // 2 * 2) * width + cols // 2 * 2
    for i in range(mask.shape[0]):
        num, components, stats, _ = cv2.connectedComponentsWithStats(mask[i, 0], connectivity=8)
        for c in range(1, num):
            selected: ndarray = components == c
            labels[i, 0][selected] = block[selected].min() + 1
            counts[i, 0][selected] = stats[c, cv2.CC_STAT_AREA]
    return labels, counts


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.parametrize(
    ("shape", "density"),
    [((3, 1, 64, 96), 0.5), ((2, 1, 256, 256), 0.3), ((1, 1, 256, 256), 0.7), ((4, 1, 2, 2), 0.5), ((1, 1, 30, 1000), 0.45)],
)
def test_connected_components_match_opencv(shape: tuple[int, int, int, int], density: float) -> None:
    pytest.importorskip("sam2._sam2_mojo", reason="build the Mojo kernel first (pixi run -e sam2-streaming-dev sam2-mojo-build)")
    from sam2.utils.misc import get_connected_components

    mask: ndarray = (np.random.default_rng(sum(shape)).random(shape) < density).astype(np.uint8)
    labels, counts = get_connected_components(torch.from_numpy(mask).cuda())
    ref_labels, ref_counts = _reference(mask)
    np.testing.assert_array_equal(labels.cpu().numpy(), ref_labels)
    np.testing.assert_array_equal(counts.cpu().numpy(), ref_counts)


def test_odd_sizes_and_wrong_shapes_raise() -> None:
    from sam2.mojo_cc import get_connected_components

    with pytest.raises(ValueError, match="even"):
        get_connected_components(torch.zeros(1, 1, 3, 4, dtype=torch.uint8))
    with pytest.raises(ValueError, match=r"\(N, 1, H, W\)"):
        get_connected_components(torch.zeros(1, 2, 4, 4, dtype=torch.uint8))


@pytest.mark.parametrize(("fill_holes", "max_hole_area"), [(False, 0.0), (True, 8.0)])
def test_hole_fill_is_off_unless_asked(fill_holes: bool, max_hole_area: float) -> None:
    from sam2.build_sam import build_sam2_generic_video_predictor

    model = build_sam2_generic_video_predictor("configs/sam2.1/sam2.1_hiera_t.yaml", device="cpu", fill_holes=fill_holes)
    assert model._transforms.max_hole_area == max_hole_area
