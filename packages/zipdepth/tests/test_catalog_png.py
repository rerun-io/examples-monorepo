"""Bit-exact test for ZipDepth's CUDA PNG Up unfilter."""

import numpy as np
import pytest
import torch
from jaxtyping import Int32, UInt8, UInt16
from numpy import ndarray
from numpy.testing import assert_array_equal
from torch import Tensor

pytest.importorskip("imagecodecs", reason="Fast PNG inflate lives in the zipdepth catalog lane")
pytest.importorskip("arkitscenes_download", reason="Representative depth encoding lives in the zipdepth catalog lane")

from arkitscenes_download.ingest.depth import decode_depth_png, encode_depth_png, inflate_depth_png_rows  # noqa: E402

from zipdepth.catalog.png import unfilter_up_cuda  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Up reconstruction needs a GPU")
def test_unfilter_up_cuda_matches_cpu_decode_for_ingest_pngs() -> None:
    """Reconstruct Up-filtered uint16 depth on CUDA bit-exactly."""
    rng: np.random.Generator = np.random.default_rng(7)
    expected_hw: UInt16[ndarray, "h w"] = rng.integers(100, 4001, (96, 128), dtype=np.uint16)
    blob_bytes: UInt8[ndarray, "n"] = np.frombuffer(encode_depth_png(expected_hw, level=1), dtype=np.uint8)

    inflated: tuple[UInt8[ndarray, "h row_bytes"], tuple[int, int]] | None = inflate_depth_png_rows(blob_bytes)
    assert inflated is not None
    filtered_hwb: UInt8[ndarray, "h row_bytes"] = inflated[0]
    assert inflated[1] == expected_hw.shape
    filtered_cuda: UInt8[Tensor, "h row_bytes"] = torch.from_numpy(filtered_hwb).to(device="cuda")

    actual_cuda: Int32[Tensor, "h w"] = unfilter_up_cuda(filtered_cuda)

    assert actual_cuda.dtype == torch.int32
    assert actual_cuda.device.type == "cuda"
    assert_array_equal(actual_cuda.cpu().numpy(), decode_depth_png(blob_bytes))
