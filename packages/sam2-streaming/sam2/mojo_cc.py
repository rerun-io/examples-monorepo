"""SAM 2's connected components on the Mojo kernel (`sam2/mojo/`), replacing the CUDA extension `sam2._C`.

Only `build_sam2_generic_video_predictor(..., fill_holes=True)` uses it. The extension is built by the
`sam2-mojo-build` task (sam2-streaming-dev environment) and runs on PyTorch's current CUDA stream through
lietorch-mojo's runtime, which the environment must provide.
"""

import torch
from torch import Tensor


def get_connected_components(mask: Tensor) -> tuple[Tensor, Tensor]:
    """8-connected components of (N, 1, H, W) masks (H, W even), as `sam2._C.get_connected_componnets`.

    Returns `(labels, counts)`, int32 (N, 1, H, W): each foreground pixel's component label (its smallest
    2x2-block index + 1; 0 = background) and the component's area (0 = background).
    """
    if mask.dim() != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must have shape (N, 1, H, W), got {tuple(mask.shape)}")
    if mask.shape[2] % 2 or mask.shape[3] % 2:
        raise ValueError(f"mask height and width must be even, got {tuple(mask.shape[2:])}")
    if mask.device.type != "cuda" or (mask.device.index or 0) != 0:
        raise RuntimeError(f"connected components run on cuda:0 only, got {mask.device}")
    # After the checks: environments without the Mojo toolchain (posekit, mamma) still validate inputs.
    from lietorch_mojo.backends import _current_raw_stream, _runtime

    from sam2 import _sam2_mojo

    mask = mask.to(torch.uint8).contiguous()
    labels, counts_init, counts = (torch.zeros(mask.shape, dtype=torch.int32, device=mask.device) for _ in range(3))
    specs = tuple((t.data_ptr(), *t.shape) for t in (mask, labels, counts_init, counts))
    _sam2_mojo.connected_components(_runtime(True), (_current_raw_stream(0),), specs)
    return labels, counts
