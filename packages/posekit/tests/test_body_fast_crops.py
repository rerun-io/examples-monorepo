"""GPU body crops retain the released image, mask and camera geometry."""

import numpy as np
import pytest
import torch

pytest.importorskip("yacs", reason="SAM body crop reference requires the body environment")


@pytest.mark.integration
@pytest.mark.parametrize("padding", [1.25, 0.9])
def test_cuda_crops_match_opencv(padding: float) -> None:
    from sam_3d_body.data.transforms import Compose, GetBBoxCenterScale, TopdownAffine, VisionTransformWrapper
    from sam_3d_body.data.utils.prepare_batch import prepare_batch
    from torchvision.transforms import ToTensor

    from posekit.models.sam3d_body_crops import BodyCropper

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    random = np.random.default_rng(8)
    image = random.integers(0, 256, (83, 127, 3), dtype=np.uint8)
    boxes = np.array([[5.3, 6.8, 78.5, 80.2], [-10.0, 21.3, 135.2, 90.8]], dtype=np.float32)
    masks = random.random((2, 83, 127), dtype=np.float32)
    scores = np.array([0.8, 0.7], dtype=np.float32)
    transform = Compose([GetBBoxCenterScale(padding), TopdownAffine(512), VisionTransformWrapper(ToTensor())])
    expected = prepare_batch(image, transform, boxes, masks, scores)
    cropper = BodyCropper(torch.device("cuda"))
    actual = cropper.prepare(
        torch.from_numpy(image).cuda()[None], torch.from_numpy(boxes).cuda(),
        torch.zeros(2, device="cuda", dtype=torch.int64), padding=padding,
        masks=torch.from_numpy(masks).cuda(), scores=torch.from_numpy(scores).cuda(),
    )
    for field in ("bbox_center", "bbox_scale", "affine_trans", "cam_int", "ori_img_size", "img_size"):
        torch.testing.assert_close(actual[field].cpu(), expected[field], atol=3e-5, rtol=1e-6)
    image_error = (actual["img"].cpu() - expected["img"]).abs()
    assert image_error.mean() < 0.0001
    assert image_error.max() <= 1.0 / 255 + 1e-7
    torch.testing.assert_close(actual["mask"].cpu(), expected["mask"], atol=1e-5, rtol=0.0)
