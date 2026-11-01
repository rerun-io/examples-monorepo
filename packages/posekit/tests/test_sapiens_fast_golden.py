"""Compare production poses with the recorded d31 bf16 reference."""

from pathlib import Path

import numpy as np
import pytest

# ruff: noqa: E402
pytest.importorskip("torch")
import torch

from posekit.models.sapiens import SapiensPoseConfig
from posekit.predictions import BoxDetections


@pytest.mark.golden
def test_fast_sapiens_four_rig_d31_positions_and_scores() -> None:
    reference_dir = Path("/tmp/selfcap-learn/pose-compare/d31")
    rigs: tuple[str, ...] = ("rig_00", "rig_01", "rig_02", "rig_03")
    for rig in rigs:
        for suffix in (".mp4", ".npz"):
            asset: Path = reference_dir / f"{rig}{suffix}"
            if not asset.exists():
                pytest.skip(f"Missing golden asset {asset}")
    decoder_module = pytest.importorskip("torchcodec.decoders", reason="torchcodec needed for rig_00.mp4")
    from sapiens2_pose.api.metadata import get_sapiens_metainfo

    model = SapiensPoseConfig().setup()
    mapping = get_sapiens_metainfo()["coco_wholebody_to_goliath_mapping"]
    mapped = sorted(int(index) for index in mapping if int(index) < 133)
    hands = np.array([index >= 91 for index in mapped])
    errors, scores, reference_scores, sizes = [], [], [], []
    for rig in rigs:
        decoder = decoder_module.VideoDecoder(str(reference_dir / f"{rig}.mp4"), device="cuda", dimension_order="NHWC", seek_mode="exact")
        with np.load(reference_dir / f"{rig}.npz") as reference:
            for row in range(0, len(reference["t"]), 8):
                boxes = reference["boxes"][row]
                slots = np.flatnonzero(np.isfinite(boxes[:, 0]))
                if not len(slots):
                    continue
                detections = BoxDetections(
                    torch.from_numpy(boxes[slots].astype(np.float32)).cuda(),
                    torch.from_numpy(reference["box_scores"][row, slots].astype(np.float32)).cuda(),
                    torch.zeros(len(slots), dtype=torch.long, device="cuda"),
                )
                offset: int = 236 if rig == "rig_02" else 237
                frames = decoder.get_frames_at([row + offset]).data
                prediction = model(frames, detections)
                expected = reference["sapiens_xy"][row, slots][:, mapped]
                errors.append(np.linalg.norm(prediction.xy.cpu().numpy()[:, mapped] - expected, axis=-1))
                scores.append(prediction.scores.cpu().numpy()[:, mapped])
                reference_scores.append(reference["sapiens_scores"][row, slots][:, mapped])
                sizes.append(np.maximum(boxes[slots, 2] - boxes[slots, 0], boxes[slots, 3] - boxes[slots, 1]))
    error = np.concatenate(errors)
    score = np.concatenate(scores)
    ref_score = np.concatenate(reference_scores)
    size = np.concatenate(sizes)
    visible = ref_score >= 0.3
    hit = error <= 0.01 * size[:, None]
    assert hit[:, ~hands][visible[:, ~hands]].mean() >= 0.89
    assert hit[:, hands][visible[:, hands]].mean() >= 0.95
    assert ((score >= 0.7) == (ref_score >= 0.7)).mean() >= 0.96
