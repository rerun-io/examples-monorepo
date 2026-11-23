"""The ``moge2_prior`` maps from monopriors' dynamic TensorRT engine agree with the torch predictor on a real MAMMA clip.

``prior_layer.compute_prior_layer`` runs the engine; this pins the band against the torch twin on the clip's own frames
(undistorted, resized, calibrated ``fov_x``), compared as ``compute_clip_priors`` returns them (depth float16, 0 =
invalid; normals PNG-coded). The first run on a GPU builds the engine (about 90 s).
"""

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.integration

DATASET: str = "dataforge-mamma-markerless-dance"
SEGMENT: str = "mamma-markerless-dance__140725_Breakdancemarkers_Improv_2_03686_1_1_1"
VALID_IOU_MIN: float = 0.99
"""Least intersection-over-union of the two valid masks (measured 1.0000 on both cameras)."""
# Limits = the worst measured camera x 1.5 (2026-11-22, RTX 5090, TensorRT 11.2; two passes, bit-identical). Measured
# rig_01 / rig_17: depth median 0.0111 / 0.0190, p95 0.0250 / 0.0862; normals mean 0.19 / 1.43 deg, p99 0.64 / 9.74 deg;
# least per-frame pearson 0.99598 / 0.99593. The gap is mostly a per-frame metric scale (0.92-1.01); the loss's depth
# term is pearson (scale and shift invariant). Against an fp32 torch reference, TRT sits 2-10x farther than torch fp16.
DEPTH_MEDIAN_MAX: float = 0.0285
"""Largest median relative depth difference where both are valid."""
DEPTH_P95_MAX: float = 0.129
"""Largest 95th-percentile relative depth difference."""
NORMAL_MEAN_DEG_MAX: float = 2.15
"""Largest mean normal angle difference, degrees."""
NORMAL_P99_DEG_MAX: float = 14.6
"""Largest 99th-percentile normal angle difference, degrees."""
PEARSON_GAP_MAX: float = 6.1e-3
"""Largest ``1 - pearson`` of one frame's depth maps (the prior's depth loss statistic)."""


def test_trt_prior_maps_match_torch_on_a_mamma_clip(spark_catalog_url: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("MoGe-2 needs CUDA")
    from monopriors.models.moge_v2.predictor import MoGeV2TorchPredictor, MoGeV2TrtPredictor

    from open4dgs.catalog_clip import read_catalog_clip
    from open4dgs.prior import decode_normals
    from open4dgs.prior_layer import compute_clip_priors

    clip = read_catalog_clip(
        spark_catalog_url, DATASET, SEGMENT, 0, 4, cameras=["rig_01/cam_00", "rig_17/cam_00"], max_side=514, decode_device=torch.device("cuda")
    )
    trt = compute_clip_priors(clip, clip.camera_names, MoGeV2TrtPredictor(encoder="vitl", heads="geometry", batch_size=4), batch_size=4)
    ref = compute_clip_priors(clip, clip.camera_names, MoGeV2TorchPredictor(encoder="vitl", heads="geometry"), batch_size=4)
    for cam, name in enumerate(clip.camera_names):
        depth, depth_ref = trt.depth[cam].float(), ref.depth[cam].float()
        valid, valid_ref = depth > 0, depth_ref > 0
        both = valid & valid_ref
        iou: float = (both.sum() / (valid | valid_ref).sum()).item()
        relative = ((depth - depth_ref).abs() / depth_ref)[both]
        normals, _ = decode_normals(trt.normals[cam])
        normals_ref, _ = decode_normals(ref.normals[cam])
        cosine = torch.nn.functional.cosine_similarity(normals[both], normals_ref[both], dim=-1).clamp(-1.0, 1.0)
        degrees = torch.rad2deg(torch.acos(cosine))
        centred, centred_ref = depth - depth.mean(dim=(1, 2), keepdim=True), depth_ref - depth_ref.mean(dim=(1, 2), keepdim=True)
        pearson: float = ((centred * centred_ref).sum(dim=(1, 2)) / (centred.norm(dim=(1, 2)) * centred_ref.norm(dim=(1, 2)))).min().item()
        summary: str = (
            f"{name}: valid IoU {iou:.4f}, depth median {relative.median().item():.5f} p95 {relative.quantile(0.95).item():.5f}, "
            f"normals mean {degrees.mean().item():.3f} p99 {degrees.quantile(0.99).item():.3f} deg, least per-frame pearson {pearson:.6f}"
        )
        print(summary)
        assert iou > VALID_IOU_MIN, summary
        assert relative.median().item() < DEPTH_MEDIAN_MAX and relative.quantile(0.95).item() < DEPTH_P95_MAX, summary
        assert degrees.mean().item() < NORMAL_MEAN_DEG_MAX and degrees.quantile(0.99).item() < NORMAL_P99_DEG_MAX, summary
        assert 1.0 - pearson < PEARSON_GAP_MAX, summary
