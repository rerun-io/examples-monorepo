"""Typed reads of upstream's run directory."""

import json
from pathlib import Path

import numpy as np
import pytest

from fdanyone_rerun.outputs import read_camera_rig, read_run_metadata


def _camera(camera_id: int) -> dict:
    return {
        "camera_id": camera_id,
        "layer_index": 0,
        "pitch": 15,
        "yaw": 60.0 * camera_id,
        "K": [[900.0, 0.0, 352.0], [0.0, 900.0, 640.0], [0.0, 0.0, 1.0]],
        "camera_to_world": np.eye(4).tolist(),
        "image_width": 704,
        "image_height": 1280,
        "video": f"videos/{camera_id:02d}.mp4",
        "skeleton_video": f"skeletons/{camera_id:02d}.mp4",
    }


def _rig(cameras: list[dict], camera_model: str = "OPENCV") -> dict:
    # world_frame, camera_frame and framing are upstream's; the reader allows fields it does not use.
    return {
        "camera_model": camera_model,
        "world_frame": {"name": "canonical_human_world"},
        "camera_frame": {},
        "framing": {},
        "front_camera_ids": [0],
        "cameras": cameras,
    }


def test_reads_the_camera_rig(tmp_path: Path) -> None:
    (tmp_path / "cameras.json").write_text(json.dumps(_rig([_camera(0), _camera(1)])))
    rig = read_camera_rig(tmp_path)
    assert [camera.video for camera in rig.cameras] == ["videos/00.mp4", "videos/01.mp4"]
    assert rig.cameras[1].intrinsics.shape == (3, 3)
    assert rig.cameras[1].world_T_cam.dtype == np.float64


@pytest.mark.parametrize(("cameras", "camera_model"), [([_camera(1), _camera(0)], "OPENCV"), ([_camera(0)], "PINHOLE")])
def test_rejects_a_rig_out_of_contract(tmp_path: Path, cameras: list[dict], camera_model: str) -> None:
    (tmp_path / "cameras.json").write_text(json.dumps(_rig(cameras, camera_model)))
    with pytest.raises(ValueError, match="cameras.json"):
        read_camera_rig(tmp_path)


def test_reads_the_run_metadata(tmp_path: Path) -> None:
    metadata = {
        "input": {"filename": "clip.mp4", "fps": "25/1", "start_time_seconds": 0.0, "num_frames": 121, "width": 2160, "height": 3840},
        "generation": {
            "seed": 42,
            "attention_backend": "sageattention",
            "inference_steps": 4,
            "elapsed_seconds": {"target_denoise": 60.5},
            "total_elapsed_seconds": 190.0,
            "peak_vram_allocated_bytes": 22_000_000_000,
            "peak_vram_reserved_bytes": 23_000_000_000,
            "view_plan": {"num_target_views": 6},
        },
        "motion": {"method": "GVHMR"},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    run = read_run_metadata(tmp_path)
    assert float(run.input.frame_rate) == 25.0
    assert run.generation.inference_steps == 4
