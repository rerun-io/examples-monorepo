from dataclasses import dataclass
from typing import List, Optional
import json
from pathlib import Path


@dataclass
class SceneBox:
    aabb: List[List[float]]
    near: float
    far: float
    radius: float
    collider_type: str


@dataclass
class SDFStudioFrame:
    # Assuming basic structure for a frame; add or remove fields as necessary
    rgb_path: Optional[str] = None
    camtoworld: Optional[List[List[float]]] = None
    intrinsics: Optional[List[List[float]]] = None
    mono_depth_path: Optional[str] = None
    mono_normal_path: Optional[str] = None
    foreground_mask: Optional[str] = None
    sfm_sparse_points_view: Optional[str] = None


@dataclass
class SDFStudioData:
    camera_model: str
    height: int
    width: int
    has_mono_prior: bool
    pairs: Optional[str]
    worldtogt: List[List[float]]
    scene_box: SceneBox
    frames: List[SDFStudioFrame]


def load_sdfstudio_from_json(json_path: Path) -> SDFStudioData:
    # load the meta.json file
    with open(json_path, "r") as f:
        data = json.load(f)

    scene_box_data = data.pop(
        "scene_box", {}
    )  # Use pop to remove 'scene_box' from 'data'
    scene_box = SceneBox(**scene_box_data) if scene_box_data else None

    frames_data = data.pop("frames", [])  # Similarly, remove 'frames' from 'data'
    frames = (
        [SDFStudioFrame(**frame_data) for frame_data in frames_data]
        if frames_data
        else []
    )

    # Filter out any unexpected keys from the data dictionary
    expected_keys = {
        "camera_model",
        "height",
        "width",
        "has_mono_prior",
        "pairs",
        "worldtogt",
    }
    filtered_data = {k: v for k, v in data.items() if k in expected_keys}

    return SDFStudioData(scene_box=scene_box, frames=frames, **filtered_data)
