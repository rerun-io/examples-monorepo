import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SceneBox:
    aabb: list[list[float]]
    near: float
    far: float
    radius: float
    collider_type: str


@dataclass
class SDFStudioFrame:
    # Assuming basic structure for a frame; add or remove fields as necessary
    rgb_path: str | None = None
    camtoworld: list[list[float]] | None = None
    intrinsics: list[list[float]] | None = None
    mono_depth_path: str | None = None
    mono_normal_path: str | None = None
    foreground_mask: str | None = None
    sfm_sparse_points_view: str | None = None


@dataclass
class SDFStudioData:
    camera_model: str
    height: int
    width: int
    has_mono_prior: bool
    pairs: str | None
    worldtogt: list[list[float]]
    scene_box: SceneBox
    frames: list[SDFStudioFrame]


def load_sdfstudio_from_json(json_path: Path) -> SDFStudioData:
    # load the meta.json file
    with open(json_path) as f:
        data = json.load(f)

    scene_box_data = data.pop(
        "scene_box", {}
    )  # Use pop to remove 'scene_box' from 'data'
    scene_box = SceneBox(**scene_box_data) if scene_box_data else None
    if scene_box is None:
        raise ValueError(f"Missing scene_box in {json_path}")

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
