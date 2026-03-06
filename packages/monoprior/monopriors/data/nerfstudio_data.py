from dataclasses import dataclass
from typing import List
import json
from pathlib import Path


@dataclass
class NerfStudioFrame:
    # Assuming basic structure for a frame; add or remove fields as necessary
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    w: int
    h: int
    file_path: str
    transform_matrix: List[List[float]]


@dataclass
class NerfStudioData:
    camera_model: str
    orientation_override: str
    frames: List[NerfStudioFrame]


def load_nerfstudio_from_json(json_path: Path) -> NerfStudioData:
    # load the meta.json file
    with open(json_path, "r") as f:
        data = json.load(f)

    frames_data = data.pop("frames", [])  # Use pop to remove 'scene_box' from 'data'
    frames = [NerfStudioFrame(**frame_data) for frame_data in frames_data]

    # Filter out any unexpected keys from the data dictionary
    expected_keys = {
        "camera_model",
        "orientation_override",
    }
    filtered_data = {k: v for k, v in data.items() if k in expected_keys}

    return NerfStudioData(frames=frames, **filtered_data)
