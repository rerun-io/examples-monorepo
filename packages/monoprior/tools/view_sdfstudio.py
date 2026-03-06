import numpy as np
import rerun as rr
from pathlib import Path
from argparse import ArgumentParser
import cv2

from jaxtyping import UInt8, Float32
from monopriors.data.sdfstudio_data import load_sdfstudio_from_json


def main(data_dir: Path):
    assert data_dir.is_dir(), f"{data_dir} is not a directory"
    assert data_dir.exists(), f"{data_dir} does not exist"
    meta_json_path = data_dir / "meta_data.json"
    assert meta_json_path.exists(), f"{meta_json_path} does not exist"

    data = load_sdfstudio_from_json(meta_json_path)

    parent_log_path = Path("world")

    for idx, frame in enumerate(data.frames):
        cam_path = parent_log_path / f"cam_{idx}"
        pinhole_path = cam_path / "pinhole"
        rgb_path = data_dir / frame.rgb_path
        depth_path = data_dir / frame.mono_depth_path
        normal_path = data_dir / frame.mono_normal_path

        rgb: UInt8[np.ndarray, "h w 3"] = cv2.cvtColor(
            cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB
        )
        depth: Float32[np.ndarray, "h w"] = np.load(depth_path)
        # between 0 and 1
        normal: Float32[np.ndarray, "3 h w"] = np.load(normal_path)
        # permute the normal to be in the correct format
        normal = np.transpose(normal, (1, 2, 0))
        # between -1 and 1
        normal = (
            normal * 2.0 - 1.0
        )  # omnidata output is normalized so we convert it back to normal here
        # change coordinate system
        normal[..., 2] *= -1

        world_T_cam_44 = np.array(frame.camtoworld)
        K_44 = np.array(frame.intrinsics)
        rr.log(f"{pinhole_path}/image", rr.Image(rgb))
        rr.log(f"{pinhole_path}/depth", rr.DepthImage(depth))
        rr.log(f"{pinhole_path}/normal", rr.Image(normal))
        break


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--sdfstudio-data-dir",
        type=Path,
        default="data/sdfstudio/sdfstudio/replica/scan5",
    )
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "sdfstudio-viewer")
    main(args.sdfstudio_data_dir)
    rr.script_teardown(args)
