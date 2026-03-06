import zipfile
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import rerun as rr
from PIL import Image
from tqdm import tqdm

from monopriors.surface_normal_models import DSineNormalPredictor, StableNormalPredictor
from monopriors.surface_normal_models.base_normal_model import SurfaceNormalPrediction


def log_compare_normals(
    parent_path,
    rgb,
    normal_np_1,
    normal_np_2,
):
    cam_path = parent_path / "cam"
    pinhole_path = cam_path / "pinhole"

    h, w, _ = rgb.shape
    # move from 0 -1 to -1 1
    normal_np_1 = (normal_np_1 - 0.5) * 2
    # # convert from h,w,c to hw, c
    # normal_np_1 = normal_np_1.reshape(-1, 3)
    # # convert from OpenCV coordinate system (RDF) to OpenGL coordinate system (RUB)
    # # convert normal map from opengl to opencv
    # rotation = np.diag([1, -1, -1])
    # normal_np_1 = normal_np_1 @ rotation

    normal_np_1 = normal_np_1.reshape(h, w, 3)

    # convert normal map 2 from LUB to RUB
    normal_np_2 = normal_np_2.reshape(-1, 3)
    rotation = np.diag([-1, 1, 1])
    normal_np_2 = normal_np_2 @ rotation
    normal_np_2 = normal_np_2.reshape(h, w, 3)

    # convert to Uint8
    normal_np_1 = ((normal_np_1 + 1) / 2 * 255).astype(np.uint8)
    normal_np_2 = ((normal_np_2 + 1) / 2 * 255).astype(np.uint8)

    rr.log(f"{pinhole_path}/image", rr.Image(rgb).compress(jpeg_quality=90))
    rr.log(f"{pinhole_path}/normal_1", rr.Image(normal_np_1).compress(jpeg_quality=90))
    rr.log(f"{pinhole_path}/normal_2", rr.Image(normal_np_2).compress(jpeg_quality=90))


def extract_zip(zip_path: Path, extract_dir: Path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


def main(zip_path: Path):
    extract_dir = zip_path.parent / zip_path.stem

    if zip_path.is_file() and zip_path.exists() and (not extract_dir.exists()):
        extract_zip(zip_path, extract_dir=extract_dir)
        final_extract_dir = extract_dir / extract_dir.name
    else:
        final_extract_dir = extract_dir / extract_dir.name

    image_dir = final_extract_dir / "corrected_images"

    assert image_dir.exists() and image_dir.is_dir(), f"Image directory not found: {image_dir}"

    image_paths = sorted(image_dir.glob("*.jpg"))

    stablenm_model = StableNormalPredictor(device="cuda")
    dsine_model = DSineNormalPredictor(device="cuda")

    parent_path = Path("world")

    for idx, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
        rr.set_time_sequence("timestep", idx)
        rgb = np.array(Image.open(image_path))

        # omni_normal_pred: SurfaceNormalPrediction = omni_model(rgb, None)
        sn_normal_pred: SurfaceNormalPrediction = stablenm_model(rgb, None)
        dsine_normal_pred: SurfaceNormalPrediction = dsine_model(rgb, None)

        # convert to numpy
        sn_normal_hw3 = sn_normal_pred.normal_hw3
        dsine_normal_hw3 = dsine_normal_pred.normal_hw3

        parent_path = Path("world")
        log_compare_normals(
            parent_path=parent_path,
            rgb=rgb,
            normal_np_1=sn_normal_hw3,
            normal_np_2=dsine_normal_hw3,
        )
        if idx > 10:
            break


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--zip-path",
        type=Path,
        required=True,
        help="Path to the image for inference",
    )
    rr.script_add_args(parser)

    args = parser.parse_args()
    rr.script_setup(args, "monopriors inference")
    main(args.zip_path)
    rr.script_teardown(args)
