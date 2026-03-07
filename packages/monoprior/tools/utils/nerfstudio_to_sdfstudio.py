import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import calibur
import cv2
import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
from jaxtyping import Bool, Float64
from tqdm import tqdm

from monopriors.data.nerfstudio_data import load_nerfstudio_from_json
from monopriors.data.sdfstudio_data import SceneBox, SDFStudioData, SDFStudioFrame
from monopriors.monoprior_models import DsineAndUnidepth, OldMonoPriorPrediction


def main(
    input_dir: Path,
    output_dir: Path,
    scene_type: Literal["indoor", "object"],
    log=False,
):
    """
    given data that follows the nerfstduio format such as the output from colmap or polycam, convert to a format
    that sdfstudio will ingest
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # load transformation json with images/intrinsics/extrinsics
    camera_parameters_path = input_dir / "transforms.json"
    ns_data = load_nerfstudio_from_json(camera_parameters_path)

    # ns_log_path = Path("ns")

    world_T_cam_cv_44_list = []
    image_paths = []
    K_33_list = []
    for _, frame in enumerate(ns_data.frames):
        image_path = input_dir / frame.file_path
        bgr = cv2.imread(str(image_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        world_T_cam_gl_44 = np.array(frame.transform_matrix).reshape(4, 4)
        K_33 = np.array([[frame.fl_x, 0, frame.cx], [0, frame.fl_y, frame.cy], [0, 0, 1]])
        K_33_list.append(K_33)

        # if log and idx % 10 == 0:
        #     cam_log_path = ns_log_path / f"cam_{idx:03d}"
        #     pinhole_log_path = cam_log_path / "pinhole"
        #     rr.log(
        #         str(cam_log_path),
        #         rr.Transform3D(
        #             translation=world_T_cam_gl_44[:3, 3],
        #             mat3x3=world_T_cam_gl_44[:3, :3],
        #             from_parent=False,
        #         ),
        #     )
        #     rr.log(
        #         str(pinhole_log_path),
        #         rr.Pinhole(
        #             image_from_camera=K_33,
        #             width=frame.w,
        #             height=frame.h,
        #             camera_xyz=rr.ViewCoordinates.RUB,
        #         ),
        #     )
        #     rr.log(
        #         f"{pinhole_log_path}/image",
        #         rr.Image(rgb),
        #     )

        height, width, _ = rgb.shape

        world_T_cam_cv_44 = calibur.convert_pose(
            world_T_cam_gl_44,
            src_convention=calibur.CC.GL,
            dst_convention=calibur.CC.CV,
        )
        world_T_cam_cv_44_list.append(world_T_cam_cv_44)
        image_paths.append(image_path)
        assert image_path.exists()

    world_T_cam_cv_b44 = np.array(world_T_cam_cv_44_list)
    K_b33 = np.array(K_33_list)

    valid_poses: Bool[np.ndarray, ""] = np.isfinite(world_T_cam_cv_b44).all(axis=2).all(axis=1)
    min_vertices: Float64[np.ndarray, "3"] = world_T_cam_cv_b44[:, :3, 3][valid_poses].min(axis=0)
    max_vertices: Float64[np.ndarray, "3"] = world_T_cam_cv_b44[:, :3, 3][valid_poses].max(axis=0)

    # Perform Camera normalization
    if scene_type == "indoor":
        scene_center = (min_vertices + max_vertices) / 2.0
        scene_scale = 2.0 / (np.max(max_vertices - min_vertices) + 3.0)

        # normalize pose to unit cube
        world_T_cam_cv_b44[:, :3, 3] -= scene_center
        world_T_cam_cv_b44[:, :3, 3] *= scene_scale

        # inverse normalization to recover original poses
        world_T_gt = np.eye(4).astype(np.float32)
        world_T_gt[:3, 3] -= scene_center
        world_T_gt[:3] *= scene_scale
        gt_T_world = np.linalg.inv(world_T_gt)
    else:
        gt_T_world = np.eye(4).astype(np.float32)

    # Generate SDFStudio data json and save related images/monopriors
    model = DsineAndUnidepth()

    frames = []
    sdf_log_path = Path("sdf")
    for idx, (valid, world_T_cam_cv_44, image_path) in enumerate(tqdm(zip(valid_poses, world_T_cam_cv_b44, image_paths, strict=True))):
        if not valid:
            continue

        # save rgb image
        target_image = output_dir / f"{idx:06d}_rgb.png"
        bgr = cv2.imread(str(image_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(target_image), rgb)
        # img = Image.open(image_path)
        # img.save(target_image)

        K_33 = K_b33[idx]
        pred: OldMonoPriorPrediction = model(rgb, K_33.astype(np.float32))
        depth_np_bhw1, normal_np_bhw3, _ = pred.to_numpy()

        # convert to normalized like omnidata (this is what nerfstudio accepts)
        normal_np_bhw3[..., 2] *= -1
        normal_np_bhw3 = (normal_np_bhw3 + 1.0) / 2.0

        rgb_path = str(target_image.relative_to(output_dir))

        # save depth and normal using numpy
        mono_depth_path = rgb_path.replace("_rgb.png", "_depth.npy")
        mono_normal_path = rgb_path.replace("_rgb.png", "_normal.npy")

        # convert hw1 depth to hw3 using grayscale to rgb
        depth_np_hw1 = depth_np_bhw1[0]
        # normalize depth to 0-1
        depth_np_hw1 = (depth_np_hw1 - depth_np_hw1.min()) / (depth_np_hw1.max() - depth_np_hw1.min())
        depth_np_hw = depth_np_hw1.squeeze()
        np.save(output_dir / mono_depth_path, depth_np_hw)
        np.save(output_dir / mono_normal_path, normal_np_bhw3[0])
        # save as images
        plt.imsave(
            output_dir / mono_depth_path.replace(".npy", ".png"),
            depth_np_hw,
            cmap="viridis",
        )
        plt.imsave(output_dir / mono_normal_path.replace(".npy", ".png"), normal_np_bhw3[0])

        frame = SDFStudioFrame(
            rgb_path=rgb_path,
            camtoworld=world_T_cam_cv_44.tolist(),
            intrinsics=K_33.tolist(),
            mono_depth_path=mono_depth_path,
            mono_normal_path=mono_normal_path,
        )

        frames.append(frame)
        rr.set_time("timestep", sequence=idx)
        if log:
            cam_log_path = sdf_log_path / "cam"
            pinhole_log_path = cam_log_path / "pinhole"
            rr.log(
                str(cam_log_path),
                rr.Transform3D(
                    translation=world_T_cam_cv_44[:3, 3],
                    mat3x3=world_T_cam_cv_44[:3, :3],
                    from_parent=False,
                ),
            )
            rr.log(
                str(pinhole_log_path),
                rr.Pinhole(
                    image_from_camera=K_b33[idx],
                    width=width,
                    height=height,
                    camera_xyz=rr.ViewCoordinates.RDF,
                ),
            )
            rr.log(
                f"{pinhole_log_path}/image",
                rr.Image(rgb),
            )
            rr.log(
                f"{pinhole_log_path}/depth",
                rr.DepthImage(depth_np_bhw1[0]),
            )
            rr.log(
                f"{pinhole_log_path}/normal",
                rr.Image(normal_np_bhw3[0]),
            )

    # scene bbox for the scannet scene
    if scene_type == "indoor":
        scene_box = SceneBox(
            aabb=[[-1, -1, -1], [1, 1, 1]],
            near=0.05,
            far=2.5,
            radius=1.0,
            collider_type="box",
        )
    else:
        raise NotImplementedError

    # meta data
    sdf_data = SDFStudioData(
        camera_model="OPENCV",
        height=height,
        width=width,
        has_mono_prior=True,
        worldtogt=gt_T_world.tolist(),
        scene_box=scene_box,
        frames=frames,
        pairs=None,
    )

    sdf_dict = asdict(sdf_data)

    # save as json
    with open(output_dir / "meta_data.json", "w", encoding="utf-8") as f:
        json.dump(sdf_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess scannet dataset to sdfstudio dataset")

    parser.add_argument("--input-dir", help="path to polycam data directory", required=True, type=Path)
    parser.add_argument("--output-dir", help="path to output directory", required=True, type=Path)
    parser.add_argument(
        "--scene-type",
        dest="scene_type",
        required=True,
        choices=["indoor", "object"],
    )
    parser.add_argument("--log", help="log to rerun", action="store_true", default=False)
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "polycam to sdfstudio")
    main(args.input_dir, args.output_dir, args.scene_type, log=args.log)
    rr.script_teardown(args)
