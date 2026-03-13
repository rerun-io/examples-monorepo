from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import rerun as rr
from jaxtyping import Float, Int, UInt8
from numpy import ndarray
from simplecv.data.skeleton.mediapipe import MEDIAPIPE_ID2NAME, MEDIAPIPE_IDS, MEDIAPIPE_LINKS
from simplecv.rerun_log_utils import RerunTyroConfig, log_video
from simplecv.video_io import VideoReader
from tqdm.auto import tqdm

from wilor_nano.pipelines.wilor_hand_pose3d_estimation_pipeline import (
    Detection,
    WiLorHandPose3dEstimationPipeline,
    WilorPreds,
)
from wilor_nano.runtime import get_torch_device, get_torch_dtype


@dataclass
class WilorConfig:
    rr_config: RerunTyroConfig
    image_path: Path | None = None
    video_path: Path | None = None
    max_frames: int | None = None


def set_annotation_context() -> None:
    rr.log(
        "/",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label="Hand", color=(0, 0, 255)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name) for id, name in MEDIAPIPE_ID2NAME.items()
                    ],
                    keypoint_connections=MEDIAPIPE_LINKS,
                ),
            ]
        ),
        static=True,
    )


def main(config: WilorConfig):
    device = get_torch_device()
    dtype = get_torch_dtype(device)

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)
    set_annotation_context()

    # make sure one is not none
    assert config.image_path is not None or config.video_path is not None
    if config.image_path:
        bgr: UInt8[ndarray, "h w 3"] = cv2.imread(str(config.image_path))
        rgb: UInt8[ndarray, "h w 3"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rr.log("image", rr.Image(rgb, color_model=rr.ColorModel.RGB))
        outputs: list[Detection] = pipe.predict(rgb)
        for output in outputs:
            handedness: Literal["left", "right"] = "right" if output["is_right"] == 1.0 else "left"
            hand_bbox: list[float] = output["hand_bbox"]
            wilor_preds: WilorPreds = output["wilor_preds"]
            hand_keypoints: Float[ndarray, "1 n_joints=21 2"] = wilor_preds["pred_keypoints_2d"]
            xyz: Float[ndarray, "1 n_joints=21 3"] = wilor_preds["pred_keypoints_3d"]
            rr.log(
                f"{handedness}_xyz",
                rr.Points3D(
                    positions=xyz,
                    class_ids=0,
                    keypoint_ids=MEDIAPIPE_IDS,
                    show_labels=False,
                    colors=(0, 255, 0),
                ),
            )

            rr.log(
                f"image/{handedness}_box",
                rr.Boxes2D(array=hand_bbox, array_format=rr.Box2DFormat.XYXY, show_labels=True),
            )
            rr.log(
                f"image/{handedness}_keypoints",
                rr.Points2D(
                    positions=hand_keypoints,
                    class_ids=0,
                    keypoint_ids=MEDIAPIPE_IDS,
                    show_labels=False,
                    colors=(0, 255, 0),
                ),
            )

    if config.video_path:
        video_reader = VideoReader(filename=config.video_path)
        frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video(
            config.video_path, video_log_path=Path("video"), timeline="video_time"
        )
        for ts_idx, (ts, bgr) in enumerate(
            zip(
                tqdm(
                    frame_timestamps_ns,
                    desc="video frames",
                    total=len(frame_timestamps_ns) if config.max_frames is None else config.max_frames,
                ),
                video_reader,
                strict=False,
            )
        ):
            if ts_idx == config.max_frames and config.max_frames is not None:
                break
            rr.set_time("video_time", duration=1e-9 * ts)
            rgb: UInt8[ndarray, "h w 3"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            outputs: list[Detection] = pipe.predict(rgb)

            # check if left or right hand are in outputs if not clear them in rerun
            # After populating `outputs` and logging hands for this frame:
            has_right: bool = any(o.get("is_right", 0.0) == 1.0 for o in outputs)
            has_left: bool = any(o.get("is_right", 0.0) == 0.0 for o in outputs)

            # Clear right-hand logs if no right hand this frame
            if not has_right:
                rr.log("video/right_box", rr.Clear(recursive=True))
                rr.log("video/right_keypoints", rr.Clear(recursive=True))
                rr.log("right_xyz", rr.Clear(recursive=True))

            # Clear left-hand logs if no left hand this frame
            if not has_left:
                rr.log("video/left_box", rr.Clear(recursive=True))
                rr.log("video/left_keypoints", rr.Clear(recursive=True))
                rr.log("left_xyz", rr.Clear(recursive=True))

            for output in outputs:
                handedness: Literal["left", "right"] = "right" if output["is_right"] == 1.0 else "left"
                hand_bbox: list[float] = output["hand_bbox"]
                wilor_preds: WilorPreds = output["wilor_preds"]
                hand_keypoints: Float[ndarray, "1 n_joints=21 2"] = wilor_preds["pred_keypoints_2d"]
                xyz: Float[ndarray, "1 n_joints=21 3"] = wilor_preds["pred_keypoints_3d"]
                rr.log(
                    f"{handedness}_xyz",
                    rr.Points3D(
                        positions=xyz,
                        class_ids=0,
                        keypoint_ids=MEDIAPIPE_IDS,
                        show_labels=False,
                        colors=(0, 255, 0),
                    ),
                )

                rr.log(
                    f"video/{handedness}_box",
                    rr.Boxes2D(array=hand_bbox, array_format=rr.Box2DFormat.XYXY, show_labels=True),
                )
                rr.log(
                    f"video/{handedness}_keypoints",
                    rr.Points2D(
                        positions=hand_keypoints,
                        class_ids=0,
                        keypoint_ids=MEDIAPIPE_IDS,
                        show_labels=False,
                        colors=(0, 255, 0),
                    ),
                )
