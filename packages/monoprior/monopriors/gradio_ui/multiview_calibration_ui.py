import uuid
from collections.abc import Generator
from pathlib import Path

import gradio as gr
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from gradio_rerun import Rerun
from jaxtyping import Float32, UInt8
from numpy import ndarray
from PIL import Image

from monopriors.apis.multiview_calibration import (
    MultiViewCalibrator,
    MultiViewCalibratorConfig,
    MVCalibResults,
    create_final_view,
    estimate_voxel_size,
)


# Whenever we need a recording, we construct a new recording stream.
# As long as the app and recording IDs remain the same, the data
# will be merged by the Viewer.
def get_recording(recording_id: uuid.UUID) -> rr.RecordingStream:
    return rr.RecordingStream(application_id="rerun_example_gradio", recording_id=recording_id)


def _prerocess():
    pass


def multiview_calibration_fn(recording_id, imgs):
    yield from _multiview_calibration_fn(recording_id, imgs)


def _multiview_calibration_fn(
    recording_id: uuid.UUID, img_files: str | list[str]
) -> Generator[bytes | None, None, None]:
    # Here we get a recording using the provided recording id.
    recording: rr.RecordingStream = get_recording(recording_id)
    stream: rr.BinaryStream = recording.binary_stream()
    parent_log_path: Path = Path("world")
    timeline: str = "video_time"

    if isinstance(img_files, str):
        img_paths = [Path(img_files)]
    elif isinstance(img_files, list):
        img_paths = [Path(f) for f in img_files]
    else:
        raise gr.Error("Invalid input for images. Please select image files.")

    if not img_paths:
        raise gr.Error("Please select at least one RGB image before running calibration.")

    img_paths.sort()

    rgb_list: list[UInt8[ndarray, "height width 3"]] = []
    for img_path in img_paths:
        with Image.open(img_path) as pil_image:
            rgb_array: UInt8[ndarray, "height width 3"] = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
        rgb_list.append(rgb_array)

    final_view: rrb.ContainerLike = create_final_view(
        parent_log_path=parent_log_path, num_images=len(rgb_list), show_videos=False
    )
    blueprint: rrb.Blueprint = rrb.Blueprint(final_view, collapse_panels=True)
    rr.send_blueprint(blueprint=blueprint, recording=recording)
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RFU, static=True, recording=recording)
    rr.set_time(timeline, duration=0, recording=recording)
    yield stream.read()

    mv_config: MultiViewCalibratorConfig = MultiViewCalibratorConfig(
        device="cuda",
        verbose=True,
    )
    mv_calibrator: MultiViewCalibrator = MultiViewCalibrator(
        parent_log_path=parent_log_path,
        config=mv_config,
    )
    mv_results: MVCalibResults = mv_calibrator(rgb_list=rgb_list, recording=recording)
    yield stream.read()

    pcd_points: Float32[ndarray, "num_points 3"] = np.asarray(mv_results.pcd.points, dtype=np.float32)
    voxel_size: float = estimate_voxel_size(pcd_points, target_points=500_000)
    pcd_ds = mv_results.pcd.voxel_down_sample(voxel_size)
    filtered_points: Float32[ndarray, "final_points 3"] = np.asarray(pcd_ds.points, dtype=np.float32)
    filtered_colors: Float32[ndarray, "final_points 3"] = np.asarray(pcd_ds.colors, dtype=np.float32)

    rr.log(
        f"{parent_log_path}/point_cloud",
        rr.Points3D(filtered_points, colors=filtered_colors),
        static=True,
        recording=recording,
    )
    yield stream.read()


with gr.Blocks() as mv_calibration_block:
    # We make a new recording id, and store it in a Gradio's session state.
    recording_id = gr.State(uuid.uuid4())
    with gr.Row():
        input_imgs = gr.File(
            label="Input Images",
            file_count="multiple",
            file_types=[".png", ".jpg", ".jpeg"],
        )
        with gr.Column():
            run_calibration_btn = gr.Button("Run Multi-view Calibration")

    with gr.Row():
        rr_viewer = Rerun(
            streaming=True,
            panel_states={
                "time": "collapsed",
                "blueprint": "collapsed",
                "selection": "collapsed",
            },
        )

    input_imgs.change(
        fn=lambda: uuid.uuid4(),
        inputs=[],
        outputs=[recording_id],
    )
    run_calibration_btn.click(
        # Using the `viewer` as an output allows us to stream data to it by yielding bytes from the callback.
        multiview_calibration_fn,
        inputs=[recording_id, input_imgs],
        outputs=[rr_viewer],
    )
