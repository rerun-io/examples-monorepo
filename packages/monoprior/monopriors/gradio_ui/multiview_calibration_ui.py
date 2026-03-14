import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Final

import gradio as gr
import numpy as np
import rerun as rr
from gradio_rerun import Rerun
from jaxtyping import UInt8
from numpy import ndarray
from PIL import Image

from monopriors.apis.multiview_calibration import (
    PARENT_LOG_PATH,
    TIMELINE,
    MultiViewCalibrator,
    MultiViewCalibratorConfig,
    run_calibration_pipeline,
)

EXAMPLE_DATA_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "data" / "examples" / "multiview"

gr.set_static_paths([str(EXAMPLE_DATA_DIR)])

_MV_CONFIG: MultiViewCalibratorConfig = MultiViewCalibratorConfig(
    device="cuda",
    verbose=True,
)
_MV_CALIBRATOR: MultiViewCalibrator = MultiViewCalibrator(
    parent_log_path=PARENT_LOG_PATH,
    config=_MV_CONFIG,
)


# Whenever we need a recording, we construct a new recording stream.
# As long as the app and recording IDs remain the same, the data
# will be merged by the Viewer.
def get_recording(recording_id: uuid.UUID) -> rr.RecordingStream:
    return rr.RecordingStream(application_id="rerun_example_gradio", recording_id=recording_id)


def multiview_calibration_fn(
    recording_id: uuid.UUID, img_files: str | list[str]
) -> Generator[bytes | None, None, None]:
    recording: rr.RecordingStream = get_recording(recording_id)
    stream: rr.BinaryStream = recording.binary_stream()

    if isinstance(img_files, str):
        img_paths: list[Path] = [Path(img_files)]
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

    with recording:
        run_calibration_pipeline(
            rgb_list=rgb_list,
            mv_calibrator=_MV_CALIBRATOR,
            parent_log_path=PARENT_LOG_PATH,
            timeline=TIMELINE,
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

    car_example_images: list[str] = sorted(str(p) for p in (EXAMPLE_DATA_DIR / "car_landscape_12").glob("*.jpg"))
    rp_capture_images: list[str] = sorted(str(p) for p in (EXAMPLE_DATA_DIR / "rp_capture_6").glob("*.jpg"))
    gr.Examples(
        examples=[
            [car_example_images],
            [rp_capture_images],
        ],
        inputs=[input_imgs],
        cache_examples=False,
    )

    input_imgs.change(
        fn=lambda: uuid.uuid4(),
        inputs=[],
        outputs=[recording_id],
    )
    run_calibration_btn.click(
        multiview_calibration_fn,
        inputs=[recording_id, input_imgs],
        outputs=[rr_viewer],
    )
