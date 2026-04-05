from dataclasses import dataclass, fields
import gradio as gr

import time

from gradio_rerun import Rerun

import rerun as rr

import beartype
from pathlib import Path
import subprocess

from mast3r_slam.config import load_config, config
from mast3r_slam.dataloader import load_dataset

import sys
import lietorch
import torch
from mast3r_slam.frame import Frame, Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    mast3r_inference_mono,
)
from mast3r_slam.tracker import FrameTracker
import torch.multiprocessing as mp
from mast3r_slam.api.inference import (
    run_backend,
)
from multiprocessing.managers import SyncManager
import gc
import shutil
from mast3r_slam.nerfstudio_utils import save_kf_to_nerfstudio
from mast3r_slam.rerun_log_utils import create_blueprints, RerunLogger

# Global variables to track the backend process, states, and model
active_backend_process = None
active_states = None

# Initialize multiprocessing start method only once
try:
    mp.set_start_method("spawn")
    DEVICE = "cuda:0"
    model = load_mast3r(device=DEVICE)
    model.share_memory()
except RuntimeError:
    # Method already set, ignore
    pass


def stop_streaming():
    global active_backend_process, active_states

    # If there's an active process and states
    if active_backend_process is not None and active_states is not None:
        # Set termination mode
        active_states.set_mode(Mode.TERMINATED)

        # Join the process with timeout to prevent hanging
        active_backend_process.join(timeout=1)

        # If process is still alive after timeout, terminate it
        if active_backend_process.is_alive():
            active_backend_process.terminate()
            active_backend_process.join()

        print("Backend process stopped")

        # Force CUDA memory cleanup
        torch.cuda.empty_cache()
        gc.collect()

        print("CUDA memory cleared")

        # Reset globals
        active_backend_process = None
        active_states = None

    return None


@rr.thread_local_stream("rerun_example_streaming_blur")
def streaming_mast3r_slam_fn(*input_params, progress=gr.Progress()):
    global active_backend_process, active_states
    stream = rr.binary_stream()

    # Clean up any leftover resources from previous runs
    stop_streaming()

    try:
        parameters = InputValues(*input_params)
        video_path = Path(parameters.video_file)
    except beartype.roar.BeartypeCallHintParamViolation as e:
        raise gr.Error(  # noqa: B904
            "Did you make sure the zipfile finished uploading?. Try to hit run again.",
            duration=20,
        )
    except Exception as e:
        raise gr.Error(  # noqa: B904
            f"Error: {e}\n Did you wait for zip file to upload?", duration=20
        )

    # rr.set_time_sequence("iteration", 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    ## rerun setup
    parent_log_path = Path("world")
    rr_logger = RerunLogger(parent_log_path)
    blueprint = create_blueprints(parent_log_path=parent_log_path)
    rr.send_blueprint(blueprint)

    progress(0.05, desc="Loading config")
    inference_config = "config/base.yaml"
    load_config(path=inference_config)

    manager: SyncManager = mp.Manager()

    progress(0.1, desc="Loading dataset")
    dataset = load_dataset(dataset_path=str(video_path))
    dataset.subsample(config["dataset"]["subsample"])

    h, w = dataset.get_img_shape()[0]
    keyframes = SharedKeyframes(manager, h, w)
    states = SharedStates(manager, h, w)

    # Store the states in global variable for access from stop function
    active_states = states

    has_calib: bool = dataset.has_calib()
    use_calib: bool = config["use_calib"]
    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            DEVICE, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)

    progress(0.15, desc="Starting Backend")
    tracker = FrameTracker(model, keyframes, DEVICE)

    backend = mp.Process(
        target=run_backend, args=(inference_config, model, states, keyframes, K)
    )
    backend.start()

    # Store the process in global variable for access from stop function
    active_backend_process = backend

    i = 0
    fps_timer: float = time.time()

    while True:
        rr.set_time("frame", sequence=i)
        mode: Mode = states.get_mode()

        if i == len(dataset):
            states.set_mode(Mode.TERMINATED)
            break

        timestamp, img = dataset[i]

        # get frames last camera pose
        T_WC: lietorch.Sim3 = (
            lietorch.Sim3.Identity(1, device=DEVICE)
            if i == 0
            else states.get_frame().T_WC
        )
        frame: Frame = create_frame(
            i, img, T_WC, img_size=dataset.img_size, device=DEVICE
        )

        if mode == Mode.INIT:
            # Initialize via mono inference, and encoded features needed for database
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)
            rr_logger.log_frame(frame, keyframes, states)
            i += 1
            continue

        if mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)

        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
            # In single threaded mode, make sure relocalization happen for every frame
            while config["single_thread"]:
                with states.lock:
                    if states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)

        else:
            raise Exception("Invalid mode")

        if add_new_kf:
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            # In single threaded mode, wait for the backend to finish
            while config["single_thread"]:
                with states.lock:
                    if len(states.global_optimizer_tasks) == 0:
                        break
                time.sleep(0.01)

        ## rerun log stuff
        rr_logger.log_frame(frame, keyframes, states)
        # log time
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        i += 1

        yield stream.read(), None

    pcd = save_kf_to_nerfstudio(
        ns_save_path=video_path.parent / "nerfstudio-output",
        keyframes=keyframes,
    )

    rr.log(
        f"{parent_log_path}/final_pointcloud",
        rr.Points3D(positions=pcd.points, colors=pcd.colors),
    )

    # Zip the nerfstudio output
    ns_output_dir = video_path.parent / "nerfstudio-output"
    zip_output_path = video_path.parent / "nerfstudio-output.zip"

    try:
        if ns_output_dir.exists():
            print(f"Zipping nerfstudio output to {zip_output_path}")
            shutil.make_archive(
                base_name=str(zip_output_path.with_suffix("")),
                format="zip",
                root_dir=video_path.parent,
                base_dir="nerfstudio-output",
            )

    except Exception as e:
        raise gr.Error(f"Failed to zip nerfstudio output: {e}")

    assert zip_output_path.exists(), f"Zip file {zip_output_path} does not exist"

    print("Finished processing")
    yield stream.read(), str(zip_output_path)
    backend.join()
    # Clean up resources before completing
    stop_streaming()


def mov_to_mp4(video_path: Path) -> Path:
    mp4_path = video_path.with_suffix(".mp4")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(video_path),
                "-c:v",
                "copy",
                "-an",
                "-y",
                str(mp4_path),
            ],
            check=True,
            capture_output=True,
        )
        video_path = mp4_path
    except subprocess.CalledProcessError as e:
        raise gr.Error(f"Failed to convert MOV to MP4: {e.stderr.decode()}")
    return video_path


@rr.thread_local_stream("rr_show_video")
def show_video_file(
    *input_params,
):
    stream = rr.binary_stream()

    try:
        parameters = InputValues(*input_params)
        video_path: Path = Path(parameters.video_file)
    except beartype.roar.BeartypeCallHintParamViolation as e:
        raise gr.Error(  # noqa: B904
            "Did you make sure the zipfile finished uploading?. Try to hit run again.",
            duration=20,
        )
    except Exception as e:
        raise gr.Error(  # noqa: B904
            f"Error: {e}\n Did you wait for zip file to upload?", duration=20
        )

    # check if file is mov, if so convert to mp4
    if video_path.suffix.lower() == ".mov":
        video_path = mov_to_mp4(video_path)

    # Log video asset which is referred to by frame references.
    video_asset = rr.AssetVideo(path=video_path)
    rr.log("video", video_asset, static=True)

    # Send automatically determined video frame timestamps.
    frame_timestamps_ns = video_asset.read_frame_timestamps_ns()
    rr.send_columns(
        "video",
        # Note timeline values don't have to be the same as the video timestamps.
        indexes=[rr.TimeNanosColumn("video_time", frame_timestamps_ns)],
        columns=rr.VideoFrameReference.columns_nanoseconds(frame_timestamps_ns),
    )
    yield stream.read()


@dataclass
class InputComponents:
    video_file: gr.File

    def to_list(self) -> list:
        return [getattr(self, f.name) for f in fields(self)]


@dataclass
class InputValues:
    video_file: str


with gr.Blocks() as mast3r_slam_block:
    with gr.Row():
        with gr.Column():
            video_file = gr.File(label="Upload Video", file_types=[".mp4", ".mov", ".MOV"])

        with gr.Column():
            with gr.Row():
                blur_btn = gr.Button("Run Mast3r Slam")
                stop_blur_btn = gr.Button("Stop Mast3r Slam")
            output_zip_file = gr.File(
                label="Download NerfStudio Output", file_count="single"
            )

    with gr.Row():
        viewer = Rerun(
            streaming=True,
            panel_states={
                "time": "collapsed",
                "blueprint": "hidden",
                "selection": "hidden",
            },
        )

    # this is to allow for kwargs as well as type hint validation of inputs with beartype
    input_params = InputComponents(
        video_file=video_file,
    )

    video_file.upload(
        fn=show_video_file, inputs=input_params.to_list(), outputs=[viewer]
    )

    examples = gr.Examples(
        examples=[
            ["data/normal-apt-tour.MOV"],
        ],
        inputs=input_params.to_list(),
        outputs=[viewer, output_zip_file],
    )
    blur_event = blur_btn.click(
        streaming_mast3r_slam_fn,
        inputs=input_params.to_list(),
        outputs=[viewer, output_zip_file],
    )
    stop_blur_btn.click(fn=stop_streaming, inputs=[], outputs=[], cancels=[blur_event])
