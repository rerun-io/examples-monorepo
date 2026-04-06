import gc
import shutil
import subprocess
import sys
import time
from multiprocessing.managers import SyncManager
from pathlib import Path

import beartype
import gradio as gr
import lietorch
import rerun as rr
import torch
import torch.multiprocessing as mp
from gradio_rerun import Rerun

from mast3r_slam.api.inference import (
    run_backend,
)
from mast3r_slam.config import config, load_config
from mast3r_slam.dataloader import load_dataset
from mast3r_slam.frame import Frame, Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    mast3r_inference_mono,
)
from mast3r_slam.nerfstudio_utils import save_kf_to_nerfstudio
from mast3r_slam.rerun_log_utils import RerunLogger, create_blueprints
from mast3r_slam.tracker import FrameTracker

# Global variables to track the backend process, states, and model
active_backend_process = None
active_states = None

# Initialize multiprocessing start method (spawn required for CUDA).
# If already set (e.g. by a parent process), reuse the existing setting.
try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass  # Method already set, safe to continue

DEVICE: str = "cuda:0"
model = load_mast3r(device=DEVICE)
model.share_memory()


def _cleanup_active_backend() -> None:
    global active_backend_process, active_states

    if active_backend_process is not None:
        active_backend_process.join(timeout=1)
        if active_backend_process.is_alive():
            active_backend_process.terminate()
            active_backend_process.join()

        print("Backend process stopped")

    torch.cuda.empty_cache()
    gc.collect()
    print("CUDA memory cleared")

    active_backend_process = None
    active_states = None


def stop_streaming():
    global active_backend_process, active_states

    # If there's an active process and states
    if active_backend_process is not None and active_states is not None:
        # Set termination mode
        active_states.set_mode(Mode.TERMINATED)

        _cleanup_active_backend()

    return None


@rr.thread_local_stream("rerun_example_streaming_blur")
def streaming_mast3r_slam_fn(
    video_file: str,
    subsample: int,
    progress=gr.Progress(),
):
    global active_backend_process, active_states
    stream = rr.binary_stream()

    # Clean up any leftover resources from previous runs
    stop_streaming()

    try:
        video_path = Path(video_file)
    except beartype.roar.BeartypeCallHintParamViolation:
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
    config["dataset"]["subsample"] = subsample

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
    stopped_early = False

    while True:
        rr.set_time("frame", sequence=i)
        mode: Mode = states.get_mode()

        if mode == Mode.TERMINATED:
            stopped_early = i < len(dataset)
            break

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
            raise RuntimeError(f"Unexpected MASt3R-SLAM mode: {mode}")

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

        yield stream.read(), None, f"Processing frame {i}/{len(dataset)}"

    if stopped_early:
        _cleanup_active_backend()
        yield stream.read(), None, "MASt3R-SLAM stopped"
        return

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
    yield stream.read(), str(zip_output_path), "MASt3R-SLAM complete"
    _cleanup_active_backend()


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
    video_file: str,
):
    stream = rr.binary_stream()

    try:
        video_path = Path(video_file)
    except beartype.roar.BeartypeCallHintParamViolation:
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
    yield stream.read(), f"Loaded preview for {video_path.name}"


def _switch_to_outputs():
    return gr.update(selected="outputs")


def _switch_to_inputs():
    return gr.update(selected="inputs")


def _reset_run_outputs():
    return None, "Starting MASt3R-SLAM..."


with gr.Blocks() as mast3r_slam_block:
    with gr.Row():
        with gr.Column(scale=1):
            tabs = gr.Tabs(selected="inputs")
            with tabs:
                with gr.TabItem("Inputs", id="inputs"):
                    gr.Markdown(
                        "Upload a video, preview it in the Rerun viewer, then run the full "
                        "MASt3R-SLAM pipeline."
                    )
                    video_file = gr.File(
                        label="Input Video",
                        file_types=[".mp4", ".mov", ".MOV"],
                    )
                    run_slam_btn = gr.Button("Run MASt3R-SLAM", variant="primary")

                    with gr.Accordion("Config", open=False):
                        subsample_slider = gr.Slider(
                            label="Frame Subsample",
                            minimum=1,
                            maximum=8,
                            step=1,
                            value=1,
                        )
                        gr.Markdown(
                            "Base config file: `config/base.yaml`\n\n"
                            "Advanced calibration and backend settings remain in the YAML "
                            "for now so the UI does not expose partially applied controls."
                        )

                with gr.TabItem("Outputs", id="outputs"):
                    status_text = gr.Textbox(
                        label="Status",
                        value="Upload a video to begin.",
                        interactive=False,
                    )
                    stop_slam_btn = gr.Button("Stop MASt3R-SLAM")
                    output_zip_file = gr.File(
                        label="Download Nerfstudio Output",
                        file_count="single",
                    )

                examples = gr.Examples(
                    examples=[
                    ["data/normal-apt-tour.mp4"],
                ],
                inputs=[video_file],
                cache_examples=False,
            )

        with gr.Column(scale=5):
            viewer = Rerun(
                streaming=True,
                panel_states={
                    "time": "collapsed",
                    "blueprint": "hidden",
                    "selection": "hidden",
                },
                height=800,
            )

    video_file.upload(
        fn=_switch_to_inputs,
        inputs=None,
        outputs=[tabs],
        api_visibility="private",
    ).then(
        fn=show_video_file,
        inputs=[video_file],
        outputs=[viewer, status_text],
    )

    load_input_event = getattr(examples, "load_input_event", None)
    if load_input_event is not None:
        load_input_event.then(
            fn=_switch_to_inputs,
            inputs=None,
            outputs=[tabs],
            api_visibility="private",
        )

    run_event = run_slam_btn.click(
        fn=_switch_to_outputs,
        inputs=None,
        outputs=[tabs],
        api_visibility="private",
    ).then(
        fn=_reset_run_outputs,
        inputs=None,
        outputs=[output_zip_file, status_text],
        api_visibility="private",
    ).then(
        fn=streaming_mast3r_slam_fn,
        inputs=[
            video_file,
            subsample_slider,
        ],
        outputs=[viewer, output_zip_file, status_text],
    )

    stop_slam_btn.click(
        fn=stop_streaming,
        inputs=[],
        outputs=[],
        cancels=[run_event],
    ).then(
        fn=lambda: "MASt3R-SLAM stopped",
        inputs=None,
        outputs=[status_text],
        api_visibility="private",
    )
