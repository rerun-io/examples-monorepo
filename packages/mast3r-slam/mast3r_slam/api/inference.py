import contextlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal

import lietorch
import rerun as rr
import rerun.blueprint as rrb
import torch
import torch.multiprocessing as mp
from jaxtyping import Float
from simplecv.rerun_log_utils import RerunTyroConfig
from torch import Tensor

import mast3r_slam.evaluate as eval
from mast3r_slam.backend_lifecycle import SlamBackend
from mast3r_slam.config import config, load_config
from mast3r_slam.dataloader import MonocularDataset, load_dataset
from mast3r_slam.frame import Frame, Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.nerfstudio_utils import save_kf_to_nerfstudio
from mast3r_slam.rerun_log_utils import RerunLogger, create_blueprints
from mast3r_slam.retrieval_database import RetrievalDatabase
from mast3r_slam.tracker import FrameTracker


def format_time(seconds: float) -> str:
    """Format a duration in seconds as mm:ss.

    Args:
        seconds: Number of seconds.

    Returns:
        A string in ``"mm:ss"`` format.
    """
    minutes: int = int(seconds // 60)
    seconds_rem: int = int(seconds % 60)
    return f"{minutes:02d}:{seconds_rem:02d}"


@dataclass
class InferenceConfig:
    """Configuration for a MASt3R-SLAM inference run."""

    rr_config: RerunTyroConfig
    """Rerun recording configuration (save path, application id, etc.)."""
    dataset: str = "data/normal-apt-tour.mp4"
    """Path to the input dataset or video file."""
    config: str = "config/base.yaml"
    """Path to the SLAM config YAML file."""
    save_as: str = "default"
    """Subdirectory name for saving results under ``logs/``."""
    no_viz: bool = False
    """If True, skip launching visualisation processes."""
    img_size: Literal[224, 512] = 512
    """Target image size for MASt3R encoder."""
    max_frames: int | None = None
    """Stop after processing this many frames (None = process all)."""
    ns_save_path: None | Path = None
    """Optional path to export keyframes in NerfStudio format."""


def mast3r_slam_inference(inf_config: InferenceConfig) -> None:
    """Run the full MASt3R-SLAM inference pipeline.

    Initialises the model, dataset, shared state, tracker and backend
    processes, then runs the tracking loop until all frames are consumed
    or ``max_frames`` is reached.  Results are saved to disk and optionally
    exported in NerfStudio format.

    Args:
        inf_config: Inference configuration dataclass.
    """
    with contextlib.suppress(RuntimeError):
        mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device: str = "cuda:0"

    ## rerun setup
    parent_log_path: Path = Path("/world")
    rr_logger: RerunLogger = RerunLogger(parent_log_path)
    # create a blueprint
    blueprint: rrb.Blueprint = create_blueprints(parent_log_path)
    rr.send_blueprint(blueprint)

    load_config(inf_config.config)
    print(inf_config.dataset)
    print(config)

    dataset: MonocularDataset = load_dataset(inf_config.dataset, img_size=inf_config.img_size)
    dataset.subsample(config["dataset"]["subsample"])

    h: int
    w: int
    h, w = dataset.get_img_shape()[0]

    model = load_mast3r(device=device)
    model.share_memory()

    has_calib: bool = dataset.has_calib()
    use_calib: bool = config["use_calib"]
    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K: Float[Tensor, "3 3"] | None = None
    if use_calib:
        assert dataset.camera_intrinsics is not None
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(device, dtype=torch.float32)

    # remove the trajectory from the previous run
    if dataset.save_results:
        save_dir: Path
        seq_name: str
        save_dir, seq_name = eval.prepare_savedir(inf_config, dataset)
        print(f"Saving results to {save_dir}")
        traj_file: Path = save_dir / f"{seq_name}.txt"
        recon_file: Path = save_dir / f"{seq_name}.pt"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()

    with SlamBackend(inf_config.config, model, h, w, K, device=device) as ctx:
        keyframes: SharedKeyframes = ctx.keyframes
        states: SharedStates = ctx.states
        tracker: FrameTracker = FrameTracker(model, keyframes, device)

        i: int = 0
        fps_timer: float = time.time()
        start_time: float = timer()

        while True:
            ctx.check_backend()
            rr.set_time("frame", sequence=i)
            mode: Mode = states.get_mode()

            n_frames: int = len(dataset) if inf_config.max_frames is None else min(inf_config.max_frames, len(dataset))
            if i == n_frames:
                states.set_mode(Mode.TERMINATED)
                break

            timestamp, img = dataset[i]

            # get frames last camera pose
            world_T_cam: lietorch.Sim3 = lietorch.Sim3.Identity(1, device=device) if i == 0 else states.get_frame().world_T_cam
            frame: Frame = create_frame(i, img, world_T_cam, img_size=dataset.img_size, device=device)

            add_new_kf: bool = False
            if mode == Mode.INIT:
                # Initialize via mono inference, and encoded features needed for database
                X_init: Float[Tensor, "hw 3"]
                C_init: Float[Tensor, "hw 1"]
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
                match_info: list
                try_reloc: bool
                add_new_kf, match_info, try_reloc = tracker.track(frame)
                if try_reloc:
                    states.set_mode(Mode.RELOC)
                states.set_frame(frame)

            elif mode == Mode.RELOC:
                X: Float[Tensor, "hw 3"]
                C: Float[Tensor, "hw 1"]
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
                raise RuntimeError(f"Invalid mode: {mode!r}")

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
                FPS: float = i / (time.time() - fps_timer)
                print(f"FPS: {FPS}")
            i += 1

        if dataset.save_results:
            save_dir, seq_name = eval.prepare_savedir(inf_config, dataset)
            eval.save_ATE(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
            eval.save_reconstruction(save_dir, f"{seq_name}.pt", dataset.timestamps, keyframes)
            eval.save_keyframes(save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes)

        if inf_config.ns_save_path is not None:
            pcd = save_kf_to_nerfstudio(
                ns_save_path=inf_config.ns_save_path,
                keyframes=keyframes,
            )
            rr.log(
                f"{parent_log_path}/final_pointcloud",
                rr.Points3D(positions=pcd.points, colors=pcd.colors),
            )

        print("done")
        print(f"Inference time: {format_time(timer() - start_time)}")
        print(f"Processed {len(keyframes)}")
        ctx.join()

    # SlamBackend.__exit__ handles: backend shutdown, manager shutdown, GPU cleanup
    if not inf_config.no_viz:
        print("All visualization processes terminated")


def relocalization(
    frame: Frame,
    keyframes: SharedKeyframes,
    factor_graph: FactorGraph,
    retrieval_database: RetrievalDatabase,
) -> bool:
    """Attempt relocalization of a frame against the keyframe database.

    Queries the retrieval database for similar keyframes, adds the frame
    to the graph, and runs global optimisation on success.

    Args:
        frame: The current lost frame.
        keyframes: Shared keyframe buffer.
        factor_graph: The global factor graph.
        retrieval_database: Image retrieval database.

    Returns:
        True if relocalization succeeded, False otherwise.
    """
    # we are adding and then removing from the keyframe, so we need to be careful.
    # The lock slows viz down but safer this way...
    with keyframes.lock:
        kf_idx: list[int] = []
        retrieval_inds: list[int] = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure: bool = False
        if kf_idx:
            keyframes.append(frame)
            n_kf: int = len(keyframes)
            kf_idx = list(kf_idx)  # convert to list
            frame_idx: list[int] = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.world_T_cam[n_kf - 1] = keyframes.world_T_cam[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure



def run_backend(config_path, model, states, keyframes, K) -> None:
    """Backend process: graph construction, retrieval, and global optimisation.

    Runs in a separate process. Continuously polls for new keyframes,
    adds matching factors, runs retrieval, and solves the global pose graph.

    Args:
        config_path: Path to the SLAM config YAML.
        model: Shared MASt3R model.
        states: Shared system state.
        keyframes: Shared keyframe buffer.
        K: Camera intrinsic matrix, or None for uncalibrated mode.
    """
    load_config(config_path)

    device: str = keyframes.device
    factor_graph: FactorGraph = FactorGraph(model, keyframes, K, device)
    retrieval_database: RetrievalDatabase = load_retriever(model)

    mode: Mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
        if mode == Mode.RELOC:
            frame: Frame = states.get_frame()
            success: bool = relocalization(frame, keyframes, factor_graph, retrieval_database)
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue
        idx: int = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        # Graph Construction
        kf_idx: list[int] = []
        # k to previous consecutive keyframes
        n_consec: int = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds: list[int] = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        lc_inds: set[int] = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            print("Database retrieval", idx, ": ", lc_inds)

        kf_idx_set: set[int] = set(kf_idx)
        kf_idx_set.discard(idx)
        kf_idx = list(kf_idx_set)
        frame_idx: list[int] = [idx] * len(kf_idx)
        if kf_idx:
            factor_graph.add_factors(kf_idx, frame_idx, config["local_opt"]["min_match_frac"])

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)
