"""Side-by-side PromptDA teacher vs ZipDepth-PromptDA student on a Polycam capture.

Runs both depth models over the same frames with the same LiDAR prompt and logs
RGB, both predictions, and their absolute difference to Rerun for visual
inspection. The student is the distilled 6.14 M-parameter model; the teacher is
the 340 M-parameter PromptDA-large TensorRT engine that produced its training
labels.

Preprocessing mirrors the training pipeline exactly: RGB is resized with
``cv2.INTER_LINEAR`` and prompt pixels failing the confidence or range test are
zeroed before the student sees them.
"""

import time
from dataclasses import dataclass, field
from itertools import batched, chain
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Bool, Float32, UInt8, UInt16
from monopriors.models.depth_completion.zipdepth_prompt import ZipDepthPrompt, load_zipdepth_prompt
from numpy import ndarray
from simplecv.data.polycam import PolycamData, PolycamDataset, load_polycam_data
from simplecv.rerun_log_utils import RerunTyroConfig
from torch import Tensor
from tqdm import tqdm

from rerun_prompt_da.apis.prompt_da_trt_polycam import network_image_hw
from rerun_prompt_da.trt_predictor import PromptDATrtPredictor

PROMPT_MIN_DEPTH_MM: int = 100
"""Shortest prompt depth the student was trained to trust, in millimetres."""
PROMPT_MAX_DEPTH_MM: int = 4000
"""Furthest prompt depth the student was trained to trust, in millimetres."""


@dataclass
class PolycamCompareConfig:
    """Runtime configuration for the teacher-versus-student Polycam comparison."""

    polycam_zip_path: Path
    """Polycam capture zip (or extracted directory) to process."""
    zipdepth_checkpoint: Path
    """Trained ZipDepth-PromptDA checkpoint to evaluate against the teacher."""
    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Rerun viewer/save/connect behavior."""
    batch_size: int = 8
    """Frames per model batch."""
    max_image_size: int = 1008
    """Longest teacher network image side; 14-aligned from the capture resolution."""
    student_height: int = 768
    """Student network input height; 768 matches the training resolution."""
    student_width: int = 1024
    """Student network input width; 1024 matches the training resolution."""
    max_frames: int | None = None
    """Optional cap on processed frames, for a quick visual check."""


def masked_prompt_metres(polycam_data: PolycamData) -> Float32[ndarray, "192 256"]:
    """Return the LiDAR prompt in metres with untrusted pixels zeroed.

    Mirrors ``zipdepth.catalog.targets._prompt_tensors`` so the student sees the
    same prompt distribution it was trained on: confidence above LOW, depth
    inside 0.1--4.0 m, everything else zero.

    Args:
        polycam_data: One decoded Polycam frame.

    Returns:
        Float32 prompt depth in metres with shape ``(192, 256)``.
    """
    prompt_depth_mm_hw: UInt16[ndarray, "192 256"] = polycam_data.original_depth_hw
    prompt_confidence_hw: UInt8[ndarray, "192 256"] = polycam_data.original_confidence_hw
    prompt_valid_hw: Bool[ndarray, "192 256"] = (
        (prompt_confidence_hw >= 1) & (prompt_depth_mm_hw >= PROMPT_MIN_DEPTH_MM) & (prompt_depth_mm_hw <= PROMPT_MAX_DEPTH_MM)
    )
    prompt_depth_m_hw: Float32[ndarray, "192 256"] = prompt_depth_mm_hw.astype(np.float32) / 1000.0
    return np.where(prompt_valid_hw, prompt_depth_m_hw, np.float32(0.0))


def create_compare_blueprint(parent_log_path: Path) -> rrb.Blueprint:
    """Lay out RGB, the LiDAR prompt, both predictions, and their difference."""
    pinhole_path: Path = parent_log_path / "cam" / "pinhole"
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(origin=str(pinhole_path / "rgb"), name="RGB (768x1024)"),
                rrb.Spatial2DView(origin=str(pinhole_path / "lidar_prompt"), name="LiDAR prompt (192x256)"),
                rrb.Spatial2DView(origin=str(pinhole_path / "abs_diff"), name="|teacher - student|"),
            ),
            rrb.Horizontal(
                rrb.Spatial2DView(origin=str(pinhole_path / "teacher_depth"), name="PromptDA-large (teacher)"),
                rrb.Spatial2DView(origin=str(pinhole_path / "student_depth"), name="ZipDepth-PromptDA (student)"),
            ),
        ),
        collapse_panels=True,
    )


def polycam_compare(config: PolycamCompareConfig) -> None:
    """Run teacher and student over one Polycam capture and stream both to Rerun."""
    parent_log_path: Path = Path("world")
    pinhole_path: Path = parent_log_path / "cam" / "pinhole"
    rr.log("/", rr.ViewCoordinates.RUB, static=True)
    rr.send_blueprint(create_compare_blueprint(parent_log_path))

    polycam_dataset: PolycamDataset = load_polycam_data(polycam_zip_or_directory_path=config.polycam_zip_path)
    student_hw: tuple[int, int] = (config.student_height, config.student_width)

    batches = batched(polycam_dataset, config.batch_size)
    first_batch: tuple[PolycamData, ...] | None = next(batches, None)
    if first_batch is None:
        raise ValueError(f"Polycam capture {config.polycam_zip_path} contains no frames.")

    teacher_hw: tuple[int, int] = network_image_hw(first_batch[0].rgb_hw3.shape[:2], config.max_image_size)
    teacher = PromptDATrtPredictor(model_type="large", image_hw=teacher_hw, batch_size=config.batch_size)
    student: ZipDepthPrompt = load_zipdepth_prompt(config.zipdepth_checkpoint).cuda().eval()

    frame_budget: int = config.max_frames if config.max_frames is not None else len(polycam_dataset)
    total_batches: int = -(-min(frame_budget, len(polycam_dataset)) // config.batch_size)

    n_frames: int = 0
    teacher_seconds: float = 0.0
    student_seconds: float = 0.0
    abs_diff_sum: float = 0.0
    abs_diff_count: int = 0
    wall_start: float = time.perf_counter()

    progress = tqdm(chain([first_batch], batches), desc="Comparing", total=total_batches)
    batch: tuple[PolycamData, ...]
    for batch in progress:
        if n_frames >= frame_budget:
            break
        batch_start: int = n_frames
        n_frames += len(batch)

        # The teacher gets the RAW prompt and the student the confidence-masked
        # one. PromptDA normalizes by the prompt's own min/max with no mask and
        # no epsilon, so zeroed holes drag its minimum to 0 and corrupt the
        # prediction; the student was trained on the masked prompt and expects it.
        raw_prompt_bhw: Float32[Tensor, "b 192 256"] = torch.from_numpy(
            np.stack([data.original_depth_hw for data in batch]).astype(np.float32) / 1000.0
        ).cuda()
        masked_prompt_bhw: Float32[Tensor, "b 192 256"] = torch.from_numpy(
            np.stack([masked_prompt_metres(data) for data in batch])
        ).cuda()
        rgb_bhw3: UInt8[Tensor, "b h w 3"] = torch.from_numpy(np.stack([data.rgb_hw3 for data in batch])).cuda()

        torch.cuda.synchronize()
        teacher_start: float = time.perf_counter()
        teacher_depth_bhw: Float32[Tensor, "b th tw"] = teacher(rgb_bhw3, raw_prompt_bhw)
        torch.cuda.synchronize()
        teacher_seconds += time.perf_counter() - teacher_start

        # The student runs at its training resolution; RGB is resized the same
        # way the training transform did, on the host with cv2.INTER_LINEAR.
        student_rgb_b3hw: UInt8[Tensor, "b 3 sh sw"] = torch.from_numpy(
            np.ascontiguousarray(
                np.stack([cv2.resize(data.rgb_hw3, (student_hw[1], student_hw[0]), interpolation=cv2.INTER_LINEAR) for data in batch]).transpose(
                    0, 3, 1, 2
                )
            )
        ).cuda()

        torch.cuda.synchronize()
        student_start: float = time.perf_counter()
        with torch.inference_mode():
            student_depth_b1hw: Float32[Tensor, "b 1 sh sw"] = student(student_rgb_b3hw, masked_prompt_bhw.unsqueeze(1))
        torch.cuda.synchronize()
        student_seconds += time.perf_counter() - student_start

        # Compare on the student's grid: resize the teacher down rather than
        # upsampling the student, so the student is never flattered by interpolation.
        teacher_on_student_b1hw: Float32[Tensor, "b 1 sh sw"] = torch.nn.functional.interpolate(
            teacher_depth_bhw.unsqueeze(1), size=student_hw, mode="bilinear", align_corners=False
        )
        abs_diff_b1hw: Float32[Tensor, "b 1 sh sw"] = (teacher_on_student_b1hw - student_depth_b1hw).abs()
        abs_diff_sum += float(abs_diff_b1hw.sum().item())
        abs_diff_count += int(abs_diff_b1hw.numel())

        teacher_mm_bhw: UInt16[ndarray, "b sh sw"] = (
            (teacher_on_student_b1hw.squeeze(1) * 1000.0).clamp(0.0, 65535.0).to(torch.uint16).cpu().numpy()
        )
        student_mm_bhw: UInt16[ndarray, "b sh sw"] = (
            (student_depth_b1hw.squeeze(1) * 1000.0).clamp(0.0, 65535.0).to(torch.uint16).cpu().numpy()
        )
        diff_mm_bhw: UInt16[ndarray, "b sh sw"] = (abs_diff_b1hw.squeeze(1) * 1000.0).clamp(0.0, 65535.0).to(torch.uint16).cpu().numpy()
        # The raw prompt is what the teacher sees; it is the model's only source
        # of metric scale, so log it at its native 192x256 beside the predictions.
        prompt_mm_bhw: UInt16[ndarray, "b 192 256"] = np.stack([data.original_depth_hw for data in batch])

        frame_offset: int
        for frame_offset in range(len(batch)):
            rr.set_time("frame_idx", sequence=batch_start + frame_offset)
            rr.log(
                str(pinhole_path / "rgb"),
                rr.Image(cv2.resize(batch[frame_offset].rgb_hw3, (student_hw[1], student_hw[0]), interpolation=cv2.INTER_LINEAR)),
            )
            rr.log(
                str(pinhole_path / "lidar_prompt"),
                rr.DepthImage(prompt_mm_bhw[frame_offset], meter=1000.0, colormap="Viridis"),
            )
            rr.log(str(pinhole_path / "teacher_depth"), rr.DepthImage(teacher_mm_bhw[frame_offset], meter=1000.0, colormap="Viridis"))
            rr.log(str(pinhole_path / "student_depth"), rr.DepthImage(student_mm_bhw[frame_offset], meter=1000.0, colormap="Viridis"))
            rr.log(
                str(pinhole_path / "abs_diff"),
                rr.DepthImage(diff_mm_bhw[frame_offset], meter=1000.0, colormap="Inferno", depth_range=(0.0, 250.0)),
            )

    wall_seconds: float = time.perf_counter() - wall_start
    mean_abs_diff_m: float = abs_diff_sum / abs_diff_count if abs_diff_count else 0.0
    print(f"frames                {n_frames}")
    print(f"teacher (PromptDA-L)  {1000.0 * teacher_seconds / n_frames:.2f} ms/frame")
    print(f"student (ZipDepth-PDA){1000.0 * student_seconds / n_frames:.2f} ms/frame")
    print(f"speedup               {teacher_seconds / student_seconds:.1f}x")
    print(f"mean |teacher-student| {1000.0 * mean_abs_diff_m:.1f} mm")
    print(f"wall                  {wall_seconds:.1f} s")


def main(config: PolycamCompareConfig) -> None:
    """Entry point for the Polycam teacher-versus-student comparison."""
    polycam_compare(config)
