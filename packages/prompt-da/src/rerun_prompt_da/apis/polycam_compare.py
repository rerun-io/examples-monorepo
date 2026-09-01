"""Side-by-side PromptDA teacher vs ZipDepth-PromptDA student on a Polycam capture.

Runs both depth models over the same frames with the same LiDAR prompt and logs
RGB, both predictions, and their absolute difference to Rerun for visual
inspection. The student is the distilled 6.14 M-parameter model; the teacher is
the 340 M-parameter PromptDA-large TensorRT engine that produced its training
labels.

Both models receive the same raw LiDAR prompt -- distillation is only meaningful
when student and teacher see identical input. RGB resizing mirrors the training
transform (``cv2.INTER_LINEAR``).
"""

import time
from dataclasses import dataclass, field, replace
from itertools import batched, chain
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Float32, UInt8, UInt16
from monopriors.models.depth_completion import AnnotatedCompletionConfig, PromptDAConfig, ZipDepthPromptConfig
from monopriors.models.depth_completion.base_completion_depth import BaseCompletionPredictor
from monopriors.models.depth_completion.prompt_da import DEFAULT_PROMPTDA_CACHE_DIR
from numpy import ndarray
from simplecv.data.polycam import PolycamData, PolycamDataset, load_polycam_data
from simplecv.rerun_log_utils import RerunTyroConfig
from torch import Tensor
from tqdm import tqdm
from trtkit import TensorRtBackendConfig, TorchBackendConfig


@dataclass
class PolycamCompareConfig:
    """Runtime configuration for the teacher-versus-student Polycam comparison."""

    polycam_zip_path: Path
    """Polycam capture zip (or extracted directory) to process."""
    teacher: AnnotatedCompletionConfig = field(
        default_factory=lambda: PromptDAConfig(
            backend=TensorRtBackendConfig(max_batch_size=8, opt_batch_size=8, cache_dir=DEFAULT_PROMPTDA_CACHE_DIR / "trt")
        )
    )
    """First completion model and backend."""
    student: AnnotatedCompletionConfig = field(
        default_factory=lambda: ZipDepthPromptConfig(checkpoint=Path(), backend=TorchBackendConfig())
    )
    """Second completion model and backend."""
    zipdepth_checkpoint: Path | None = None
    """Compatibility alias for ``student=zipdepth-promptda --checkpoint``."""
    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Rerun viewer/save/connect behavior."""
    batch_size: int = 8
    """Frames per model batch."""
    max_frames: int | None = None
    """Optional cap on processed frames, for a quick visual check."""



def create_compare_blueprint(
    parent_log_path: Path,
    teacher_label: str = "teacher",
    student_label: str = "student",
) -> rrb.Blueprint:
    """Lay out RGB, the LiDAR prompt, both predictions, and their difference."""
    pinhole_path: Path = parent_log_path / "cam" / "pinhole"
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(origin=str(pinhole_path / "rgb"), name="RGB (768x1024)"),
                rrb.Spatial2DView(origin=str(pinhole_path / "lidar_prompt"), name="LiDAR prompt (192x256)"),
            ),
            rrb.Horizontal(
                rrb.Spatial2DView(origin=str(pinhole_path / "teacher_depth"), name=teacher_label),
                rrb.Spatial2DView(origin=str(pinhole_path / "student_depth"), name=student_label),
                rrb.Spatial2DView(origin=str(pinhole_path / "bilinear_depth"), name="bilinear upsample (0 params)"),
            ),
            rrb.Horizontal(
                rrb.Spatial2DView(origin=str(pinhole_path / "abs_diff"), name="|teacher - student|"),
                rrb.Spatial2DView(origin=str(pinhole_path / "abs_diff_bilinear"), name="|teacher - bilinear|"),
            ),
        ),
        collapse_panels=True,
    )


def polycam_compare(config: PolycamCompareConfig) -> None:
    """Run teacher and student over one Polycam capture and stream both to Rerun."""
    parent_log_path: Path = Path("world")
    pinhole_path: Path = parent_log_path / "cam" / "pinhole"
    teacher_label: str = type(config.teacher).__name__
    student_config: AnnotatedCompletionConfig = config.student
    if config.zipdepth_checkpoint is not None:
        if not isinstance(student_config, ZipDepthPromptConfig):
            raise ValueError("--zipdepth-checkpoint requires the ZipDepth-PromptDA student config.")
        student_config = replace(student_config, checkpoint=config.zipdepth_checkpoint)
    student_label: str = type(student_config).__name__
    rr.log("/", rr.ViewCoordinates.RUB, static=True)
    rr.send_blueprint(create_compare_blueprint(parent_log_path, teacher_label, student_label))

    polycam_dataset: PolycamDataset = load_polycam_data(polycam_zip_or_directory_path=config.polycam_zip_path)

    batches = batched(polycam_dataset, config.batch_size)
    first_batch: tuple[PolycamData, ...] | None = next(batches, None)
    if first_batch is None:
        raise ValueError(f"Polycam capture {config.polycam_zip_path} contains no frames.")

    comparison_hw: tuple[int, int] = first_batch[0].rgb_hw3.shape[:2]
    teacher: BaseCompletionPredictor = config.teacher.setup()
    student: BaseCompletionPredictor = student_config.setup()

    frame_budget: int = config.max_frames if config.max_frames is not None else len(polycam_dataset)
    total_batches: int = -(-min(frame_budget, len(polycam_dataset)) // config.batch_size)

    n_frames: int = 0
    teacher_seconds: float = 0.0
    student_seconds: float = 0.0
    abs_diff_sum: float = 0.0
    bilinear_sum: float = 0.0
    abs_diff_count: int = 0
    wall_start: float = time.perf_counter()

    progress = tqdm(chain([first_batch], batches), desc="Comparing", total=total_batches)
    batch: tuple[PolycamData, ...]
    for batch in progress:
        if n_frames >= frame_budget:
            break
        batch_start: int = n_frames
        n_frames += len(batch)

        # Both models get the same RAW prompt. Distillation only makes sense when
        # student and teacher see identical input, and PromptDA normalizes by the
        # prompt's own min/max with no mask -- zeroed holes would drag its minimum
        # to 0 and corrupt the teacher.
        raw_prompt_bhw: Float32[Tensor, "b 192 256"] = torch.from_numpy(
            np.stack([data.original_depth_hw for data in batch]).astype(np.float32) / 1000.0
        ).cuda()
        rgb_bhw3: UInt8[Tensor, "b h w 3"] = torch.from_numpy(np.stack([data.rgb_hw3 for data in batch])).cuda()

        torch.cuda.synchronize()
        teacher_start: float = time.perf_counter()
        teacher_depth_bhw: Float32[Tensor, "b h w"] = teacher(rgb_bhw3, raw_prompt_bhw)
        torch.cuda.synchronize()
        teacher_seconds += time.perf_counter() - teacher_start

        torch.cuda.synchronize()
        student_start: float = time.perf_counter()
        student_depth_bhw: Float32[Tensor, "b h w"] = student(rgb_bhw3, raw_prompt_bhw)
        torch.cuda.synchronize()
        student_seconds += time.perf_counter() - student_start

        teacher_on_student_b1hw: Float32[Tensor, "b 1 h w"] = teacher_depth_bhw.unsqueeze(1)
        student_depth_b1hw: Float32[Tensor, "b 1 h w"] = student_depth_bhw.unsqueeze(1)
        # Zero-parameter control: bilinearly upsample the same raw prompt. Logged beside
        # the student so the difference the network actually buys is visible, not asserted.
        bilinear_b1hw: Float32[Tensor, "b 1 h w"] = torch.nn.functional.interpolate(
            raw_prompt_bhw.unsqueeze(1), size=comparison_hw, mode="bilinear", align_corners=False
        )
        abs_diff_b1hw: Float32[Tensor, "b 1 h w"] = (teacher_on_student_b1hw - student_depth_b1hw).abs()
        abs_diff_bilinear_b1hw: Float32[Tensor, "b 1 h w"] = (teacher_on_student_b1hw - bilinear_b1hw).abs()
        bilinear_sum += float(abs_diff_bilinear_b1hw.sum().item())
        abs_diff_sum += float(abs_diff_b1hw.sum().item())
        abs_diff_count += int(abs_diff_b1hw.numel())

        teacher_mm_bhw: UInt16[ndarray, "b h w"] = (
            (teacher_on_student_b1hw.squeeze(1) * 1000.0).clamp(0.0, 65535.0).to(torch.uint16).cpu().numpy()
        )
        student_mm_bhw: UInt16[ndarray, "b h w"] = (
            (student_depth_b1hw.squeeze(1) * 1000.0).clamp(0.0, 65535.0).to(torch.uint16).cpu().numpy()
        )
        diff_mm_bhw: UInt16[ndarray, "b h w"] = (abs_diff_b1hw.squeeze(1) * 1000.0).clamp(0.0, 65535.0).to(torch.uint16).cpu().numpy()
        bilinear_mm_bhw: UInt16[ndarray, "b h w"] = (bilinear_b1hw.squeeze(1) * 1000.0).clamp(0.0, 65535.0).to(torch.uint16).cpu().numpy()
        diff_bil_mm_bhw: UInt16[ndarray, "b h w"] = (
            (abs_diff_bilinear_b1hw.squeeze(1) * 1000.0).clamp(0.0, 65535.0).to(torch.uint16).cpu().numpy()
        )
        # The raw prompt is what the teacher sees; it is the model's only source
        # of metric scale, so log it at its native 192x256 beside the predictions.
        prompt_mm_bhw: UInt16[ndarray, "b 192 256"] = np.stack([data.original_depth_hw for data in batch])

        frame_offset: int
        for frame_offset in range(len(batch)):
            rr.set_time("frame_idx", sequence=batch_start + frame_offset)
            rr.log(
                str(pinhole_path / "rgb"),
                rr.Image(batch[frame_offset].rgb_hw3),
            )
            rr.log(
                str(pinhole_path / "lidar_prompt"),
                rr.DepthImage(prompt_mm_bhw[frame_offset], meter=1000.0, colormap="Viridis"),
            )
            rr.log(str(pinhole_path / "teacher_depth"), rr.DepthImage(teacher_mm_bhw[frame_offset], meter=1000.0, colormap="Viridis"))
            rr.log(str(pinhole_path / "student_depth"), rr.DepthImage(student_mm_bhw[frame_offset], meter=1000.0, colormap="Viridis"))
            rr.log(
                str(pinhole_path / "bilinear_depth"),
                rr.DepthImage(bilinear_mm_bhw[frame_offset], meter=1000.0, colormap="Viridis"),
            )
            rr.log(
                str(pinhole_path / "abs_diff"),
                rr.DepthImage(diff_mm_bhw[frame_offset], meter=1000.0, colormap="Inferno", depth_range=(0.0, 250.0)),
            )
            rr.log(
                str(pinhole_path / "abs_diff_bilinear"),
                rr.DepthImage(diff_bil_mm_bhw[frame_offset], meter=1000.0, colormap="Inferno", depth_range=(0.0, 250.0)),
            )

    wall_seconds: float = time.perf_counter() - wall_start
    mean_abs_diff_m: float = abs_diff_sum / abs_diff_count if abs_diff_count else 0.0
    print(f"frames                {n_frames}")
    print(f"{teacher_label:<22}{1000.0 * teacher_seconds / n_frames:.2f} ms/frame")
    print(f"{student_label:<22}{1000.0 * student_seconds / n_frames:.2f} ms/frame")
    print(f"speedup               {teacher_seconds / student_seconds:.1f}x")
    print(f"mean |teacher-student|  {1000.0 * mean_abs_diff_m:.1f} mm")
    print(f"mean |teacher-bilinear| {1000.0 * bilinear_sum / abs_diff_count:.1f} mm  (zero-parameter control)")
    print(f"wall                  {wall_seconds:.1f} s")


def main(config: PolycamCompareConfig) -> None:
    """Entry point for the Polycam teacher-versus-student comparison."""
    polycam_compare(config)
