"""Three-way Polycam comparison: raw ARKit LiDAR, PromptDA-large, and ZipDepth-PromptDA.

Runs both depth models over the same frames with the same raw prompt and fuses
each depth source into its own TSDF volume, so the three reconstructions can be
compared as geometry rather than only as depth images. Accumulated depth error
shows up in a mesh in ways a per-frame depth view hides.

The raw ARKit stream is the device's own upsampled depth -- the baseline any
phone gives you for free, and the thing both models have to beat.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import batched, chain
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from einops import rearrange
from jaxtyping import Float32, UInt8, UInt16
from monopriors.models.depth_completion.zipdepth_prompt import ZipDepthPrompt, load_zipdepth_prompt
from numpy import ndarray
from simplecv.camera_parameters import rescale_intri
from simplecv.data.polycam import DepthConfidenceLevel, PolycamData, PolycamDataset, load_polycam_data
from simplecv.ops.tsdf_depth_fuser import Open3DFuser
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole
from torch import Tensor
from tqdm import tqdm

from rerun_prompt_da.apis.prompt_da_polycam import filter_depth
from rerun_prompt_da.apis.prompt_da_trt_polycam import network_image_hw
from rerun_prompt_da.mesh_logging import log_fused_mesh
from rerun_prompt_da.trt_predictor import PromptDATrtPredictor

SOURCES: tuple[str, str, str] = ("arkit", "promptda", "zipdepth")
"""Depth sources compared, in increasing order of expected quality."""


@dataclass
class PolycamTrioConfig:
    """Runtime configuration for the three-way Polycam comparison."""

    polycam_zip_path: Path
    """Polycam capture zip (or extracted directory) to process."""
    zipdepth_checkpoint: Path
    """Trained ZipDepth-PromptDA checkpoint; also the PyTorch fallback when no engine is given."""
    zipdepth_engine: Path | None = None
    """Static-batch TensorRT fp16 engine for the student, built by ``zipdepth-export-prompted-trt``.

    Parity against PyTorch is 0.003 mm at the 95th percentile -- four orders of magnitude
    below the model's own ~22 mm error -- so running the engine costs nothing in fidelity
    and makes the timing comparison against the TensorRT teacher apples-to-apples."""
    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Rerun viewer/save/connect behavior."""
    batch_size: int = 8
    """Frames per model batch."""
    max_image_size: int = 1008
    """Longest PromptDA network image side; 14-aligned from the capture resolution."""
    max_depth_range_meter: float = 4.0
    """Depth beyond this is dropped before TSDF fusion."""
    depth_fusion_resolution: float = 0.04
    """TSDF voxel size in metres."""
    max_frames: int | None = None
    """Optional cap on processed frames."""
    log_incremental_mesh: bool = False
    """Log all three fused meshes after every batch instead of only at the end.

    Shows the reconstructions growing as the capture is walked, which makes it obvious
    where each source starts accumulating drift. Costs one mesh upload per source per
    batch, so the recording is substantially larger."""


ERROR_RANGE_MM: tuple[float, float] = (0.0, 250.0)
"""Shared colour range for both error panes, so they read against each other."""
SPIN_SPEED: float = 0.25
"""Orbital spin applied to every mesh view, so all three rotate together and the
reconstructions can be compared from the same angle without manual navigation."""


def create_trio_blueprint(parent_log_path: Path) -> rrb.Blueprint:
    """Lay out the fused meshes, plus a tab comparing each source against the teacher."""
    pinhole_path: Path = parent_log_path / "cam" / "pinhole"
    lowres_path: Path = parent_log_path / "cam" / "pinhole_lowres"
    return rrb.Blueprint(
        rrb.Tabs(
            _mesh_tab(parent_log_path, pinhole_path, lowres_path),
            _comparison_tab(pinhole_path, lowres_path),
        ),
        collapse_panels=True,
    )


def _comparison_tab(pinhole_path: Path, lowres_path: Path) -> rrb.Vertical:
    """Two rows against a common reference: the teacher.

    Row 2 is the size of the problem -- everything PromptDA changes about the raw ARKit
    depth. Row 1 is the student's residual -- the part of that correction it did not
    reproduce. Both error panes share one scale, so they read directly against each other.
    """
    return rrb.Vertical(
        rrb.Horizontal(
            rrb.Spatial2DView(origin=str(lowres_path / "arkit_depth"), name="raw ARKit LiDAR (192x256, native)"),
            rrb.Spatial2DView(origin=str(pinhole_path / "zipdepth_depth"), name="ZipDepth-PromptDA (6.95M)"),
            rrb.Spatial2DView(origin=str(pinhole_path / "err_student"), name="|ZipDepth - PromptDA|  (student residual)"),
        ),
        rrb.Horizontal(
            rrb.Spatial2DView(origin=str(lowres_path / "arkit_depth"), name="raw ARKit LiDAR (192x256, native)"),
            rrb.Spatial2DView(origin=str(pinhole_path / "promptda_depth"), name="PromptDA-large (DAv2, 340M)"),
            rrb.Spatial2DView(origin=str(lowres_path / "err_arkit"), name="|ARKit - PromptDA|  (what the teacher fixes)"),
        ),
        name="side by side",
    )


def _mesh_tab(parent_log_path: Path, pinhole_path: Path, lowres_path: Path) -> rrb.Horizontal:
    """The fused reconstructions beside the raw depth streams."""
    return rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial3DView(
                    origin=str(parent_log_path / "arkit"),
                    contents=["$origin/**", str(parent_log_path / "cam") + "/**"],
                    name="mesh: raw ARKit LiDAR (192x256)",
                    eye_controls=rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital, spin_speed=SPIN_SPEED),
                ),
                rrb.Spatial3DView(
                    origin=str(parent_log_path / "promptda"),
                    contents=["$origin/**", str(parent_log_path / "cam") + "/**"],
                    name="mesh: PromptDA-large (DAv2, 340M)",
                    eye_controls=rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital, spin_speed=SPIN_SPEED),
                ),
                rrb.Spatial3DView(
                    origin=str(parent_log_path / "zipdepth"),
                    contents=["$origin/**", str(parent_log_path / "cam") + "/**"],
                    name="mesh: ZipDepth-PromptDA (6.95M)",
                    eye_controls=rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital, spin_speed=SPIN_SPEED),
                ),
            ),
            rrb.Vertical(
                rrb.Spatial2DView(origin=str(pinhole_path / "image"), name="RGB"),
                rrb.Spatial2DView(origin=str(lowres_path / "confidence"), name="ARKit confidence (192x256)"),
                rrb.Spatial2DView(origin=str(lowres_path / "arkit_depth"), name="raw ARKit LiDAR (192x256, native)"),
                rrb.Spatial2DView(origin=str(pinhole_path / "promptda_depth"), name="PromptDA-large"),
                rrb.Spatial2DView(origin=str(pinhole_path / "zipdepth_depth"), name="ZipDepth-PromptDA"),
            ),
        column_shares=[3, 2],
        name="meshes",
    )


def _log_source_mesh(parent_log_path: Path, source: str, fuser: Open3DFuser) -> None:
    """Log one source's fused TSDF mesh under its own entity, at the current time."""
    mesh = fuser.get_mesh()
    mesh.compute_vertex_normals()
    log_fused_mesh(None, str(parent_log_path / source / "mesh"), mesh, static=False)


def polycam_trio(config: PolycamTrioConfig) -> None:
    """Fuse and log raw ARKit, PromptDA, and ZipDepth-PromptDA depth for one capture."""
    parent_log_path: Path = Path("world")
    pinhole_path: Path = parent_log_path / "cam" / "pinhole"
    lowres_path: Path = parent_log_path / "cam" / "pinhole_lowres"
    rr.log("/", rr.ViewCoordinates.RUB, static=True)
    rr.send_blueprint(create_trio_blueprint(parent_log_path))
    rr.log(
        str(lowres_path / "confidence"),
        rr.AnnotationContext(
            [
                rr.AnnotationInfo(id=0, label="low", color=(220, 40, 40)),
                rr.AnnotationInfo(id=1, label="medium", color=(235, 200, 40)),
                rr.AnnotationInfo(id=2, label="high", color=(60, 200, 90)),
            ]
        ),
        static=True,
    )

    dataset: PolycamDataset = load_polycam_data(polycam_zip_or_directory_path=config.polycam_zip_path)
    fusers: dict[str, Open3DFuser] = {
        source: Open3DFuser(fusion_resolution=config.depth_fusion_resolution, max_fusion_depth=config.max_depth_range_meter)
        for source in SOURCES
    }

    batches = batched(dataset, config.batch_size)
    first_batch: tuple[PolycamData, ...] | None = next(batches, None)
    if first_batch is None:
        raise ValueError(f"Polycam capture {config.polycam_zip_path} contains no frames.")

    # The raw LiDAR is a fixed 192x256 per PolycamData; one static scale transform maps
    # its 2D frame onto the full-res image (the posekit SAM2-mask pattern, track_ui.py).
    native_hw: tuple[int, int] = first_batch[0].original_depth_hw.shape
    native_scale: float = first_batch[0].rgb_hw3.shape[1] / native_hw[1]
    rr.log(str(lowres_path), rr.Transform3D(scale=native_scale), static=True)

    capture_hw: tuple[int, int] = first_batch[0].rgb_hw3.shape[:2]
    teacher = PromptDATrtPredictor(
        model_type="large",
        image_hw=network_image_hw(capture_hw, config.max_image_size),
        batch_size=config.batch_size,
    )
    # Both closures take BHW3 and do their own layout work INSIDE, so the timing
    # brackets measure the same preprocessing for student and teacher alike.
    student_backend: str
    predict_student: Callable[[UInt8[Tensor, "b h w 3"], Float32[Tensor, "b 1 192 256"]], Float32[Tensor, "b 1 h w"]]
    if config.zipdepth_engine is None:
        # fuse_for_inference folds the RepVGG branches, matching how eval_catalog.py
        # and the TRT export run the model; unfused eager understates its speed.
        student: ZipDepthPrompt = load_zipdepth_prompt(config.zipdepth_checkpoint).cuda().fuse_for_inference()

        def predict_student(rgb_bhw3: UInt8[Tensor, "b h w 3"], prompt_b1hw: Float32[Tensor, "b 1 192 256"]) -> Float32[Tensor, "b 1 h w"]:
            image_b3hw: UInt8[Tensor, "b 3 h w"] = rearrange(rgb_bhw3, "b h w c -> b c h w").contiguous()  # pyrefly: ignore  # bad-argument-type — einops stub false positive
            with torch.inference_mode():
                return student(image_b3hw, prompt_b1hw)

        student_backend = "PyTorch eager"
    else:
        from trtkit import TensorRtRuntime

        runtime = TensorRtRuntime(config.zipdepth_engine)

        def predict_student(rgb_bhw3: UInt8[Tensor, "b h w 3"], prompt_b1hw: Float32[Tensor, "b 1 192 256"]) -> Float32[Tensor, "b 1 h w"]:
            # The engine bakes the uint8->[0,1] scaling the wrapper does in torch.
            image_b3hw: Float32[Tensor, "b 3 h w"] = rearrange(rgb_bhw3, "b h w c -> b c h w").contiguous().float() / 255.0  # pyrefly: ignore  # bad-argument-type — einops stub false positive
            outputs = runtime({"image": image_b3hw, "prompt_depth": prompt_b1hw})
            return outputs["depth"].clone()

        student_backend = "TensorRT fp16"

    frame_budget: int = config.max_frames if config.max_frames is not None else len(dataset)
    n_frames: int = 0
    seconds: dict[str, float] = {"promptda": 0.0, "zipdepth": 0.0}
    wall_start: float = time.perf_counter()

    progress = tqdm(chain([first_batch], batches), desc="Trio", total=-(-min(frame_budget, len(dataset)) // config.batch_size))
    batch: tuple[PolycamData, ...]
    for batch in progress:
        if n_frames >= frame_budget:
            break
        batch_start: int = n_frames
        n_frames += len(batch)

        # Both models see the same RAW prompt: PromptDA normalizes by its own min/max
        # with no mask, and the student was trained on the raw prompt to match.
        prompt_bhw: Float32[Tensor, "b 192 256"] = torch.from_numpy(
            np.stack([data.original_depth_hw for data in batch]).astype(np.float32) / 1000.0
        ).cuda()
        rgb_bhw3: UInt8[Tensor, "b h w 3"] = torch.from_numpy(np.stack([data.rgb_hw3 for data in batch])).cuda()

        torch.cuda.synchronize()
        started: float = time.perf_counter()
        teacher_bhw: Float32[Tensor, "b h w"] = teacher(rgb_bhw3, prompt_bhw)
        torch.cuda.synchronize()
        seconds["promptda"] += time.perf_counter() - started

        torch.cuda.synchronize()
        started = time.perf_counter()
        student_bhw: Float32[Tensor, "b h w"] = predict_student(rgb_bhw3, prompt_bhw.unsqueeze(1)).squeeze(1)
        torch.cuda.synchronize()
        seconds["zipdepth"] += time.perf_counter() - started

        predicted_mm: dict[str, UInt16[ndarray, "b h w"]] = {
            "promptda": (teacher_bhw * 1000.0).clamp(0.0, 65535.0).to(torch.uint16).cpu().numpy(),
            "zipdepth": (student_bhw * 1000.0).clamp(0.0, 65535.0).to(torch.uint16).cpu().numpy(),
        }

        frame_offset: int
        polycam_data: PolycamData
        for frame_offset, polycam_data in enumerate(batch):
            rr.set_time("frame_idx", sequence=batch_start + frame_offset)
            k_matrix: Float32[ndarray, "3 3"] | None = polycam_data.pinhole_params.intrinsics.k_matrix
            if k_matrix is None:
                raise ValueError("Polycam pinhole intrinsics must include a 3x3 k_matrix for TSDF fusion.")

            log_pinhole(camera=polycam_data.pinhole_params, cam_log_path=parent_log_path / "cam")
            rr.log(str(pinhole_path / "image"), rr.Image(polycam_data.rgb_hw3).compress(jpeg_quality=75))

            # The raw LiDAR is 192x256. Log and fuse it at that size -- upsampling it first
            # would flatter the sensor and hide the 16x resolution gap the models close.
            # A static Transform3D scales its 2D frame onto the full-res image, the same
            # pattern posekit uses for SAM2 masks (track_ui.py:160).
            rr.log(
                str(lowres_path / "arkit_depth"),
                rr.DepthImage(polycam_data.original_depth_hw, meter=1000.0, colormap="Viridis"),
            )
            # Polycam stores confidence as 0/54/255; compact to class ids 0/1/2 so the
            # annotation context colors apply (red = low, yellow = medium, green = high).
            confidence_class_hw: UInt8[ndarray, "nh nw"] = (
                (polycam_data.original_confidence_hw >= DepthConfidenceLevel.MEDIUM).astype(np.uint8)
                + (polycam_data.original_confidence_hw >= DepthConfidenceLevel.HIGH).astype(np.uint8)
            )
            rr.log(str(lowres_path / "confidence"), rr.SegmentationImage(confidence_class_hw))

            # Errors share one reference -- the teacher -- so the two panes read against
            # each other: row 2 is the correction PromptDA makes to the raw sensor, row 1
            # is the part of that correction the student did not reproduce. Row 2 is
            # measured at the sensor's own resolution by pooling the teacher down to it.
            teacher_full_hw: Float32[ndarray, "h w"] = predicted_mm["promptda"][frame_offset].astype(np.float32)
            student_error_hw: UInt16[ndarray, "h w"] = (
                np.abs(predicted_mm["zipdepth"][frame_offset].astype(np.float32) - teacher_full_hw).clip(0.0, 65535.0).astype(np.uint16)
            )
            rr.log(
                str(pinhole_path / "err_student"),
                rr.DepthImage(student_error_hw, meter=1000.0, colormap="Inferno", depth_range=ERROR_RANGE_MM),
            )
            teacher_native_hw: Float32[ndarray, "nh nw"] = cv2.resize(
                teacher_full_hw, (native_hw[1], native_hw[0]), interpolation=cv2.INTER_AREA
            )
            arkit_error_hw: UInt16[ndarray, "nh nw"] = (
                np.abs(polycam_data.original_depth_hw.astype(np.float32) - teacher_native_hw).clip(0.0, 65535.0).astype(np.uint16)
            )
            rr.log(
                str(lowres_path / "err_arkit"),
                rr.DepthImage(arkit_error_hw, meter=1000.0, colormap="Inferno", depth_range=ERROR_RANGE_MM),
            )

            # Each source fuses on its own grid: the models at full resolution, the raw
            # sensor at 192x256 with matching RGB and rescaled intrinsics, so the ARKit
            # mesh shows what the LiDAR alone actually reconstructs.
            native_rgb_hw3: UInt8[ndarray, "nh nw 3"] = cv2.resize(
                polycam_data.rgb_hw3, (native_hw[1], native_hw[0]), interpolation=cv2.INTER_AREA
            )
            native_k_33: Float32[ndarray, "3 3"] | None = rescale_intri(
                camera_intrinsics=polycam_data.pinhole_params.intrinsics,
                target_height=native_hw[0],
                target_width=native_hw[1],
            ).k_matrix
            if native_k_33 is None:
                raise ValueError("rescaled Polycam intrinsics must include a 3x3 k_matrix for native-resolution fusion.")
            fuse_inputs: dict[str, tuple[UInt16[ndarray, "h w"], UInt8[ndarray, "h w"], Float32[ndarray, "3 3"], UInt8[ndarray, "h w 3"]]] = {
                "arkit": (polycam_data.original_depth_hw, polycam_data.original_confidence_hw, native_k_33, native_rgb_hw3),
                "promptda": (predicted_mm["promptda"][frame_offset], polycam_data.confidence_hw, k_matrix, polycam_data.rgb_hw3),
                "zipdepth": (predicted_mm["zipdepth"][frame_offset], polycam_data.confidence_hw, k_matrix, polycam_data.rgb_hw3),
            }
            source: str
            for source in SOURCES:
                depth_mm, source_confidence, source_k_33, source_rgb = fuse_inputs[source]
                if source != "arkit":
                    rr.log(str(pinhole_path / f"{source}_depth"), rr.DepthImage(depth_mm, meter=1000.0, colormap="Viridis"))
                fusers[source].fuse_frames(
                    depth_hw=filter_depth(
                        depth_mm=depth_mm,
                        confidence=source_confidence,
                        confidence_threshold=DepthConfidenceLevel.MEDIUM,
                        max_depth_meter=config.max_depth_range_meter,
                    ),
                    K_33=source_k_33,
                    cam_T_world_44=polycam_data.pinhole_params.extrinsics.cam_T_world,
                    rgb_hw3=source_rgb,
                )

        if config.log_incremental_mesh:
            for source in SOURCES:
                _log_source_mesh(parent_log_path, source, fusers[source])

    if not config.log_incremental_mesh:
        # The incremental path already logged the final state after the last batch.
        for source in SOURCES:
            _log_source_mesh(parent_log_path, source, fusers[source])

    wall_seconds: float = time.perf_counter() - wall_start
    print(f"frames                {n_frames}")
    print(f"PromptDA-large        {1000.0 * seconds['promptda'] / n_frames:.2f} ms/frame  (TensorRT fp16)")
    print(f"ZipDepth-PromptDA     {1000.0 * seconds['zipdepth'] / n_frames:.2f} ms/frame  ({student_backend})")
    print(f"speedup               {seconds['promptda'] / seconds['zipdepth']:.1f}x")
    print(f"wall                  {wall_seconds:.1f} s  (three TSDF volumes)")


def main(config: PolycamTrioConfig) -> None:
    """Entry point for the three-way Polycam comparison."""
    polycam_trio(config)
