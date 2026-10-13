"""Graph-tier check of vistadream's single-image graph, driven by ``VERIFY.toml`` through ``python -m dagkit.verify``.

No depth GT is registered on the catalog, so the reference is the PRE-PORT golden of each case. The pre-port pipeline is
broken at import (``@dataclass`` on the wrong definition, a removed ``monopriors`` factory); ``pre``/``pre2`` ran the OLD
``SingleImagePipeline`` with the smallest fix applied and never committed (see ``VERIFY.toml``). ``score`` reads only
golden rrds, never the pipeline.

Golden contract (``recording_id`` = case name; clock ``video_time``: the image at t = 0, trajectory step i at i / 30 s):

- ``/world/splats``: the final scene as ``rr.GaussianSplats3D`` (linear scales, xyzw quaternions, RGBA = base colour and
  opacity), t = 0.
- ``/world/input`` (``Transform3D``, identity) and ``/world/input/pinhole`` (``Pinhole``): the input camera, t = 0, with
  ``rgb`` (``Image`` uint8 RGB, the processed input), ``depth`` (``DepthImage`` float32, MoGe), ``valid`` (``Image`` L
  uint8 0/255, MoGe validity), and the final scene rendered at that camera: ``render_rgb`` (``Image`` uint8 RGB) and
  ``render_depth`` (``DepthImage`` float32).
- ``/world/render`` (``Transform3D``, camera from world) and ``/world/render/pinhole`` (``Pinhole``) with ``rgb``
  (``Image`` uint8 RGB): one render per trajectory step.
- ``/verify/single_image/run``: static ``AnyValues`` (``wall_s``, ``n_splats``, ``stage``, ``code``).
"""

import random
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import rerun as rr
import torch
import torch.nn.functional as F
from dagkit.verify import Case, Phase, golden_path
from jaxtyping import Bool, Float32, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry

PACKAGE_DIR: Path = Path(__file__).resolve().parents[2]
"""``packages/vistadream``: ``Case.input`` is relative to it."""
TIMELINE: str = "video_time"
FPS: int = 30
INPUT: str = "/world/input/pinhole"
RENDER: str = "/world/render/pinhole"
PREPORT_CODE: str = "pre-port SingleImagePipeline + uncommitted import fix"


def _log_camera(recording: rr.RecordingStream, path: str, cam_T_world: ndarray, k: ndarray, width: int, height: int) -> None:
    rotation: ndarray = cam_T_world[:3, :3]
    recording.log(path, rr.Transform3D(translation=cam_T_world[:3, 3], mat3x3=rotation, relation=rr.TransformRelation.ChildFromParent))
    recording.log(f"{path}/pinhole", rr.Pinhole(image_from_camera=k, resolution=[width, height], camera_xyz=rr.ViewCoordinates.RDF))


def run(case: Case, phase: Phase, recording: rr.RecordingStream) -> None:
    """The OLD single-image pipeline (initialise, then ``_coarse`` for ``stage = coarse``) with seeded RNGs; its own Rerun
    logging goes to a discarded stream. The final scene, the input camera and the trajectory renders go to ``recording``."""
    from vistadream.api.single_img_pipeline import SingleImageConfig, SingleImagePipeline
    from vistadream.ops.gs.basic import Frame
    from vistadream.ops.trajs import _generate_trajectory

    assert case.input is not None, "vistadream cases are local images"
    stage: str = case.params["stage"]
    if stage != "no-outpaint":  # FLUX.1-Fill-dev is ~34 GB: refuse to start a download from a verify run
        from huggingface_hub import snapshot_download

        snapshot_download("black-forest-labs/FLUX.1-Fill-dev", local_files_only=True)
    seed: int = int(case.params["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = SingleImageConfig(
        rr_config=cast(Any, None),  # unused by the pipeline; constructing one would open a global stream
        image_path=PACKAGE_DIR / case.input,
        stage=cast(Any, stage),
        max_resolution=cast(Any, int(case.params["max_resolution"])),
        expansion_percent=float(case.params.get("expansion_percent", "0.3")),
        n_frames=int(case.params.get("n_frames", "10")),
    )
    started: float = time.perf_counter()
    with rr.RecordingStream("vistadream-preport-discarded"):  # the old pipeline logs through the global API
        pipeline = SingleImagePipeline(config)
        pipeline.setup_rerun()
        pipeline._initialize()
        if stage == "coarse":
            pipeline._coarse()
    scene = pipeline.scene
    intrinsics = pipeline.shared_intrinsics
    assert intrinsics is not None and intrinsics.k_matrix is not None
    n_render: int = min(len(scene.frames) * 25 if len(scene.frames) > 2 else 150, 150)  # the old _render_splats rule
    trajectory: ndarray = _generate_trajectory(scene, nframes=n_render)

    recording.log("/world", rr.ViewCoordinates.RDF, static=True)
    recording.set_time(TIMELINE, duration=np.timedelta64(0, "ns"))
    with torch.no_grad():
        frames = scene.gaussian_frames
        centers: Float32[ndarray, "n 3"] = torch.cat([f.xyz.reshape(-1, 3) for f in frames]).float().cpu().numpy()
        scales: Float32[ndarray, "n 3"] = torch.exp(torch.cat([f.scale.reshape(-1, 3) for f in frames])).float().cpu().numpy()
        quats_wxyz = F.normalize(torch.cat([f.rotation.reshape(-1, 4) for f in frames]), dim=1)
        rgb = torch.sigmoid(torch.cat([f.rgb.reshape(-1, 3) for f in frames]))
        opacity = torch.sigmoid(torch.cat([f.opacity.reshape(-1, 1) for f in frames]))
        rgba: UInt8[ndarray, "n 4"] = (torch.cat([rgb, opacity], dim=1) * 255.0).round().clamp(0, 255).byte().cpu().numpy()
        recording.log(
            "/world/splats",
            rr.GaussianSplats3D(centers=centers, scales=scales, quaternions=quats_wxyz[:, [1, 2, 3, 0]].float().cpu().numpy(), colors=rgba),
        )
        source = scene.frames[0 if stage == "no-outpaint" else 1]  # _initialize adds the outpaint frame first, then the input
        assert source.rgb is not None and source.dpt is not None and source.dpt_conf_mask is not None and source.intrinsic is not None
        height, width = source.rgb.shape[:2]
        _log_camera(recording, "/world/input", np.eye(4), source.intrinsic, width, height)
        recording.log(f"{INPUT}/rgb", rr.Image((source.rgb * 255.0).round().astype(np.uint8), color_model="RGB"))
        recording.log(f"{INPUT}/depth", rr.DepthImage(source.dpt.astype(np.float32), meter=1.0))
        recording.log(f"{INPUT}/valid", rr.Image(source.dpt_conf_mask.astype(np.uint8) * 255, color_model="L"))
        render_rgb, render_depth, _ = scene._render_RGBD(Frame(H=height, W=width, intrinsic=source.intrinsic, cam_T_world=np.eye(4)))
        recording.log(f"{INPUT}/render_rgb", rr.Image((render_rgb.clamp(0, 1) * 255.0).round().byte().cpu().numpy(), color_model="RGB"))
        recording.log(f"{INPUT}/render_depth", rr.DepthImage(render_depth.float().cpu().numpy(), meter=1.0))
        for step, cam_T_world in enumerate(trajectory):
            recording.set_time(TIMELINE, duration=np.timedelta64(step * 1_000_000_000 // FPS, "ns"))
            frame = Frame(H=intrinsics.height, W=intrinsics.width, intrinsic=intrinsics.k_matrix, cam_T_world=cam_T_world)
            view_rgb, _, _ = scene._render_RGBD(frame)
            _log_camera(recording, "/world/render", cam_T_world, intrinsics.k_matrix, intrinsics.width, intrinsics.height)
            recording.log(f"{RENDER}/rgb", rr.Image((view_rgb.clamp(0, 1) * 255.0).round().byte().cpu().numpy(), color_model="RGB"))
    code: str = PREPORT_CODE if phase != "post" else "post"
    recording.log(
        "/verify/single_image/run",
        rr.AnyValues(wall_s=time.perf_counter() - started, n_splats=float(len(centers)), stage=stage, code=code),
        static=True,
    )
    print(f"{case.name} {phase}: {len(centers)} splats, {len(trajectory)} trajectory renders in {time.perf_counter() - started:.1f} s")


def _pixels(table: pa.Table, entity: str, archetype: str, dtype: type) -> tuple[list[int], list[ndarray]]:
    """Every logged image of ``entity`` (``Image`` / ``DepthImage``): its ``video_time`` (ns) and its ``h w [c]`` array."""
    buffers: pa.ChunkedArray = table[f"{entity}:{archetype}:buffer"]
    formats: pa.ChunkedArray = table[f"{entity}:{archetype}:format"]
    times: list[int] = []
    images: list[ndarray] = []
    for t, buffer, image_format in zip(table[TIMELINE].cast(pa.int64()).to_pylist(), buffers, formats, strict=True):
        if not buffer.is_valid:
            continue
        flat: pa.Array = buffer.values
        while pa.types.is_list(flat.type) or pa.types.is_large_list(flat.type):
            flat = flat.flatten()
        spec: dict = image_format.as_py()[0]
        times.append(int(t))
        image: ndarray = np.frombuffer(flat.to_numpy(zero_copy_only=False).tobytes(), dtype=dtype).reshape(spec["height"], spec["width"], -1)
        images.append(image[..., 0] if image.shape[-1] == 1 else image)
    return times, images


def _read(golden: Path) -> pa.Table:
    with rr.server.Server(host="127.0.0.1", datasets={"golden": [str(golden)]}) as server:
        dataset: DatasetEntry = server.client().get_dataset("golden")  # an entry must outlive its queries
        return dataset.filter_contents([f"{INPUT}/**", f"{RENDER}/**", "/world/splats"]).reader(index=TIMELINE).to_arrow_table()


def score(case: Case, golden: Path) -> dict[str, float]:
    """Banded: ``input_psnr_db`` (the scene rendered at the input camera vs the processed input image, all pixels),
    ``input_depth_absrel`` (that render's depth vs MoGe's on MoGe-valid pixels) and ``novel_rgb_mae`` (trajectory renders
    vs the case's ``pre`` golden at the same steps, mean |difference| in 0-255 levels; ``pre`` itself scores 0).
    Reported, not banded: ``n_splats`` and ``novel_frames``."""
    table: pa.Table = _read(golden)
    [rgb] = _pixels(table, f"{INPUT}/rgb", "Image", np.uint8)[1]
    [render] = _pixels(table, f"{INPUT}/render_rgb", "Image", np.uint8)[1]
    [depth] = _pixels(table, f"{INPUT}/depth", "DepthImage", np.float32)[1]
    [render_depth] = _pixels(table, f"{INPUT}/render_depth", "DepthImage", np.float32)[1]
    [valid_u8] = _pixels(table, f"{INPUT}/valid", "Image", np.uint8)[1]
    mse: float = float(np.mean((render.astype(np.float64) - rgb.astype(np.float64)) ** 2))
    valid: Bool[ndarray, "h w"] = (valid_u8 > 0) & np.isfinite(depth) & (depth > 0) & np.isfinite(render_depth)
    absrel: float = float(np.mean(np.abs(render_depth[valid] - depth[valid]) / depth[valid])) if valid.any() else float("nan")

    reference: Path = golden if golden.with_suffix("").name == f"{case.name}.pre" else golden_path("vistadream", "single_image", case.name, "pre")
    if not reference.exists():
        raise FileNotFoundError(f"{reference}: the pre-port reference golden of case {case.name!r} is missing (run --phase pre first)")
    ours: dict[int, ndarray] = dict(zip(*_pixels(table, f"{RENDER}/rgb", "Image", np.uint8), strict=True))
    theirs: dict[int, ndarray] = (
        ours if reference == golden else dict(zip(*_pixels(_read(reference), f"{RENDER}/rgb", "Image", np.uint8), strict=True))
    )
    common: list[int] = sorted(ours.keys() & theirs.keys())
    errors: list[float] = [
        float(np.mean(np.abs(ours[t].astype(np.float64) - theirs[t].astype(np.float64)))) for t in common if ours[t].shape == theirs[t].shape
    ]
    centers: pa.ChunkedArray = table["/world/splats:GaussianSplats3D:centers"]
    n_splats: int = sum(len(c.values) for c in centers if c.is_valid)
    return {
        "input_psnr_db": 10.0 * float(np.log10(255.0**2 / mse)) if mse > 0 else float("inf"),
        "input_depth_absrel": absrel,
        "novel_rgb_mae": float(np.mean(errors)) if errors and len(errors) == len(theirs) == len(ours) else float("nan"),
        "n_splats": float(n_splats),
        "novel_frames": float(len(ours)),
    }
