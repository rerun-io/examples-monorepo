"""Full-segment accelerated inference producing a calibrated, compact layer."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray
from rerun.catalog import DatasetEntry
from rerun.chunk import RrdReader
from serde.json import from_json, to_json
from simplecv.catalog_video import CatalogVideo, catalog_codec, catalog_keyframes, packet_views
from simplecv.catalog_video_codec import wrap_mp4
from torchcodec.decoders import VideoDecoder
from trtkit import TensorRtBackendConfig

from posekit.apis.egohand_comparison import EgoCamera, project_ego
from posekit.apis.egohand_selfcap import SEGMENT, VIDEO
from posekit.apis.egohand_world import MeshLod, camera_world_poses, make_mesh_lod, mesh_normals, write_world_layer
from posekit.models.egohand import EgoHandConfig, EgoHandDetectorConfig, HandDetections, resolve_hand_identities
from posekit.models.egohand_geometry import OAKCamera
from posekit.predictions import BoxDetections


@dataclass(frozen=True, slots=True)
class LayerConfig:
    """Local artifacts and explicit identity source for the single allowed segment."""

    root: Path = Path("/tmp/egohand/world-layer")
    """Inference streams, model metadata and final RRD."""
    assets: Path = Path("/tmp/egohand")
    """Released weights and cached acceleration artifacts."""
    identity_rrd: Path = Path("/tmp/selfcap-dev/c4/out/keypoints") / f"{SEGMENT}.rrd"
    """Existing layer whose application and recording identifiers are reused."""


def generate_layer(dataset: DatasetEntry, cameras: list[EgoCamera], config: LayerConfig) -> Path:
    """Infer one dev segment, transform to world, project and write a layer.

    Camera records must come from selfcap-curate read_ego_cameras. Registration
    is separate: this function only reads the catalog and writes local files.
    """
    if dataset.name != "selfcap-post-dev":
        raise ValueError("This experiment only reads selfcap-post-dev")
    config.root.mkdir(parents=True, exist_ok=True)
    reference = RrdReader(config.identity_rrd).recordings()[0]
    if reference.recording_id != SEGMENT:
        raise ValueError("Layer identity does not match the requested segment")
    output: Path = config.root / f"{SEGMENT}.rrd"
    parts: dict[str, Path] = {side: config.root / f"{side}.parquet" for side in ("left", "right")}
    lods: dict[str, MeshLod] = {}
    if (config.root / "inference.done").exists():
        lods = {side: from_json(MeshLod, (config.root / f"{side}-lod.json").read_text()) for side in parts}
    else:
        cache: Path = config.root / "rgb.parquet"
        if not cache.exists():
            pq.write_table(dataset.filter_segments(SEGMENT).filter_contents([VIDEO]).reader(index="video_time").to_arrow_table(), cache)
        table: pa.Table = pq.read_table(cache)
        samples: pa.Table = table.filter(table[f"{VIDEO}:VideoStream:sample"].is_valid())
        times: Int64[ndarray, "n"] = samples["video_time"].to_numpy().view(np.int64)
        keys: list[bool] = catalog_keyframes(samples[f"{VIDEO}:VideoStream:is_keyframe"])
        first: int = next(i for i, key in enumerate(keys) if key)
        video: CatalogVideo = CatalogVideo(
            times[first:],
            packet_views(samples[f"{VIDEO}:VideoStream:sample"])[first:],
            keys[first:],
            catalog_codec(samples[f"{VIDEO}:VideoStream:codec"], VIDEO),
        )
        times = video.t_ns
        times_s: Float64[ndarray, "n"] = times.astype(np.float64) * 1e-9
        decoder: VideoDecoder = VideoDecoder(
            wrap_mp4(video.samples, video.keyframes, 24, video.codec), device="cuda", dimension_order="NHWC", seek_mode="exact"
        )
        rgb: EgoCamera = next(camera for camera in cameras if camera.cam == 0)
        camera: OAKCamera = OAKCamera(rgb.K, rgb.distortion, *rgb.size)
        poses, pose_valid = camera_world_poses(rgb, times_s)
        detector = EgoHandDetectorConfig(
            config.assets / "weights/detector/hand_detector.onnx",
            assign_identity=False,
            backend=TensorRtBackendConfig(max_batch_size=8, opt_batch_size=4, cache_dir=config.assets / "engines"),
        ).setup()
        boxes, scores, indices, sides = [], [], [], []
        for start in range(0, len(times), 16):
            stop: int = min(start + 16, len(times))
            frames = decoder.get_frames_at(list(range(start, stop))).data
            detected = detector(frames)
            boxes.append(detected.boxes.xyxy)
            scores.append(detected.boxes.scores)
            indices.append(detected.boxes.frame_indices + start)
            sides.extend(detected.sides)
        detections: HandDetections = resolve_hand_identities(
            HandDetections(BoxDetections(torch.cat(boxes), torch.cat(scores), torch.cat(indices)), tuple(sides)), len(times)
        )
        predictor = EgoHandConfig(
            config.assets / "weights/egohand/egohand_s060000.ckpt",
            config.assets / "weights/soma",
            batch_size=32,
            gpu_crops=True,
            backbone_onnx=config.assets / "dinov3-bf16.onnx",
            backbone_backend=TensorRtBackendConfig(max_batch_size=32, opt_batch_size=32, cache_dir=config.assets / "engines"),
        ).setup()
        vector = pa.list_(pa.float32(), 3)
        pixel = pa.list_(pa.float32(), 2)
        schema: pa.Schema = pa.schema(
            [
                ("video_time", pa.int64()),
                *[
                    (f"hand_{name}", pa.list_(kind))
                    for name, kind in (
                        ("vertices", vector),
                        ("normals", vector),
                        ("joints", vector),
                        ("rgb", pixel),
                        ("mono_left", pixel),
                        ("mono_right", pixel),
                    )
                ],
            ]
        )
        writers = {side: pq.ParquetWriter(path, schema, compression="zstd") for side, path in parts.items()}
        try:
            for start in range(0, len(times), 16):
                stop = min(start + 16, len(times))
                frames = decoder.get_frames_at(list(range(start, stop))).data
                selected = torch.where((detections.boxes.frame_indices >= start) & (detections.boxes.frame_indices < stop))[0].tolist()
                local = HandDetections(
                    BoxDetections(
                        detections.boxes.xyxy[selected], detections.boxes.scores[selected], detections.boxes.frame_indices[selected] - start
                    ),
                    tuple(detections.sides[i] for i in selected),
                )
                result = predictor(frames, local, camera)
                assert result.keypoints.xyz_camera is not None
                for side in parts:
                    joints: Float32[ndarray, "b 25 3"] = np.full((stop - start, 25, 3), np.nan, np.float32)
                    rows = [
                        {"video_time": int(t), **{f"hand_{name}": [] for name in ("vertices", "normals", "joints", "rgb", "mono_left", "mono_right")}}
                        for t in times[start:stop]
                    ]
                    for hand, label in enumerate(result.sides):
                        if label != side:
                            continue
                        index: int = int(result.keypoints.frame_indices[hand])
                        if not pose_valid[start + index]:
                            continue
                        xyz: Float32[ndarray, "25 3"] = result.keypoints.xyz_camera[hand].cpu().numpy()
                        vertices: Float32[ndarray, "v 3"] = result.vertices[hand].cpu().numpy()
                        if side not in lods:
                            lods[side] = make_mesh_lod(vertices, predictor.predictor.model.soma_hand.faces(side).astype(np.uint32))
                            (config.root / f"{side}-lod.json").write_text(to_json(lods[side]))
                        transform: Float64[ndarray, "4 4"] = poses[start + index]
                        world: Float32[ndarray, "v 3"] = (lods[side].apply(vertices) @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)
                        joints[index] = xyz @ transform[:3, :3].T + transform[:3, 3]
                        rows[index].update(
                            hand_vertices=world.tolist(),
                            hand_normals=mesh_normals(world, lods[side].faces).tolist(),
                            hand_joints=joints[index].tolist(),
                            hand_rgb=result.keypoints.xy[hand].cpu().numpy().tolist(),
                        )
                    for mono, field in ((1, "mono_left"), (2, "mono_right")):
                        projected = project_ego(next(cam for cam in cameras if cam.cam == mono), times_s[start:stop], joints)
                        for index, row in enumerate(rows):
                            if row["hand_joints"]:
                                row[f"hand_{field}"] = projected[index].tolist()
                    writers[side].write_table(pa.Table.from_pylist(rows, schema=schema))
                if start % 160 == 0:
                    print(f"world layer: {start}/{len(times)} frames", flush=True)
        finally:
            for writer in writers.values():
                writer.close()
        (config.root / "inference.done").write_text(f"{len(times)} frames\n")
    write_world_layer(parts, lods, output, reference.application_id, reference.recording_id)
    return output
