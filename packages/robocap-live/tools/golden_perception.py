"""Golden data for robocap-live's hand perception port: handtrack's own code on fixed s66 inputs.

Runs handtrack @ 54eaf309 (the code the Rust port follows) on the CPU, from the repo root in the root pixi.toml's ``handtrack`` env:

    pixi run -e handtrack --frozen \
        python packages/robocap-live/tools/golden_perception.py --inputs <inputs dir> --detnet-weights <DetNet-F .weights.pt> \
        --keynet-weights <KeyNet-F .weights.pt> --out packages/robocap-live/crates/robocap-live/tests/data/perception

``--inputs`` holds two s66 framesets from the s66 dump (``f<index>_c<camera>.u8``, 1920x1080 luma of cameras 0..3), the dump's
``rig.json`` and the reference run's JSONL rows of those framesets and the ones before them (``reference_rows.json``: landmarks
of tracked hands, world_from_rig). They were cut from a full s66 dump (``robocap-live-dump/1``, frameset indices 448 and 1336:
each ``frames.bin`` record's 80-byte header, then its present cameras' 1920*1080 bytes; keep cameras 0..3) and from the
reference run's ``s66-cams4.jsonl`` (rows 447, 448, 1335, 1336 keyed by index). Everything goes through handtrack's functions: ``project``, ``unproject``, the robocap_track.py
``BarLetterbox`` (both its antialiased-bilinear small image and the runtime's 3x3 area mean), DetNet-F + ``decode_detections``,
``decode_heatmaps``/``decode_distance``, and ``PerspectiveKeyNetEstimator`` with the real KeyNet weights (its model wrapped to
record the crops, keypoint inputs and raw outputs). The estimator's planning poses are fixed world landmarks (the reference
fit of the previous frame), fed through a patched ``landmarks`` so the estimator code runs unchanged.

Frames are stored only inside rectangles that cover every crop's footprint (zeros elsewhere); the script checks that the
estimator gives bit-identical crops on those masked frames, so the Rust test samples exactly what Python sampled. Crops are
stored as u16 (value * 65535, error < 8e-6), everything else as f32 or inline JSON.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import handtrack.keynet_perspective as keynet_perspective
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from handtrack.geometry.camera import CameraRig, project
from handtrack.geometry.letterbox import NET_HEIGHT, NET_WIDTH, Letterbox
from handtrack.hand.pose import generic_hand_model
from handtrack.keynet_perspective import PerspectiveKeyNetEstimator
from handtrack.labels.heatmaps import decode_distance, decode_heatmaps
from handtrack.labels.perspective import CropCameras, crop_rays, unproject
from handtrack.models.detnet import DetNetF, decode_detections
from handtrack.models.keynet import KeyNetF, KeyNetOutput, keynet_for_state
from handtrack.pipeline import load_weights, read_state
from handtrack.tracker import CropRequest
from jaxtyping import Float32, UInt8
from numpy import ndarray
from torch import Tensor

WIDTH: int = 1920
HEIGHT: int = 1080
CAMERAS: int = 6
PHI: float = 0.9701733589172363
"""s66's native-fit phi (live_reference.S66_NATIVE_PHI)."""


@dataclass(frozen=True, slots=True)
class Config:
    inputs: Path
    """The cut s66 framesets, ``rig.json`` and ``reference_rows.json`` (see the module docstring)."""
    detnet_weights: Path
    """DetNet-F weights (the golden files use the 2026-09-29 demo checkpoint ``detnet-demo-0929.weights.pt``; the manifest
    records the sha256)."""
    keynet_weights: Path
    """KeyNet-F weights (the golden files use the ``keynet-strong-pinch`` run's ``best.weights.pt``; the manifest records the
    sha256)."""
    out: Path = Path("packages/robocap-live/crates/robocap-live/tests/data/perception")
    frames: tuple[int, ...] = (448, 1336)


@dataclass(frozen=True, slots=True)
class BarLetterbox(Letterbox):
    """robocap_track.py's BarLetterbox (with live_reference.py's ``area`` variant): 1/3 resize into rows 60..420."""

    pad_y: float = 0.0
    small: str = "bilinear"

    def __post_init__(self) -> None:
        pass

    def to_net(self, uv: Float32[Tensor, "*b 2"]) -> Float32[Tensor, "*b 2"]:
        scaled: Float32[Tensor, "*b 2"] = (uv + 0.5) * self.scale - 0.5
        return torch.stack((scaled[..., 0] + self.pad_x, scaled[..., 1] + self.pad_y), dim=-1)

    def from_net(self, uv: Float32[Tensor, "*b 2"]) -> Float32[Tensor, "*b 2"]:
        return (torch.stack((uv[..., 0] - self.pad_x, uv[..., 1] - self.pad_y), dim=-1) + 0.5) / self.scale - 0.5

    def apply(self, image: UInt8[Tensor, "*b h w"]) -> UInt8[Tensor, "*b 480 640"]:
        height: int = round(self.source_height * self.scale)
        width: int = round(self.source_width * self.scale)
        flat: Float32[Tensor, "n 1 h w"] = image.reshape(-1, 1, *image.shape[-2:]).float()
        resized_float: Float32[Tensor, "n 1 sh sw"] = (
            F.avg_pool2d(flat, kernel_size=round(1.0 / self.scale)) if self.small == "area"
            else F.interpolate(flat, size=(height, width), mode="bilinear", antialias=True, align_corners=False)
        )
        resized: UInt8[Tensor, "*b sh sw"] = resized_float.round().to(torch.uint8).reshape(*image.shape[:-2], height, width)
        return F.pad(resized, (int(self.pad_x), NET_WIDTH - width - int(self.pad_x), int(self.pad_y), NET_HEIGHT - height - int(self.pad_y)))


LETTERBOX: BarLetterbox = BarLetterbox(WIDTH, HEIGHT, False, 1.0 / 3.0, 0.0, 60.0)
LETTERBOX_AREA: BarLetterbox = BarLetterbox(WIDTH, HEIGHT, False, 1.0 / 3.0, 0.0, 60.0, "area")


@dataclass(frozen=True, slots=True)
class Fixed:
    """A planning 'pose' that is already its 21 world landmarks (the patched ``landmarks`` returns them)."""

    points: Float32[Tensor, "21 3"]


_landmarks = keynet_perspective.landmarks
keynet_perspective.landmarks = lambda model, pose, side: pose.points if isinstance(pose, Fixed) else _landmarks(model, pose, side)


class Recorder(torch.nn.Module):
    """KeyNet-F that remembers its inputs and outputs."""

    def __init__(self, model: KeyNetF) -> None:
        super().__init__()
        self.model: KeyNetF = model
        self.calls: list[tuple[Tensor, Tensor, KeyNetOutput]] = []

    def forward(self, crops: Float32[Tensor, "n 1 96 96"], features: Float32[Tensor, "n 63"]) -> KeyNetOutput:
        output: KeyNetOutput = self.model(crops, features)
        self.calls.append((crops.clone(), features.clone(), output))
        return output


def f32(values: ndarray | Tensor) -> list:
    """Nested lists of the float32 values as exact Python floats."""
    array: ndarray = values.detach().cpu().numpy() if isinstance(values, Tensor) else np.asarray(values)
    return array.astype(np.float32).astype(np.float64).tolist()


def finite_json(value: object) -> object:
    """``value`` with every NaN float as None (JSON has no NaN; the Rust side reads null as NaN)."""
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [finite_json(item) for item in value]
    return value


def camera_rig(rig: dict) -> CameraRig:
    cameras: list[dict] = rig["cameras"]
    return CameraRig(
        names=tuple(f"/world/rig_00/cam_{camera:02d}" for camera in range(len(cameras))),
        image_size=torch.tensor([[float(c["width"]), float(c["height"])] for c in cameras], dtype=torch.float32),
        cam_from_rig=torch.tensor([c["cam_from_rig"] for c in cameras], dtype=torch.float32),
        focal=torch.tensor([c["focal"] for c in cameras], dtype=torch.float32),
        principal=torch.tensor([c["principal"] for c in cameras], dtype=torch.float32),
        fisheye62=torch.tensor([c["fisheye62"] for c in cameras], dtype=torch.float32),
    )


def rig_as_float32(rig: dict) -> dict:
    """rig.json with every number rounded to float32, as handtrack's CameraRig holds it."""
    out: dict = json.loads(json.dumps(rig))
    for camera in out["cameras"]:
        for key in ("cam_from_rig", "focal", "principal", "fisheye62"):
            camera[key] = f32(np.asarray(camera[key]))
    return out


@dataclass(slots=True)
class Call:
    """One estimator call: its requests and what came out."""

    name: str
    frame: int
    world_from_rig: Float32[Tensor, "4 4"]
    views: list[dict] = field(default_factory=list)


def footprint(rig: CameraRig, cameras: CropCameras, camera: int) -> tuple[int, int, int, int] | None:
    """Integer pixel rectangle (x0, y0, x1, y1), exclusive end, holding every bilinear tap of a crop."""
    rays: Float32[Tensor, "n p 3"] = crop_rays(cameras)
    pixels: Float32[Tensor, "n p 2"] = project(rig.select([camera]), rays[:, None])[:, 0]
    ok = (rays[..., 2] > 1e-6) & torch.isfinite(pixels).all(dim=-1)
    if not bool(ok.any()):
        return None
    chosen: Float32[Tensor, "k 2"] = pixels[ok]
    x0: int = max(int(math.floor(float(chosen[:, 0].min()))) - 2, 0)
    y0: int = max(int(math.floor(float(chosen[:, 1].min()))) - 2, 0)
    x1: int = min(int(math.ceil(float(chosen[:, 0].max()))) + 3, WIDTH)
    y1: int = min(int(math.ceil(float(chosen[:, 1].max()))) + 3, HEIGHT)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def main(config: Config) -> None:
    torch.set_num_threads(4)
    out: Path = config.out
    out.mkdir(parents=True, exist_ok=True)
    rig_json: dict = rig_as_float32(json.loads((config.inputs / "rig.json").read_text()))
    (out / "rig.json").write_text(json.dumps(rig_json, indent=1))
    rig: CameraRig = camera_rig(rig_json)
    rows: dict = json.loads((config.inputs / "reference_rows.json").read_text())
    frames: dict[int, list[Tensor]] = {}
    for frame in config.frames:
        frames[frame] = [torch.from_numpy(np.fromfile(config.inputs / f"f{frame}_c{c}.u8", dtype=np.uint8).reshape(HEIGHT, WIDTH)) if c < 4
                         else torch.zeros((HEIGHT, WIDTH), dtype=torch.uint8) for c in range(CAMERAS)]
    manifest: dict = {"format": "robocap-live-perception-golden/1", "handtrack_commit": "54eaf309", "phi": PHI, "frames": list(config.frames)}

    # Cameras: project fixed camera-frame points (in front, off-axis, behind) and unproject fixed pixels.
    points: Float32[Tensor, "k 3"] = torch.tensor([[0.0, 0.0, 1.0], [0.1, -0.05, 0.4], [-0.3, 0.2, 0.5], [0.5, 0.4, 0.3], [-0.6, -0.3, 0.1],
                                                    [0.2, 0.1, -0.3], [1.0, 0.0, 0.0], [0.02, 0.03, 0.25], [-0.15, 0.25, 0.35]], dtype=torch.float32)
    pixels: Float32[Tensor, "q 2"] = torch.tensor([[960.0, 540.0], [100.0, 80.0], [1800.0, 1000.0], [500.0, 700.0], [1400.0, 300.0], [10.0, 540.0],
                                                    [1919.0, 1079.0], [956.2, 525.4]], dtype=torch.float32)
    manifest["camera"] = {
        "points_cam": f32(points),
        "project": [f32(project(rig.select([c]), points[None, None])[0, 0]) for c in range(CAMERAS)],
        "pixels": f32(pixels),
        "unproject": [f32(unproject(rig.select([c]), pixels)) for c in range(CAMERAS)],
    }
    uv: Float32[Tensor, "k 2"] = torch.tensor([[0.0, 0.0], [959.5, 539.5], [1919.0, 1079.0], [-3.0, 2000.0]], dtype=torch.float32)
    manifest["letterbox_points"] = {"full": f32(uv), "to_net": f32(LETTERBOX.to_net(uv)), "from_net_of_full": f32(LETTERBOX.from_net(uv))}

    # DetNet: real letterboxed frames (antialiased bilinear, as the Python runs fed it) -> raw -> decode.
    detnet: DetNetF = DetNetF()
    detnet_sha: str = load_weights(detnet, config.detnet_weights)
    detnet.eval()
    letterboxed: UInt8[Tensor, "b 480 640"] = torch.stack([LETTERBOX.apply(frames[frame][c]) for frame in config.frames for c in range(4)])
    with torch.inference_mode():
        raw = detnet(letterboxed[:, None].float() / 255.0)
    synthetic_center: Float32[Tensor, "s 2 2"] = torch.tensor([[[0.5, 0.5], [0.1, 0.9]], [[-0.1, 1.2], [0.33, 0.66]]])
    synthetic_radius: Float32[Tensor, "s 2"] = torch.tensor([[0.05, 0.2], [0.0, -0.01]])
    synthetic_logit: Float32[Tensor, "s 2"] = torch.tensor([[0.0, 1.3862944], [-30.0, 30.0]])
    center: Float32[Tensor, "n 2 2"] = torch.cat([raw.center, synthetic_center])
    radius: Float32[Tensor, "n 2"] = torch.cat([raw.radius, synthetic_radius])
    logit: Float32[Tensor, "n 2"] = torch.cat([raw.presence_logit, synthetic_logit])
    decoded = decode_detections(type(raw)(center, radius, logit), 0.8)
    manifest["detnet"] = {"weights_sha256": detnet_sha, "inputs": [f"f{frame}_c{c}" for frame in config.frames for c in range(4)] + ["synthetic0", "synthetic1"],
                          "center": f32(center), "radius": f32(radius), "presence_logit": f32(logit), "circle": f32(decoded.circle),
                          "probability": f32(decoded.probability), "present_0_8": decoded.present.tolist(), "box": f32(decoded.box)}
    circles: dict[tuple[int, int], Float32[Tensor, "2 3"]] = {(frame, c): decoded.circle[i * 4 + c] for i, frame in enumerate(config.frames) for c in range(4)}
    print("detnet probability", decoded.probability[:8].tolist(), flush=True)

    # Heatmap decoding on fixed synthetic inputs: random, zeros, ties, edge peaks, negatives.
    generator: torch.Generator = torch.Generator().manual_seed(7)
    heat: Float32[Tensor, "3 21 18 18"] = torch.rand((3, 21, 18, 18), generator=generator) ** 8
    heat[0, 0] = 0.0
    heat[0, 1] = 0.0
    heat[0, 1, 4, 5] = 0.7
    heat[0, 1, 9, 2] = 0.7  # a tie: torch's max picks one of them
    heat[0, 2, 0, 0] = 2.0  # corner peak
    heat[0, 3, 17, 9] = 2.0  # bottom edge peak
    heat[1, 4] = -heat[1, 4]  # all non-positive
    axis: Float32[Tensor, "18"] = torch.arange(18, dtype=torch.float32)
    heat[2, 5] = torch.exp(-((axis[:, None] - 6.3) ** 2 + (axis[None, :] - 11.8) ** 2) / 2)  # an exact Gaussian
    distance: Float32[Tensor, "3 21 18"] = torch.rand((3, 21, 18), generator=generator) ** 4
    distance[0, 0] = 0.0
    distance[0, 1, 0] = 3.0
    distance[0, 2, 17] = 3.0
    distance[2, 3] = torch.exp(-((axis - 4.6) ** 2) / 2)
    points_crop, confidence = decode_heatmaps(heat)
    (out / "heatmaps_f32.bin").write_bytes(heat.numpy().astype("<f4").tobytes())
    (out / "distance_f32.bin").write_bytes(distance.numpy().astype("<f4").tobytes())
    manifest["heatmaps"] = {"heatmaps": "heatmaps_f32.bin [3,21,18,18]", "distance": "distance_f32.bin [3,21,18]",
                            "points_crop": f32(points_crop), "confidence": f32(confidence), "d_rel_mm": f32(decode_distance(distance))}

    # KeyNet on perspective crops through the estimator itself.
    state, keynet_sha = read_state(config.keynet_weights)
    keynet: KeyNetF = keynet_for_state(state)
    keynet.load_state_dict(state)
    recorder: Recorder = Recorder(keynet.eval())
    estimator: PerspectiveKeyNetEstimator = PerspectiveKeyNetEstimator(recorder, rig, (LETTERBOX,) * CAMERAS, (0.0,) * CAMERAS, generic_hand_model(), PHI)

    def landmarks_of(index: int, side: int) -> Float32[Tensor, "21 3"]:
        return torch.tensor(rows[str(index)]["hands"][side]["landmarks"], dtype=torch.float32)

    def pose_of(index: int) -> Float32[Tensor, "4 4"]:
        return torch.tensor(rows[str(index)]["world_from_rig"], dtype=torch.float32).reshape(4, 4)

    first, second = config.frames
    world_448: Float32[Tensor, "4 4"] = pose_of(first)
    world_1336: Float32[Tensor, "4 4"] = pose_of(second)
    # Landmarks behind camera 2 at frame 1336: the reference left hand mirrored through camera 2's image plane.
    left: Float32[Tensor, "21 3"] = landmarks_of(second - 1, 0)
    cam_from_rig: Float32[Tensor, "4 4"] = rig.cam_from_rig[2]
    rig_points: Float32[Tensor, "21 3"] = (left - world_1336[:3, 3]) @ world_1336[:3, :3]
    cam_points: Float32[Tensor, "21 3"] = rig_points @ cam_from_rig[:3, :3].T + cam_from_rig[:3, 3]
    cam_points[:, 2] = -cam_points[:, 2].abs()
    behind_rig: Float32[Tensor, "21 3"] = (cam_points - cam_from_rig[:3, 3]) @ cam_from_rig[:3, :3]
    behind_world: Float32[Tensor, "21 3"] = behind_rig @ world_1336[:3, :3].T + world_1336[:3, 3]
    nan_circle: Float32[Tensor, "3"] = torch.full((3,), math.nan)
    plans: list[tuple[str, int, Float32[Tensor, "4 4"], tuple, list[tuple[int, int, Float32[Tensor, "3"]]]]] = [
        # (name, frame, world_from_rig, poses per side, views (camera, side, circle))
        ("tracked_448", first, world_448, (Fixed(landmarks_of(first - 1, 0)), Fixed(landmarks_of(first - 1, 1))),
         [(0, 0, circles[(first, 0)][0]), (1, 0, circles[(first, 1)][0]), (0, 1, circles[(first, 0)][1]), (1, 1, circles[(first, 1)][1])]),
        ("acquire_448", first, world_448, (None, None), [(0, 0, circles[(first, 0)][0]), (0, 1, circles[(first, 0)][1])]),
        ("tracked_1336", second, world_1336, (Fixed(landmarks_of(second - 1, 0)), Fixed(landmarks_of(second - 1, 1))),
         [(2, 0, circles[(second, 2)][0]), (3, 0, circles[(second, 3)][0]), (0, 1, circles[(second, 0)][1]), (1, 1, circles[(second, 1)][1])]),
        ("edge_1336", second, world_1336, (Fixed(behind_world), None),
         [(2, 0, circles[(second, 2)][0]), (1, 1, circles[(second, 1)][1]), (5, 1, nan_circle)]),
    ]
    calls: list[dict] = []
    crops_all: list[ndarray] = []
    raw_all: list[ndarray] = []
    rects: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for name, frame, world_from_rig, poses, views in plans:
        request: CropRequest = CropRequest(
            camera=torch.tensor([v[0] for v in views]), side=torch.tensor([v[1] for v in views]), crop_from_net=torch.zeros((len(views), 3, 3)),
            keypoint_input=torch.zeros((len(views), 63)), poses=poses, world_from_rig=world_from_rig, circles=torch.stack([v[2] for v in views]),
            native_images=tuple(frames[frame]),
        )
        planned: list[tuple[CropCameras, Tensor]] = [estimator._cameras(request, v[0], v[1], i) for i, v in enumerate(views)]
        for (cameras, _), (camera, _, _) in zip(planned, views, strict=True):
            usable: bool = estimator._usable(cameras)
            used: CropCameras = cameras if usable else CropCameras(torch.eye(3)[None], torch.ones(1), cameras.mirror)
            rect = footprint(rig, used, camera)
            if rect is not None and camera < 4:  # cameras 4 and 5 are zero frames here
                rects.setdefault((frame, camera), []).append(rect)
        recorder.calls.clear()
        estimate = estimator(torch.zeros((CAMERAS, 480, 640), dtype=torch.uint8), frame, request)
        crops, features, output = recorder.calls[0]
        crops_all.append(crops[:, 0].numpy())
        flat: list[Tensor] = [output.heatmaps.reshape(len(views), -1), output.distance.reshape(len(views), -1), output.presence_logit[:, None],
                              output.pinch_logit[:, None] if output.pinch_logit is not None else torch.full((len(views), 1), math.nan)]
        raw_all.append(torch.cat(flat, dim=1).numpy())
        calls.append({
            "name": name, "frame": frame, "world_from_rig": f32(world_from_rig), "phi": PHI,
            "views": [{"camera": v[0], "side": v[1], "circle_net": f32(v[2]),
                       "landmarks_world": None if poses[v[1]] is None else f32(poses[v[1]].points)} for v in views],
            "crop_rotation": [f32(p[0].rotation[0]) for p in planned], "crop_focal": [float(p[0].focal[0]) for p in planned],
            "crop_mirror": [bool(p[0].mirror[0]) for p in planned], "keypoint_input": f32(features),
            "points_net": f32(estimate.points_net), "d_rel_mm": f32(estimate.d_rel_mm), "presence": f32(estimate.presence),
            "confidence": f32(estimate.confidence), "pinch": None if estimate.pinch is None else f32(estimate.pinch),
        })
        print(name, "presence", estimate.presence.tolist(), "focal", [round(float(p[0].focal[0]), 3) for p in planned], flush=True)

    # Masked frames: keep only the crop footprints; the estimator must sample identical crops from them. Stored per (frame, camera)
    # as its rectangles plus the masked pixels in row-major order.
    masked: dict[int, list[Tensor]] = {frame: [torch.zeros((HEIGHT, WIDTH), dtype=torch.uint8) for _ in range(CAMERAS)] for frame in config.frames}
    stored: list[dict] = []
    window_bytes: list[bytes] = []
    offset: int = 0
    for (frame, camera), boxes in sorted(rects.items()):
        mask: ndarray = np.zeros((HEIGHT, WIDTH), dtype=bool)
        for x0, y0, x1, y1 in boxes:
            mask[y0:y1, x0:x1] = True
        masked[frame][camera][torch.from_numpy(mask)] = frames[frame][camera][torch.from_numpy(mask)]
        data: bytes = frames[frame][camera].numpy()[mask].tobytes()
        stored.append({"frame": frame, "camera": camera, "rects": [list(box) for box in boxes], "offset": offset, "length": len(data)})
        window_bytes.append(data)
        offset += len(data)
    (out / "frame_pixels_u8.bin").write_bytes(b"".join(window_bytes))
    manifest["frame_pixels"] = {"file": "frame_pixels_u8.bin", "order": "per entry: pixels inside the union of rects, row-major", "entries": stored}
    for (name, frame, world_from_rig, poses, views), crops in zip(plans, crops_all, strict=True):
        request = CropRequest(
            camera=torch.tensor([v[0] for v in views]), side=torch.tensor([v[1] for v in views]), crop_from_net=torch.zeros((len(views), 3, 3)),
            keypoint_input=torch.zeros((len(views), 63)), poses=poses, world_from_rig=world_from_rig, circles=torch.stack([v[2] for v in views]),
            native_images=tuple(masked[frame]),
        )
        recorder.calls.clear()
        estimator(torch.zeros((CAMERAS, 480, 640), dtype=torch.uint8), frame, request)
        again: ndarray = recorder.calls[0][0][:, 0].numpy()
        if not np.array_equal(again, crops):
            raise ValueError(f"{name}: crops from the masked frames differ by {np.abs(again - crops).max()}")
    crops_cat: ndarray = np.concatenate(crops_all)
    (out / "crops_u16.bin").write_bytes(np.round(crops_cat * 65535.0).astype("<u2").tobytes())
    raw_cat: ndarray = np.concatenate(raw_all).astype("<f4")
    (out / "keynet_raw_f32.bin").write_bytes(raw_cat.tobytes())
    manifest["keynet"] = {"weights_sha256": keynet_sha, "crops": f"crops_u16.bin [{len(crops_cat)},96,96] = round(crop * 65535)",
                          "raw": f"keynet_raw_f32.bin [{len(raw_cat)},{raw_cat.shape[1]}] = heatmaps 21*18*18, distance 21*18, presence_logit, pinch_logit",
                          "calls": calls}

    # Letterbox of two masked frames (both small-image variants), stored over the bounding box of what can be nonzero.
    letterbox: list[dict] = []
    payload: list[bytes] = []
    offset = 0
    for frame, camera in ((first, 0), (second, 2)):
        for variant, box in (("area", LETTERBOX_AREA), ("bilinear", LETTERBOX)):
            image: ndarray = box.apply(masked[frame][camera]).numpy()
            rows_nz: ndarray = np.nonzero(image.any(axis=1))[0]
            cols_nz: ndarray = np.nonzero(image.any(axis=0))[0]
            top, bottom, left_x, right_x = int(rows_nz.min()), int(rows_nz.max()) + 1, int(cols_nz.min()), int(cols_nz.max()) + 1
            data = image[top:bottom, left_x:right_x].tobytes()
            letterbox.append({"frame": frame, "camera": camera, "variant": variant, "box": [left_x, top, right_x, bottom], "offset": offset})
            payload.append(data)
            offset += len(data)
    (out / "letterbox_u8.bin").write_bytes(b"".join(payload))
    manifest["letterbox"] = {"file": "letterbox_u8.bin", "entries": letterbox}

    # Crop maps: the source pixel of every 11th crop pixel for a tracked crop and for the placeholder (identity, focal 1).
    tracked_cameras: CropCameras = estimator._cameras(CropRequest(
        camera=torch.tensor([0]), side=torch.tensor([1]), crop_from_net=torch.zeros((1, 3, 3)), keypoint_input=torch.zeros((1, 63)),
        poses=(None, Fixed(landmarks_of(first - 1, 1))), world_from_rig=world_448, circles=torch.zeros((1, 3)), native_images=tuple(frames[first])), 0, 1, 0)[0]
    placeholder: CropCameras = CropCameras(torch.eye(3)[None], torch.ones(1), torch.tensor([True]))
    maps: list[dict] = []
    for label, cameras, camera in (("tracked_right_cam0", tracked_cameras, 0), ("placeholder_right_cam3", placeholder, 3)):
        rays: Float32[Tensor, "1 p 3"] = crop_rays(cameras)
        pixels_map: Float32[Tensor, "1 p 2"] = project(rig.select([camera]), rays[:, None])[:, 0]
        ok = (rays[..., 2] > 1e-6) & torch.isfinite(pixels_map).all(dim=-1)
        pick: ndarray = np.arange(0, 96 * 96, 11)
        maps.append({"name": label, "camera": camera, "rotation": f32(cameras.rotation[0]), "focal": float(cameras.focal[0]), "mirror": bool(cameras.mirror[0]),
                     "pixel_index": pick.tolist(), "source": f32(pixels_map[0, pick]), "valid": ok[0, pick].tolist()})
    manifest["crop_maps"] = maps

    (out / "manifest.json").write_text(json.dumps(finite_json(manifest), allow_nan=False))
    total: int = sum(path.stat().st_size for path in out.iterdir())
    print(f"wrote {out}: {total / 1e6:.2f} MB, frame pixels {sum(len(b) for b in window_bytes) / 1e6:.2f} MB, {len(crops_cat)} crops", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
