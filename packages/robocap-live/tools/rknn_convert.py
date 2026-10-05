"""Convert robocap-live's ONNX hand nets to RKNN (RK3588) and score them on the RKNN-Toolkit2 simulator against PyTorch.

Runs in a separate pixi project for Rockchip's RKNN-Toolkit2 (the x86 converter + simulator; the root pixi.toml has no
rknn-toolkit2 env): linux-64, conda-forge ``python 3.11.*`` and ``setuptools <70``, and from PyPI ``rknn-toolkit2 ==2.3.2``,
``torch ==2.4.0`` (CPU wheel index), ``onnx >=1.16.1,<1.18``, ``onnxruntime >=1.17,<1.20``, ``onnxsim`` and ``tyro >=0.9,<0.10``:

    pixi run --manifest-path <rknn-toolkit2 project>/pixi.toml python packages/robocap-live/tools/rknn_convert.py \
        --work-dir <dir> --keynet-cache <dir> --calibration-dir <dir>

Inputs (from ``handtrack/apis/export_nets_onnx.py``): ``<models>/{detnet_pooled_b1,keynet_b1,keynet_b4}.onnx`` and
``<work>/{detnet,keynet}_heldout.npz`` (held-out inputs with their PyTorch FP32 outputs).

Every model takes its image as **u8** (``mean 0, std 255`` is baked into the RKNN model, so the NPU sees u8 / 255 as PyTorch
does); KeyNet's ``keypoints`` stay float32. INT8 models are calibrated on the 2026-09-29 study's sets (256 pooled DetNet frames,
256 KeyNet crops + priors from the training caches) plus, when present, RoboCap s66 sets in ``<work>/s66_{detnet,keynet}_calib.npz``.

Scores are decoded the way handtrack decodes them, against the PyTorch outputs on the same inputs:
- DetNet: centre error px (640 x 480 net frame) and radius error px on the slots PyTorch reports present, presence agreement;
- KeyNet: keypoint error px after ``decode_heatmaps`` (crop px, and net px through the crop's linear part as the earlier study
  scored it), ``decode_distance`` error mm, presence agreement, pinch probability error.

The report goes to ``<models>/rknn_report.json``; per-build simulator outputs to ``<work>/sim/``.
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import tyro
from numpy import ndarray
from rknn.api import RKNN

Net: TypeAlias = Literal["detnet", "keynet"]
Precision: TypeAlias = Literal["int8", "fp16"]
Algorithm: TypeAlias = Literal["normal", "mmse", "kl_divergence"]

HEATMAP: int = 18
CROP: int = 96
DISTANCE_RANGE_MM: float = 130.0
DETNET_OUTPUTS: tuple[str, ...] = ("center", "radius", "presence_logit")
EvalSet: TypeAlias = Literal["heldout", "s66"]
EVAL_FILES: dict[str, str] = {"heldout": "{net}_heldout.npz", "s66": "s66_{net}_heldout.npz"}
"""``heldout``: the training caches' held-out samples (UmeTrack + SHOW3D, with labels). ``s66``: real RoboCap Cap A inputs that
handtrack's tracker fed the nets on the s66 clip (every other sample; the rest calibrate), with PyTorch CPU FP32 outputs."""
KEYNET_OUTPUTS: tuple[str, ...] = ("heatmaps", "distance", "presence_logit", "pinch_logit")


@dataclass(frozen=True, slots=True)
class Build:
    """One RKNN model to build: ``<net>_b<batch>_<precision>[_<algorithm>].rknn``."""

    net: Net
    batch: int
    precision: Precision
    algorithm: Algorithm = "normal"
    """INT8 calibration algorithm (ignored for FP16)."""

    @property
    def name(self) -> str:
        suffix: str = f"_{self.algorithm}" if self.precision == "int8" and self.algorithm != "normal" else ""
        return f"{self.net}_b{self.batch}_{self.precision}{suffix}"

    @property
    def onnx(self) -> str:
        return "detnet_pooled_b1.onnx" if self.net == "detnet" else f"keynet_b{self.batch}.onnx"


DEFAULT_BUILDS: tuple[Build, ...] = (
    Build("detnet", 1, "int8"),
    Build("detnet", 1, "fp16"),
    Build("keynet", 1, "int8"),
    Build("keynet", 1, "int8", "mmse"),
    Build("keynet", 1, "fp16"),
    Build("keynet", 4, "int8"),
    Build("keynet", 4, "fp16"),
)


@dataclass
class Config:
    """Build RKNN models from the exported ONNX graphs and score them on the simulator against PyTorch."""

    work_dir: Path
    """Holds ``*_heldout.npz`` and the optional s66 calibration sets; receives calibration files and simulator outputs."""
    keynet_cache: Path
    """The KeyNet perspective test-samples cache, for ``crop_from_net.npy`` and ``points_crop.npy`` (net-frame errors,
    ground-truth error)."""
    calibration_dir: Path
    """The 2026-09-29 study's INT8 calibration sets: ``detnet_pooled/dataset.txt`` and ``keynet/dataset.txt``, one sample per
    line naming its npy files (float u8 / 255; KeyNet: crop, then keypoint priors)."""
    models_dir: Path = Path(__file__).resolve().parents[1] / "models"
    """Holds the ONNX inputs; receives the ``.rknn`` files and ``rknn_report.json``."""
    builds: tuple[str, ...] = ()
    """Names of the builds to run (e.g. ``keynet_b1_int8``); all defaults when empty."""
    eval_samples: int = 0
    """Score only the first N held-out samples (0 = all)."""
    single_core_mode: bool = False
    """RKNN's RK3588 single-core build (smaller model, no multi-core); off by default."""
    score_device: Path | None = None
    """Instead of building: score device outputs (``nets_bench --heldout <set dir> --out <this dir>``) against ``device_set``."""
    device_set: EvalSet = "heldout"
    """The evaluation set the device outputs came from."""
    device_label: str = ""
    """Report key for the device run, e.g. ``cap_a detnet_b1_fp16 + keynet_b1_int8``."""


@dataclass
class DetNetScore:
    """A DetNet build vs PyTorch FP32 on the same pooled frames."""

    samples: int
    present_slots: int
    """Slots PyTorch reports present (logit > 0): the centre/radius errors are taken over these."""
    centre_px_mean: float
    centre_px_p90: float
    centre_px_max: float
    radius_px_mean: float
    radius_px_max: float
    presence_agreement: float
    """Share of all slots where (logit > 0) matches PyTorch."""
    presence_logit_abs_max: float
    max_abs_diff: dict[str, float]


@dataclass
class KeyNetScore:
    """A KeyNet build vs PyTorch FP32 on the same crops and priors."""

    samples: int
    positives: int
    keypoint_crop_px_mean: float
    """Mean over positive crops and landmarks of |decode(npu) - decode(torch)| in 96-px crop pixels."""
    keypoint_crop_px_p90: float
    keypoint_crop_px_max: float
    keypoint_net_px_mean: float | None
    """The same shift mapped to the 640 x 480 net frame through the crop's linear part (the 2026-09-29 study's 'vs FP32 px')."""
    gt_net_px_torch: float | None
    """PyTorch's mean ground-truth error in net px (the study's rule), for scale; None without labels (s66)."""
    gt_net_px_npu: float | None
    d_rel_mm_mean: float
    d_rel_mm_max: float
    presence_agreement: float
    pinch_probability_abs_mean: float
    """|sigmoid(npu) - sigmoid(torch)| of the pinch logit over positive crops."""
    pinch_probability_abs_p90: float
    pinch_probability_abs_max: float
    pinch_agreement: float
    """Share of positive crops where (pinch probability > 0.5) matches PyTorch."""
    max_abs_diff: dict[str, float]


@dataclass
class BuildRecord:
    """One RKNN build and its simulator score."""

    name: str
    file: str
    sha256: str
    bytes: int
    onnx: str
    precision: str
    algorithm: str
    batch: int
    calibration: dict[str, int]
    """Calibration samples per source (INT8 only)."""
    build_seconds: float
    simulate_seconds: float
    scores: dict[str, dict] = field(default_factory=dict)
    """Simulator scores per evaluation set (``heldout``, ``s66``)."""


def refine_peak(profiles: ndarray, peak: ndarray) -> ndarray:
    """numpy port of handtrack's ``_refine_peak``: a three-sample log-quadratic vertex, clamped to two bins."""
    size: int = profiles.shape[-1]
    centre: ndarray = np.clip(peak, 1, size - 2)
    samples: ndarray = np.take_along_axis(profiles, (centre[..., None] + np.arange(-1, 2)), axis=-1)
    logs: ndarray = np.log(np.maximum(samples, np.float32(1e-30)))
    curvature: ndarray = logs[..., 0] - 2 * logs[..., 1] + logs[..., 2]
    usable: ndarray = (curvature < -1e-6) & (samples > 0).all(axis=-1)
    offset: ndarray = 0.5 * (logs[..., 0] - logs[..., 2]) / np.where(usable, curvature, np.float32(1.0))
    return np.where(usable, centre.astype(np.float32) + np.clip(offset, -2.0, 2.0), peak.astype(np.float32))


def decode_heatmaps(heatmaps: ndarray) -> ndarray:
    """numpy port of ``handtrack.labels.heatmaps.decode_heatmaps``: [b, k, 18, 18] -> crop pixels [b, k, 2]."""
    flat: ndarray = heatmaps.reshape(*heatmaps.shape[:2], -1)
    index: ndarray = flat.argmax(axis=-1)
    x: ndarray = index % HEATMAP
    y: ndarray = index // HEATMAP
    rows: ndarray = np.take_along_axis(heatmaps, y[..., None, None], axis=-2)[..., 0, :]
    columns: ndarray = np.take_along_axis(heatmaps, x[..., None, None], axis=-1)[..., 0]
    points: ndarray = np.stack((refine_peak(rows, x), refine_peak(columns, y)), axis=-1)
    return (points + 0.5) * (CROP / HEATMAP) - 0.5


def decode_distance(distance: ndarray) -> ndarray:
    """numpy port of ``decode_distance``: [b, k, 18] -> relative distance mm."""
    index: ndarray = np.clip(refine_peak(distance, distance.argmax(axis=-1)), 0, distance.shape[-1] - 1)
    return index * (2 * DISTANCE_RANGE_MM / (distance.shape[-1] - 1)) - DISTANCE_RANGE_MM


def sigmoid(x: ndarray) -> ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def study_set(study: Path, name: str, inputs: int) -> list[ndarray]:
    """The 2026-09-29 study's calibration set (float /255 npy files) as arrays; image back to u8 (it was u8 / 255 exactly)."""
    lines: list[str] = (study / name / "dataset.txt").read_text().split("\n")
    arrays: list[ndarray] = [np.concatenate([np.load(line.split()[j]) for line in lines if line]) for j in range(inputs)]
    arrays[0] = np.rint(arrays[0] * 255.0).astype(np.uint8)
    return arrays


def calibration(build: Build, work: Path, study_dir: Path) -> tuple[Path, dict[str, int]]:
    """Write per-batch npy files + ``dataset.txt`` for an INT8 build; return the list file and the sample counts per source."""
    sources: dict[str, int] = {}
    if build.net == "detnet":
        frames: list[ndarray] = [study_set(study_dir, "detnet_pooled", 1)[0][:, 0]]
        sources["study_detnet_pooled"] = len(frames[0])
        s66: Path = work / "s66_detnet_calib.npz"
        if s66.exists():
            with np.load(s66) as archive:
                frames.append(archive["pooled_u8"])
            sources["s66"] = len(frames[-1])
        arrays: list[ndarray] = [np.concatenate(frames)[:, None]]
    else:
        study: list[ndarray] = study_set(study_dir, "keynet", 2)
        crops: list[ndarray] = [study[0][:, 0]]
        priors: list[ndarray] = [study[1]]
        sources["study_keynet"] = len(crops[0])
        s66 = work / "s66_keynet_calib.npz"
        if s66.exists():
            with np.load(s66) as archive:
                crops.append(archive["crops_u8"])
                priors.append(archive["keypoints"].astype(np.float32))
            sources["s66"] = len(crops[-1])
        arrays = [np.concatenate(crops)[:, None], np.concatenate(priors)]
    directory: Path = work / "calibration" / f"{build.net}_b{build.batch}"
    directory.mkdir(parents=True, exist_ok=True)
    count: int = len(arrays[0])
    order: ndarray = np.random.default_rng(1001).permutation(count)
    lines: list[str] = []
    for start in range(0, count - count % build.batch, build.batch):
        picked: ndarray = order[start : start + build.batch]
        paths: list[str] = []
        for slot, array in enumerate(arrays):
            path: Path = directory / f"{start:05d}_{slot}.npy"
            np.save(path, np.ascontiguousarray(array[picked]))
            paths.append(str(path))
        lines.append(" ".join(paths))
    listing: Path = directory / "dataset.txt"
    listing.write_text("\n".join(lines) + "\n")
    return listing, sources


def simulate(rknn: RKNN, build: Build, feeds: list[ndarray], shapes: list[tuple[int, ...]]) -> list[ndarray]:
    """Run every held-out sample through the simulator in static batches (the last batch wraps; its extra rows are dropped)."""
    count: int = len(feeds[0])
    collected: list[list[ndarray]] = [[] for _ in shapes]
    for start in range(0, count, build.batch):
        indices: ndarray = np.arange(start, start + build.batch) % count
        values: list | None = rknn.inference(inputs=[np.ascontiguousarray(feed[indices]) for feed in feeds], data_format=["nchw"] * len(feeds))
        if values is None:
            raise RuntimeError(f"{build.name}: simulator returned no outputs at sample {start}")
        kept: int = min(build.batch, count - start)
        for slot, value in enumerate(values):
            collected[slot].append(np.asarray(value, dtype=np.float32).reshape(build.batch, *shapes[slot])[:kept])
    return [np.concatenate(parts) for parts in collected]


def score_detnet(outputs: list[ndarray], reference: dict[str, ndarray]) -> DetNetScore:
    center, radius, logit = outputs
    present: ndarray = reference["presence_logit"] > 0
    centre_px: ndarray = np.linalg.norm((center - reference["center"]) * np.array([640.0, 480.0]), axis=-1)[present]
    radius_px: ndarray = (np.abs(radius - reference["radius"]) * 640.0)[present]
    diffs: dict[str, float] = {name: float(np.abs(value - reference[name]).max()) for name, value in zip(DETNET_OUTPUTS, outputs, strict=True)}
    return DetNetScore(len(center), int(present.sum()), float(centre_px.mean()), float(np.quantile(centre_px, 0.9)), float(centre_px.max()),
                       float(radius_px.mean()), float(radius_px.max()), float(np.mean((logit > 0) == present)),
                       float(np.abs(logit - reference["presence_logit"]).max()), diffs)


def score_keynet(outputs: list[ndarray], reference: dict[str, ndarray], linear: ndarray | None, truth: ndarray | None) -> KeyNetScore:
    heatmaps, distance, presence, pinch = outputs
    positive: ndarray = reference["kind"] == 0
    points: ndarray = decode_heatmaps(heatmaps)
    shift: ndarray = points - reference["points_crop"]
    crop_px: ndarray = np.linalg.norm(shift, axis=-1)[positive]
    net_px: float | None = None
    gt_torch: float | None = None
    gt_npu: float | None = None
    if linear is not None and truth is not None:
        net_px = float(np.linalg.norm(np.einsum("bij,bkj->bki", linear, shift), axis=-1)[positive].mean())
        gt_torch = float(np.linalg.norm(np.einsum("bij,bkj->bki", linear, reference["points_crop"] - truth), axis=-1)[positive].mean())
        gt_npu = float(np.linalg.norm(np.einsum("bij,bkj->bki", linear, points - truth), axis=-1)[positive].mean())
    d_rel: ndarray = np.abs(decode_distance(distance) - reference["d_rel_mm"])[positive]
    pinch_probability: ndarray = sigmoid(pinch[:, 0])[positive]
    pinch_torch: ndarray = sigmoid(reference["pinch_logit"])[positive]
    pinch_error: ndarray = np.abs(pinch_probability - pinch_torch)
    targets: dict[str, ndarray] = {"heatmaps": reference["heatmaps"], "distance": reference["distance"],
                                   "presence_logit": reference["presence_logit"][:, None], "pinch_logit": reference["pinch_logit"][:, None]}
    diffs: dict[str, float] = {name: float(np.abs(value - targets[name]).max()) for name, value in zip(KEYNET_OUTPUTS, outputs, strict=True)}
    return KeyNetScore(len(heatmaps), int(positive.sum()), float(crop_px.mean()), float(np.quantile(crop_px, 0.9)), float(crop_px.max()),
                       net_px, gt_torch, gt_npu, float(d_rel.mean()), float(d_rel.max()),
                       float(np.mean((presence[:, 0] > 0) == (reference["presence_logit"] > 0))), float(pinch_error.mean()),
                       float(np.quantile(pinch_error, 0.9)), float(pinch_error.max()), float(np.mean((pinch_probability > 0.5) == (pinch_torch > 0.5))), diffs)


def load_reference(config: Config, net: Net, which: EvalSet) -> dict[str, ndarray] | None:
    """An evaluation set's inputs and PyTorch outputs; None when its file is absent."""
    path: Path = config.work_dir / EVAL_FILES[which].format(net=net)
    if not path.exists():
        return None
    with np.load(path) as archive:
        reference: dict[str, ndarray] = {name: archive[name] for name in archive.files}
    if config.eval_samples:
        reference = {name: value[: config.eval_samples] for name, value in reference.items()}
    return reference


def labels(config: Config, which: EvalSet, count: int) -> tuple[ndarray | None, ndarray | None]:
    """The held-out crops' linear crop-to-net maps and ground-truth keypoints (none for s66)."""
    if which != "heldout":
        return None, None
    linear: ndarray = np.linalg.inv(np.load(config.keynet_cache / "crop_from_net.npy")[:count].astype(np.float64))[:, :2, :2]
    return linear, np.load(config.keynet_cache / "points_crop.npy")[:count]


def score_set(config: Config, net: Net, which: EvalSet, reference: dict[str, ndarray], outputs: list[ndarray]) -> dict:
    if net == "detnet":
        return asdict(score_detnet(outputs, reference))
    linear, truth = labels(config, which, len(outputs[0]))
    return asdict(score_keynet(outputs, reference, linear, truth))


def image_feed(build: Build, reference: dict[str, ndarray]) -> ndarray:
    """The image input as the runtime gets it: u8 for INT8 models (the input is quantised at about the u8 step anyway), float
    values on the u8 scale [0, 255] for FP16 models (the model divides by 255), so FP16 sees the unrounded crop or pool.

    DetNet's float pool is the 4 x 4 mean of the 640 x 480 frame when the set has the frames (s66); the caches store only the
    rounded pool. KeyNet's float crop is the bilinear crop when the set has it (s66); the caches store u8 crops.
    """
    if build.net == "detnet":
        if build.precision == "fp16" and "frames_u8" in reference:
            frames: ndarray = reference["frames_u8"].astype(np.float32)
            return frames.reshape(len(frames), 120, 4, 160, 4).mean(axis=(2, 4))[:, None]
        return reference["pooled_u8"][:, None] if build.precision == "int8" else reference["pooled_u8"][:, None].astype(np.float32)
    if build.precision == "fp16" and "crops_f32" in reference:
        return (reference["crops_f32"] * 255.0)[:, None]
    return reference["crops_u8"][:, None] if build.precision == "int8" else reference["crops_u8"][:, None].astype(np.float32)


def run_build(config: Config, build: Build) -> BuildRecord:
    """Build one model, export it, simulate the held-out set and score it."""
    work: Path = config.work_dir
    is_key: bool = build.net == "keynet"
    started: float = time.perf_counter()
    rknn: RKNN = RKNN(verbose=False)
    try:
        options: dict = {"target_platform": "rk3588", "mean_values": [[0], [0] * 63] if is_key else [[0]],
                         "std_values": [[255], [1] * 63] if is_key else [[255]], "quantized_dtype": "w8a8", "quantized_method": "channel",
                         "quantized_algorithm": build.algorithm, "single_core_mode": config.single_core_mode}
        if rknn.config(**options) != 0 or rknn.load_onnx(model=str(config.models_dir / build.onnx)) != 0:
            raise RuntimeError(f"{build.name}: config/load_onnx failed")
        sources: dict[str, int] = {}
        listing: Path | None = None
        if build.precision == "int8":
            listing, sources = calibration(build, work, config.calibration_dir)
        if rknn.build(do_quantization=build.precision == "int8", dataset=None if listing is None else str(listing)) != 0:
            raise RuntimeError(f"{build.name}: build failed")
        artifact: Path = config.models_dir / f"{build.name}.rknn"
        if rknn.export_rknn(str(artifact)) != 0:
            raise RuntimeError(f"{build.name}: export failed")
        built: float = time.perf_counter()
        if rknn.init_runtime() != 0:
            raise RuntimeError(f"{build.name}: simulator init failed")
        scores: dict[str, dict] = {}
        for which in ("heldout", "s66"):
            reference: dict[str, ndarray] | None = load_reference(config, build.net, which)
            if reference is None:
                continue
            if is_key:
                feeds: list[ndarray] = [image_feed(build, reference), reference["keypoints"].astype(np.float32)]
                outputs: list[ndarray] = simulate(rknn, build, feeds, [(21, HEATMAP, HEATMAP), (21, HEATMAP), (1,), (1,)])
            else:
                outputs = simulate(rknn, build, [image_feed(build, reference)], [(2, 2), (2,), (2,)])
            scores[which] = score_set(config, build.net, which, reference, outputs)
            np.savez(work / "sim" / f"{build.name}_{which}.npz", **dict(zip(KEYNET_OUTPUTS if is_key else DETNET_OUTPUTS, outputs, strict=True)))
    finally:
        rknn.release()
    return BuildRecord(build.name, artifact.name, hashlib.sha256(artifact.read_bytes()).hexdigest(), artifact.stat().st_size, build.onnx,
                       build.precision, build.algorithm if build.precision == "int8" else "-", build.batch, sources, built - started,
                       time.perf_counter() - built, scores)


def check_decoder(config: Config) -> float:
    """The numpy decoder must reproduce handtrack's torch decoder on the PyTorch heatmaps (max px difference)."""
    reference: dict[str, ndarray] | None = load_reference(config, "keynet", "heldout")
    if reference is None:
        raise FileNotFoundError(f"{config.work_dir}/keynet_heldout.npz: run handtrack's export_nets_onnx first")
    return float(np.abs(decode_heatmaps(reference["heatmaps"]) - reference["points_crop"]).max())


def score_device(config: Config, report_path: Path) -> None:
    """Score raw device outputs (``nets_bench --heldout``) against an evaluation set and store them in the report."""
    directory: Path = config.score_device if config.score_device is not None else Path()
    record: dict = {"set": config.device_set}
    for net in ("detnet", "keynet"):
        reference: dict[str, ndarray] | None = load_reference(config, net, config.device_set)
        path: Path = directory / f"{net}_out_f32.bin"
        if reference is None or not path.exists():
            continue
        raw: ndarray = np.fromfile(path, dtype="<f4")
        if net == "detnet":
            rows: ndarray = raw.reshape(-1, 8)
            outputs: list[ndarray] = [rows[:, :4].reshape(-1, 2, 2), rows[:, 4:6], rows[:, 6:8]]
        else:
            rows = raw.reshape(-1, 21 * HEATMAP * HEATMAP + 21 * HEATMAP + 2)
            split: int = 21 * HEATMAP * HEATMAP
            outputs = [rows[:, :split].reshape(-1, 21, HEATMAP, HEATMAP), rows[:, split : split + 21 * HEATMAP].reshape(-1, 21, HEATMAP),
                       rows[:, -2:-1], rows[:, -1:]]
        count: int = len(outputs[0])
        record[net] = score_set(config, net, config.device_set, {name: value[:count] for name, value in reference.items()}, outputs)
    report: dict = json.loads(report_path.read_text())
    report.setdefault("device", {})[config.device_label or str(directory)] = record
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(record, indent=1))


def main(config: Config) -> None:
    (config.work_dir / "sim").mkdir(parents=True, exist_ok=True)
    report_path: Path = config.models_dir / "rknn_report.json"
    if config.score_device is not None:
        score_device(config, report_path)
        return
    decoder_error: float = check_decoder(config)
    print(f"numpy decode_heatmaps vs torch: max {decoder_error:.2e} px")
    if decoder_error > 1e-3:
        raise RuntimeError("numpy decode_heatmaps does not reproduce handtrack's decoder")
    chosen: tuple[Build, ...] = tuple(build for build in DEFAULT_BUILDS if not config.builds or build.name in config.builds)
    previous: dict = json.loads(report_path.read_text()) if report_path.exists() else {}
    records: dict[str, dict] = previous.get("builds", {})
    for build in chosen:
        record: BuildRecord = run_build(config, build)
        records[record.name] = asdict(record)
        report_path.write_text(json.dumps({"toolkit": "rknn-toolkit2 2.3.2", "target": "rk3588", "decoder_check_px": decoder_error, "builds": records,
                                           "device": previous.get("device", {})}, indent=2) + "\n")
        print(json.dumps(records[record.name], indent=1), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
