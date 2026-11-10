"""DetNet producers on the Rerun 0.42 dataloader: catalog -> CPU luma decode -> labels, in spawn workers; letterbox on the GPU.

One ``RerunIterableDataset`` per training dataset replays a frozen ``Manifest`` (index ``frame_index``, one field per camera
video), so each epoch's order is reproducible. Samples carry only their fields; ``Manifest.worker_plan`` names the
(segment, frame) of each, and the worker labels it from a per-segment table it builds once: the existing hand-model path
(``segment_labels``) for UmeTrack and SHOW3D, ``coco133_hands`` for datasets that ship 2D keypoints (GSI, Gen-HumanEgo).
Workers letterbox and pool too, so only 160x120 samples reach the training process.
"""

import dataclasses
import itertools
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from multiprocessing import resource_tracker
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import torch
from jaxtyping import Bool, Float32, Int64, UInt8
from numpy import ndarray
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from handtrack.data.augment import DetNetAugment, augment_detnet
from handtrack.data.batches import DetNetBatch
from handtrack.data.cache import CENTRE_MARGIN, MAX_CIRCLE_RADIUS
from handtrack.data.catalog import UPRIGHT_TURNS, layout_for, list_segments, read_hand_timeline, read_rig, read_statics
from handtrack.data.echo import Producer
from handtrack.data.luma import grey_from_luma, grey_from_rgb
from handtrack.data.segment_labels import segment_labels
from handtrack.data.stream import DetNetSamples, ImageHands, detnet_samples
from handtrack.geometry.letterbox import Letterbox, letterbox_for
from handtrack.labels.coco2d import coco133_hands
from handtrack.labels.crops import scale_intensity

LabelKind = Literal["hand_model", "coco133"]
SHARING_STRATEGY: str = "file_system"
"""Tensors from workers travel as named shared-memory files: the default file-descriptor strategy opens an authenticated socket
per tensor in the training process, and those handshakes starved the training loop."""
WORKER_NICENESS: int = 10
RESERVED_CORES: int = 4
"""The first cores stay free of decode workers, for the training process's threads (kernel launches, producers)."""
"""Decode workers run at this niceness so the training process preempts them."""
HANDS: str = "/world/gt/hands"
SIDES: tuple[str, str] = ("left", "right")


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """One training dataset as the producer reads it."""

    dataset: str
    """Catalog dataset name."""
    dataset_index: int
    """Position in the batch source's dataset list (``DetNetBatch.dataset``)."""
    cameras: tuple[str, ...]
    """Camera entities ``/world/rig_NN/cam_MM``; the video is at ``<camera>/pinhole/video``."""
    codec: Literal["h264", "h265", "av1"]
    labels: LabelKind
    camera_offset: int
    """First camera id of this dataset in ids unique across datasets."""
    min_score: float | None = None
    """coco133 only: a hand is trusted when its ``scores:pred_score_quality`` reaches this (None: the dataset has no score)."""
    quarter_turns: tuple[int, ...] | None = None
    """Clockwise quarter turns per camera before the letterbox (``catalog.UPRIGHT_TURNS``); None: the legacy letterbox."""


DATASET_SPECS: dict[str, tuple[tuple[str, ...], Literal["h264", "h265", "av1"], LabelKind, int, float | None]] = {
    "dataforge-umetrack": (tuple(f"/world/rig_00/cam_0{i}" for i in range(4)), "av1", "hand_model", 0, None),
    "dataforge-show3d": (("/world/rig_01/cam_00", "/world/rig_01/cam_01"), "av1", "hand_model", 4, None),
    "gsi-six-view": (tuple(f"/world/rig_00/cam_0{i}" for i in range(6)), "h265", "coco133", 8, None),
    "dataforge-gen-humanego-subset": (tuple(f"/world/rig_00/cam_0{i}" for i in range(6)), "h264", "coco133", 14, 0.8),
    "dataforge-hot3d-quest3": (("/world/rig_00/cam_00", "/world/rig_00/cam_01"), "av1", "hand_model", 6, None),
}
"""Datasets the producers read: cameras, codec, label path, first unique camera id, and the coco133 trust score.
HOT3D is evaluation only (its catalog split has no training segments)."""


def source_spec(dataset: str, dataset_index: int, upright: bool = False) -> SourceSpec:
    """``upright``: turn every camera upright (``UPRIGHT_TURNS``), frames and labels alike."""
    cameras, codec, labels, camera_offset, min_score = DATASET_SPECS[dataset]
    return SourceSpec(dataset, dataset_index, cameras, codec, labels, camera_offset, min_score, UPRIGHT_TURNS[dataset] if upright else None)


def camera_letterbox(spec: SourceSpec, camera: int, width: int, height: int) -> Letterbox:
    return letterbox_for(width, height, None if spec.quarter_turns is None else spec.quarter_turns[camera])


@dataclass(frozen=True, slots=True)
class FrameLabels:
    """Every camera's labels for each frame of one segment (frames indexed by ``frame_index``)."""

    image_valid: Bool[Tensor, "f c"]
    hand_label: Int64[Tensor, "f c 2"]
    circles: Float32[Tensor, "f c 2 3"]
    net_xy: Float32[Tensor, "f c 2 21 2"]
    in_front: Bool[Tensor, "f c 2 21"]


def _video_field(camera: str) -> str:
    return f"{camera}/pinhole/video:VideoStream:sample"


def _frame_times(dataset, segment: str, camera: str) -> tuple[Int64[ndarray, "f"], Int64[ndarray, "f"]]:
    """frame_index -> video_time (ns) of one camera's video."""
    entity: str = f"{camera}/pinhole/video"
    table: pa.Table = dataset.filter_segments([segment]).filter_contents([entity]).reader(index="frame_index").select("frame_index", "video_time", _video_field(camera)).to_arrow_table()
    present: ndarray = np.asarray(table[_video_field(camera)].is_valid())
    frames: Int64[ndarray, "f"] = np.asarray(table["frame_index"].to_numpy(zero_copy_only=False), dtype=np.int64)[present]
    times: Int64[ndarray, "f"] = table["video_time"].cast(pa.int64()).to_numpy(zero_copy_only=False).astype(np.int64)[present]
    return frames, times


def hand_model_labels(dataset, name: str, segment: str, infos: dict, quarter_turns: tuple[int, ...] | None = None) -> FrameLabels:
    """UmeTrack / SHOW3D: the existing ground-truth labels (hand model forward kinematics, our lens models), without meshes."""
    info = infos[segment]
    layout = layout_for(name)
    statics: pa.Table = read_statics(dataset, info)
    rig, letterboxes = read_rig(statics, info, quarter_turns)
    timeline = read_hand_timeline(dataset, info, statics)
    frames, times = _frame_times(dataset, segment, layout.cameras[0])
    rows: Int64[ndarray, "k"] = np.searchsorted(timeline.video_time_ns, times).clip(0, len(timeline.video_time_ns) - 1)
    matched: Bool[ndarray, "k"] = timeline.video_time_ns[rows] == times
    labels = segment_labels(timeline, rig, letterboxes, rows, layout.pose_gated, mesh_labels=False)
    count: int = int(frames.max()) + 1 if len(frames) else 0
    cameras: int = len(letterboxes)
    out = _empty_labels(count, cameras)
    at: Tensor = torch.from_numpy(frames)
    out.image_valid[at] = labels.image_valid & torch.from_numpy(matched)[:, None]
    out.hand_label[at] = labels.hand_label
    out.circles[at] = labels.circles
    out.net_xy[at] = labels.projection.net_xy
    out.in_front[at] = labels.projection.in_front
    return out


def _empty_labels(frames: int, cameras: int) -> FrameLabels:
    return FrameLabels(
        image_valid=torch.zeros((frames, cameras), dtype=torch.bool), hand_label=torch.zeros((frames, cameras, 2), dtype=torch.int64),
        circles=torch.full((frames, cameras, 2, 3), torch.nan), net_xy=torch.full((frames, cameras, 2, 21, 2), torch.nan),
        in_front=torch.zeros((frames, cameras, 2, 21), dtype=torch.bool),
    )


def _scalar(column: pa.ChunkedArray) -> Float32[ndarray, "f"]:
    """First value of a list-of-number column per row; NaN where null or empty."""
    return np.asarray([v[0] if v else np.nan for v in column.to_pylist()], dtype=np.float32)


def dense_rows(column: pa.ChunkedArray, length: int, inner: int = 1) -> tuple[Float32[ndarray, "f ..."], Bool[ndarray, "f"]]:
    """Rows of a list column (``list<float>`` or ``list<fixed_size_list<float, inner>>``) holding exactly ``length`` items, as one
    float32 array of shape (rows, length[, inner]) with NaN in every other row, and the mask of those rows. Vectorised: no Python
    loop over rows (the producers share the training process's interpreter lock)."""
    lists: pa.ListArray = column.combine_chunks() if column.num_chunks != 1 else column.chunk(0)
    lengths: ndarray = lists.value_lengths().fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
    flat: pa.Array = lists.flatten()  # honours slicing and skips null rows
    if inner > 1:
        flat = flat.flatten()
    values: Float32[ndarray, "v"] = flat.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    ok: Bool[ndarray, "f"] = lengths == length
    starts: Int64[ndarray, "f"] = np.concatenate(([0], np.cumsum(lengths * inner)[:-1]))
    shape: tuple[int, ...] = (len(lengths), length, inner) if inner > 1 else (len(lengths), length)
    out: Float32[ndarray, "f ..."] = np.full(shape, np.nan, dtype=np.float32)
    out[ok] = values[starts[ok][:, None] + np.arange(length * inner)].reshape(-1, *shape[1:])
    return out, ok


def coco133_labels(dataset, spec: SourceSpec, segment: str) -> FrameLabels:
    """GSI / Gen-HumanEgo: labels from the stored lens-aware ``coco133_uv_projected`` keypoints and the per-hand trust columns."""
    statics: pa.Table = dataset.filter_segments([segment]).filter_contents([f"{c}/pinhole" for c in spec.cameras]).reader(index=None).to_arrow_table()
    entities: list[str] = [f"{c}/pinhole/coco133_uv_projected" for c in spec.cameras] + [f"{HANDS}/{s}/{p}" for s in SIDES for p in ("confidence", "scores")]
    table: pa.Table = dataset.filter_segments([segment]).filter_contents(entities).reader(index="frame_index").to_arrow_table()
    frames: Int64[ndarray, "f"] = np.asarray(table["frame_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    count: int = int(frames.max()) + 1 if len(frames) else 0
    out = _empty_labels(count, len(spec.cameras))
    trusted: Bool[ndarray, "f 2"] = np.ones((len(frames), 2), dtype=bool)
    for side, name in enumerate(SIDES):
        confidence: Float32[ndarray, "f"] = _scalar(table[f"{HANDS}/{name}/confidence:Scalars:scalars"])
        trusted[:, side] &= np.nan_to_num(confidence) > 0
        if spec.min_score is not None:
            trusted[:, side] &= np.nan_to_num(_scalar(table[f"{HANDS}/{name}/scores:pred_score_quality"])) >= spec.min_score
    at: Tensor = torch.from_numpy(frames)
    for c, camera in enumerate(spec.cameras):
        width, height = (int(v) for v in statics[f"{camera}/pinhole:Pinhole:resolution"][0].as_py()[0])
        uv, has_points = dense_rows(table[f"{camera}/pinhole/coco133_uv_projected:Points2D:positions"], 133, inner=2)
        weight, has_weight = dense_rows(table[f"{camera}/pinhole/coco133_uv_projected:simplecv.KeypointConfidence2D:confidences"], 133)
        usable: Bool[ndarray, "f"] = has_points & has_weight
        uv[~usable] = np.nan
        weight = np.where(usable[:, None], weight, np.float32(0.0))
        hands = coco133_hands(torch.from_numpy(uv), torch.from_numpy(weight), torch.from_numpy(trusted), camera_letterbox(spec, c, width, height))
        out.image_valid[at, c] = hands.image_valid
        out.hand_label[at, c] = hands.hand_label
        out.circles[at, c] = hands.circles
        out.net_xy[at, c] = hands.net_xy
        out.in_front[at, c] = hands.in_front
    return out


class LabelledFrames(IterableDataset):
    """Runs in a spawn worker: replays this worker's manifest shard and yields ready ``DetNetSamples`` chunks (CPU tensors).

    The worker also letterboxes and pools each image, so only 160x120 bytes cross the process boundary and the training
    process does no per-image work (its Python threads would otherwise starve the training loop of the interpreter lock).
    """

    def __init__(self, spec: SourceSpec, segments: Sequence[str], manifest_path: str, epoch: int, catalog_url: str, chunk: int = 256) -> None:
        self.spec, self.segments, self.manifest_path, self.epoch, self.catalog_url, self.chunk = spec, list(segments), manifest_path, epoch, catalog_url, chunk

    def __iter__(self) -> Iterator[DetNetSamples]:
        import rerun as rr
        from rerun.experimental.dataloader import BlockShuffle, DataSource, Field, Manifest, RerunIterableDataset, VideoFrameDecoder, Yuv420Frame

        torch.set_num_threads(1)
        # The file_system strategy starts torch_shm_manager from this worker. It must not inherit the multiprocessing resource
        # tracker's pipe: the tracker exits only when every copy closes, the manager waits for the training process to disconnect,
        # and the training process waits for the tracker at exit, so the interpreter never exits.
        tracker_fd: int | None = getattr(resource_tracker._resource_tracker, "_fd", None)  # private, absent from the stubs
        if tracker_fd is not None:
            os.set_inheritable(tracker_fd, False)
        torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)
        # Decode workers fill every core; the training process must still get CPU at once to keep the GPU fed.
        os.nice(WORKER_NICENESS)
        cores: set[int] = os.sched_getaffinity(0)
        if len(cores) > RESERVED_CORES:
            os.sched_setaffinity(0, sorted(cores)[RESERVED_CORES:])
        dataset = rr.catalog.CatalogClient(self.catalog_url).get_dataset(self.spec.dataset)
        manifest = Manifest.from_parquet(self.manifest_path).shuffle(BlockShuffle(), seed=self.epoch)
        fields = {f"cam{c}": Field(_video_field(camera), decode=VideoFrameDecoder(codec=self.spec.codec, output_format="yuv420p"))
                  for c, camera in enumerate(self.spec.cameras)}
        stream = RerunIterableDataset.from_manifest(manifest, DataSource(dataset=dataset, segments=self.segments), fields)
        info = get_worker_info()
        groups, order = manifest.worker_plan(0, info.id if info is not None else 0)
        keys: pa.Table = pa.concat_tables(groups).select(["segment_id", "anchor"])
        segment_ids: list[str] = keys["segment_id"].to_pylist()
        anchors: list[int] = keys["anchor"].to_pylist()
        infos: dict = {i.segment_id: i for i in list_segments(dataset, self.spec.dataset)} if self.spec.labels == "hand_model" else {}
        cache: dict[str, FrameLabels] = {}
        pending: list[DetNetSamples] = []
        for k, sample in enumerate(stream):
            segment: str = segment_ids[order[k]]
            frame: int = int(anchors[order[k]])
            labels = cache.get(segment)
            if labels is None:
                cache.clear()  # BlockShuffle keeps a worker on one segment for a block; one segment's labels at a time bounds memory
                labels = hand_model_labels(dataset, self.spec.dataset, segment, infos, self.spec.quarter_turns) if self.spec.labels == "hand_model" else coco133_labels(dataset, self.spec, segment)
                cache[segment] = labels
            if frame >= labels.image_valid.shape[0]:
                continue
            for c in range(len(self.spec.cameras)):
                if not bool(labels.image_valid[frame, c]):
                    continue
                video = sample[f"cam{c}"]
                if not isinstance(video, Yuv420Frame):
                    raise TypeError(f"{self.spec.dataset} {segment} frame {frame}: camera {c} decoded to {type(video).__name__}, not YUV420")
                full: Bool[Tensor, "1"] = torch.tensor([str(video.color_range).lower().endswith("full")])
                grey: UInt8[Tensor, "1 h w"] = grey_from_luma(video.y[:, 0] if video.y.dim() == 4 else video.y, full)
                net: UInt8[Tensor, "1 480 640"] = camera_letterbox(self.spec, c, int(grey.shape[-1]), int(grey.shape[-2])).apply(grey)
                hand_label: Int64[Tensor, "1 2"] = labels.hand_label[frame, c][None]
                truth = ImageHands(net_xy=labels.net_xy[frame, c][None], points_cam=torch.zeros((1, 2, 21, 3)), in_front=labels.in_front[frame, c][None],
                                   valid=torch.ones((1, 2), dtype=torch.bool))
                pending.append(detnet_samples(net, hand_label, labels.circles[frame, c][None], truth, self.spec.dataset_index, self.spec.camera_offset + c))
                if len(pending) == self.chunk:
                    yield _concat(pending)
                    pending = []
        if pending:
            yield _concat(pending)


def _concat(parts: list[DetNetSamples]) -> DetNetSamples:
    return DetNetSamples(**{f.name: torch.cat([getattr(p, f.name) for p in parts]) for f in dataclasses.fields(DetNetSamples)})


def build_manifest(spec: SourceSpec, segments: Sequence[str], catalog_url: str, path: Path, workers: int) -> Path:
    """Scan the source once into a parquet manifest (natural order; each epoch shuffles it). Reused when it already exists."""
    if path.exists():
        return path
    import rerun as rr
    from rerun.experimental.dataloader import DataSource, Field, Manifest, VideoFrameDecoder

    dataset = rr.catalog.CatalogClient(catalog_url).get_dataset(spec.dataset)
    fields = {f"cam{c}": Field(_video_field(camera), decode=VideoFrameDecoder(codec=spec.codec, output_format="yuv420p")) for c, camera in enumerate(spec.cameras)}
    manifest = Manifest.generate(DataSource(dataset=dataset, segments=list(segments)), "frame_index", fields, num_workers_per_rank=workers,
                                 required_fields=list(fields))
    path.parent.mkdir(parents=True, exist_ok=True)
    partial: Path = path.with_suffix(".partial.parquet")
    manifest.write_parquet(partial)
    # On the NAS's NFS shares a file created over NFS can refuse later opens until its mode is set explicitly.
    os.chmod(partial, 0o644)
    os.replace(partial, path)
    return path


PASS_STRIDE: int = 1_000_000
"""An endless producer's pass p of epoch e shuffles with seed e * PASS_STRIDE + p."""


def rerun_producer(spec: SourceSpec, segments: Sequence[str], *, catalog_url: str, manifest_path: Path, workers: int, device: torch.device,
                   endless: bool = False) -> Producer:
    """A ``Producer`` for ``EchoStream``: spawn workers stream ready samples (decoded, labelled, letterboxed, pooled).
    ``endless``: after one pass over the manifest start the next (a new shuffle), for ``EchoStream(epoch_steps=...)``."""

    def produce(epoch: int) -> Iterator[DetNetSamples]:
        for seed in itertools.count(epoch * PASS_STRIDE) if endless else (epoch,):
            yield from _one_pass(seed)

    def _one_pass(seed: int) -> Iterator[DetNetSamples]:
        torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)
        loader = DataLoader(LabelledFrames(spec, segments, str(manifest_path), seed, catalog_url), batch_size=None, num_workers=workers,
                            multiprocessing_context="spawn", pin_memory=False, prefetch_factor=4, persistent_workers=False)
        yield from loader

    return produce


NVDEC_SLICE: int = 64
"""Frames per NVDEC call: bounds the RGB peak (64 x 1080p = 400 MB) before the luma + letterbox transform shrinks it."""
NOMINAL_FPS: int = 30
"""Rate written into the decoder's MP4 wrapper; frames are addressed by position, so it does not change which frame is served."""


def nvdec_producer(spec: SourceSpec, segments: Sequence[str], *, catalog_url: str, device: torch.device, endless: bool = False,
                   fetch_block_size: int = 1024, chunk: int = 256) -> Producer:
    """A ``Producer`` that decodes on NVDEC in the training process (simplecv's segment-wide ``KeyedNvdecDecoder``).

    The Rerun dataloader still queries the catalog and orders samples (``BlockShuffle``); frames are decoded, turned to luma and
    letterboxed on the device, so no CPU decode workers and no inter-process copies remain.
    """
    from simplecv.rerun_dataloader import KeyedFrame, KeyedNvdecDecoder

    def to_net(camera: int):
        def transform(frames: UInt8[Tensor, "n 3 h w"]) -> UInt8[Tensor, "n 480 640"]:
            return camera_letterbox(spec, camera, int(frames.shape[-1]), int(frames.shape[-2])).apply(grey_from_rgb(frames))

        return transform

    def produce(epoch: int) -> Iterator[DetNetSamples]:
        import rerun as rr
        from rerun.experimental.dataloader import BlockShuffle, DataSource, Field, RerunIterableDataset

        dataset = rr.catalog.CatalogClient(catalog_url).get_dataset(spec.dataset)
        fields = {f"cam{c}": Field(_video_field(camera), decode=KeyedNvdecDecoder(dataset, f"{camera}/pinhole/video".lstrip("/"), "frame_index", device,
                                                                                   NOMINAL_FPS, max_frames_per_call=NVDEC_SLICE, transform=to_net(c)))
                  for c, camera in enumerate(spec.cameras)}
        stream = RerunIterableDataset(DataSource(dataset=dataset, segments=list(segments)), index="frame_index", fields=fields,
                                      fetch_block_size=fetch_block_size, shuffle_strategy=BlockShuffle(), decode_threads=1)
        infos: dict = {i.segment_id: i for i in list_segments(dataset, spec.dataset)} if spec.labels == "hand_model" else {}
        cache: dict[str, FrameLabels] = {}
        for seed in itertools.count(epoch * PASS_STRIDE) if endless else (epoch,):
            stream.set_epoch(seed)
            nets: list[Tensor] = []
            rows: list[tuple[FrameLabels, int, int]] = []
            for sample in stream:
                for c in range(len(spec.cameras)):
                    keyed = sample[f"cam{c}"]
                    if not isinstance(keyed, KeyedFrame):
                        raise TypeError(f"{spec.dataset}: camera {c} decoded to {type(keyed).__name__}, not a KeyedFrame")
                    labels = cache.get(keyed.segment_id)
                    if labels is None:
                        cache.clear()  # a fetch block stays inside one segment; one segment's labels at a time bounds memory
                        labels = (hand_model_labels(dataset, spec.dataset, keyed.segment_id, infos, spec.quarter_turns) if spec.labels == "hand_model"
                                  else coco133_labels(dataset, spec, keyed.segment_id))
                        cache[keyed.segment_id] = labels
                    if keyed.index >= labels.image_valid.shape[0] or not bool(labels.image_valid[keyed.index, c]):
                        continue
                    nets.append(keyed.image)
                    rows.append((labels, keyed.index, c))
                if len(nets) >= chunk:
                    yield _samples(spec, nets, rows, device)
                    nets, rows = [], []
            if nets:
                yield _samples(spec, nets, rows, device)

    return produce


def _samples(spec: SourceSpec, nets: list[Tensor], rows: list[tuple[FrameLabels, int, int]], device: torch.device) -> DetNetSamples:
    """One chunk of device-side net frames with their labels gathered on the host and moved once."""
    hand_label: Int64[Tensor, "m 2"] = torch.stack([labels.hand_label[f, c] for labels, f, c in rows]).to(device)
    circles: Float32[Tensor, "m 2 3"] = torch.stack([labels.circles[f, c] for labels, f, c in rows]).to(device)
    net_xy: Float32[Tensor, "m 2 21 2"] = torch.stack([labels.net_xy[f, c] for labels, f, c in rows]).to(device)
    in_front: Bool[Tensor, "m 2 21"] = torch.stack([labels.in_front[f, c] for labels, f, c in rows]).to(device)
    truth = ImageHands(net_xy=net_xy, points_cam=torch.zeros((*net_xy.shape[:-1], 3), device=device), in_front=in_front,
                       valid=torch.ones(hand_label.shape, dtype=torch.bool, device=device))
    samples: DetNetSamples = detnet_samples(torch.stack(nets), hand_label, circles, truth, spec.dataset_index, 0)
    camera: Int64[Tensor, "m"] = torch.tensor([spec.camera_offset + c for _, _, c in rows], device=device)
    return dataclasses.replace(samples, camera=camera)


def training_finish(augment: DetNetAugment, intensity_range: tuple[float, float] = (0.6, 1.4)):
    """``EchoStream`` finish for training: circle sanity, intensity scaling and the GPU augmentation, as the cache path did."""

    def finish(samples: DetNetSamples, generator: torch.Generator) -> DetNetBatch:
        centre: Float32[Tensor, "b 2 2"] = samples.circle[..., :2]
        sane: Bool[Tensor, "b 2"] = (samples.circle[..., 2] <= MAX_CIRCLE_RADIUS) & (centre >= -CENTRE_MARGIN).all(-1) & (centre <= 1 + CENTRE_MARGIN).all(-1)
        pooled: Float32[Tensor, "b 1 120 160"] = scale_intensity(samples.pooled[:, None].float() / 255.0, generator, *intensity_range)
        batch = DetNetBatch(pooled=pooled, circle=samples.circle, presence=samples.presence, circle_mask=samples.circle_mask & sane,
                            presence_mask=samples.presence_mask, dataset=samples.dataset)
        return augment_detnet(batch, samples.points, samples.in_front, augment, generator) if augment.enabled else batch

    return finish
