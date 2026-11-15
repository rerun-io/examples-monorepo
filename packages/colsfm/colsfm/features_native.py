"""The native TensorRT extraction path both engine backends run through.

`colsfm.features_trt` (the blob's static batch-1 ALIKED) and
`colsfm.features_raco` (fabio-sim's batch- and shape-dynamic RaCo-ALIKED) once
carried a copy each of the same six steps: decode and resize on a bounded thread
pool, stage or convert to the engine's tensor, execute, gate on the score,
truncate to a keypoint budget, map the normalised coordinates back onto the
original image, and write the two database columns. Only the first and the last
of those are model-specific, and neither is model-specific in a way that needs
its own copy of the loop.

**What is shared.** Everything from the tensor-binding boundary inwards:
`prepared_stream` feeds the GPU, `FeatureBatch` names what came back off the
bindings, `select_keypoints` is the gate plus the top-k, and
`store_feature_batch` writes the rows. `run_native_extraction` sequences them
over a list of `ExtractionGroup`s, reusing one `TensorRTSession` and one
`GpuPreprocessor` per engine and input size.

**What the backends still own**, through a `NativeModel` and the groups they
build:

* **Bindings** — `NativeModel.image_binding` is the graph's image input, and the
  name `GpuPreprocessor` gives its own output so the two chain without a rename.
* **Score meaning** — `NativeModel.score_meaning` records that the blob's graph
  emits a detector response and RaCo's emits a selection rank. The gate is the
  same `>=` either way; what differs is the number the caller supplies, and
  `colsfm.features` supplies it (`BLOB_DETECTOR_THRESHOLD` against
  `RACO_MIN_SCORE`). Recording it here is what keeps a later refactor from
  "unifying" the two into one threshold.
* **Batch limits** — the blob's graph declares a static batch of 1 and has no
  axis to widen; RaCo's profiles admit 1..16 fixed and 1..8 shape-dynamic. The
  caller validates its own bound and passes `batch_size`.
* **Engine selection and sizing** — one group for the blob, one group per
  network size for RaCo, each naming the engine it resolved and the size to
  resize to. A group that leaves the size unset takes the engine's own declared
  input shape, which is how the blob's static graph avoids repeating 1200x1920.

**Subsampling order is deliberately unchanged**: gate first, then rank the
survivors and keep the strongest `max_num_features`. Both backends did exactly
that, and the SSC thinning that actually selects points on the RaCo path happens
later, on LightGlue's scores (`colsfm.matching_raco`).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Final, Literal, TypeAlias

import cv2
import numpy as np
import pycolmap
from jaxtyping import Bool, Float32, Int64, UInt8
from numpy import ndarray

from colsfm.database import Descriptors, KeypointsXY, database_transaction
from colsfm.tensorrt_runtime import DeviceTensor, GpuPreprocessor, TensorRTSession

NetworkBatch: TypeAlias = Float32[ndarray, "batch 3 network_height network_width"]
"""One batch of preprocessed images as an engine takes it: planar RGB in [0, 1]."""

NetworkImage: TypeAlias = Float32[ndarray, "1 3 network_height network_width"]
"""One preprocessed image, i.e. a `NetworkBatch` of one."""

ResizedImageBGR: TypeAlias = UInt8[ndarray, "network_height network_width 3"]
"""One decoded image stretched to an engine's input size, still `cv2.imread`'s BGR HWC uint8."""

NormalizedKeypoints: TypeAlias = Float32[ndarray, "num_keypoints 2"]
"""An ALIKED head's own output coordinates, normalised to [-1, 1] over the network input."""

KeypointScores: TypeAlias = Float32[ndarray, " num_keypoints"]
"""One image's `scores` output, whatever `NativeModel.score_meaning` says it means."""

ScoreMeaning: TypeAlias = Literal["detector_response", "selection_rank"]
"""What a graph's `scores` output actually is.

`detector_response` is the blob's ALIKED head: a real detector confidence, which
`detector_threshold` gates meaningfully. `selection_rank` is RaCo's exported
`arange(2048, 0, -1) / 2048`, a strictly decreasing priority that never touched a
score map, so gating it above zero truncates by rank and means nothing else. The
two are not interchangeable and this alias exists so nothing collapses them."""

ALIKED_IMAGE_BINDING: Final[str] = "image"
"""The image input both ALIKED graphs declare, and `GpuPreprocessor`'s output name."""

DESCRIPTOR_TYPE: Final[pycolmap.FeatureExtractorType] = pycolmap.FeatureExtractorType.ALIKED_N16ROT
"""What the descriptor blob is labelled as in the database.

The blob's `aliked.onnx` is ALIKED-n16 (`colsfm.features` module docstring), and
COLMAP's nearest first-class descriptor type is `ALIKED_N16ROT`. The label only
travels with the blob; nothing in this pipeline dispatches on it, and the
descriptors are recovered bit-exactly either way
(`docs/spec/pycolmap-capabilities.md` §4)."""

DEFAULT_PREPROCESSING_WORKERS: Final[int] = 8
"""Decode-and-resize threads, matching `tools/raco_extract.ExtractConfig`."""

PREPROCESSING_LOOKAHEAD: Final[int] = 2
"""Prepared frames held per worker, so the GPU never waits and memory stays bounded.

Each prepared frame is a 1920x1200x3 uint8 image (6.9 MB), or four times that on
the host-conversion path, so an unbounded `ThreadPoolExecutor.map` over RoboCap's
4528 frames would want tens of gigabytes. This window holds `workers * lookahead`
of them."""

DEFAULT_GPU_PREPROCESSING: Final[bool] = True
"""Whether the uint8-to-float32 conversion runs on the GPU rather than the host.

`False` restores the original host path — `[..., ::-1].transpose(2, 0, 1)` to
float32 and `/= 255` in the decode workers, then a pageable float32 upload — and
exists so the two can be measured against each other and so a machine whose
TensorRT will not build the four-layer preprocessing network still works."""


@dataclass(frozen=True, slots=True)
class NativeModel:
    """The model-specific half of a native extraction: bindings and score meaning.

    Small on purpose. Everything that varies per *run* — which engine, which
    input size, how many images at once — travels on the `ExtractionGroup`s and
    the `batch_size` argument instead, because those vary within one backend.
    """

    name: str
    """The backend's name, as `colsfm.features.FeatureBackend` spells it."""
    image_binding: str
    """The graph's image input; also the name `GpuPreprocessor` marks its output with."""
    descriptor_type: pycolmap.FeatureExtractorType
    """The label the stored descriptor blob carries."""
    score_meaning: ScoreMeaning
    """What the graph's `scores` output is; see `ScoreMeaning`."""


@dataclass(frozen=True, slots=True)
class ImageTask:
    """One image to extract, and where its row already lives."""

    image_id: int
    """The database `image_id`, owned by `colsfm.database.create_database`."""
    image_path: Path
    """Absolute path to the image file."""
    image_width: int
    """Original width in pixels, from the image's camera."""
    image_height: int
    """Original height in pixels, from the image's camera."""


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """One decoded image, stretched to the engine's input size."""

    task: ImageTask
    """Which image this is."""
    resized_bgr_hwc: ResizedImageBGR
    """The decoded image at the engine's input size, still BGR HWC uint8.

    This is what the GPU path stages and uploads: 6.9 MB rather than the 27.6 MB
    of the float32 tensor, and no host arithmetic beyond the decode."""
    network_bchw: NetworkImage | None
    """Float32 planar RGB in [0, 1]; only the host-conversion path builds it."""

    def network_image(self) -> NetworkImage:
        """The float32 tensor, for callers that asked for host conversion.

        Returns:
            Float32 `[1, 3, height, width]` planar RGB in [0, 1].

        Raises:
            RuntimeError: When this image was prepared for the GPU path, which
                deliberately leaves the conversion to `GpuPreprocessor`.
        """
        if self.network_bchw is None:
            raise RuntimeError(f"{self.task.image_path} was prepared for the GPU path; no host tensor exists")
        return self.network_bchw


@dataclass(frozen=True, slots=True)
class FeatureBatch:
    """One execution's outputs, named and given a batch axis.

    The typed boundary the review asked for: past this point nothing reads raw
    binding names or guesses at a shape. A batch-1 graph whose `scores` binding
    comes back as `[2048]` and a batched graph whose `scores` comes back as
    `[8, 2048]` both arrive here as `[batch, num_keypoints]`.
    """

    keypoints_bn2: Float32[ndarray, "batch num_keypoints 2"]
    """Normalised xy coordinates in [-1, 1] over the network input."""
    descriptors_bnd: Float32[ndarray, "batch num_keypoints descriptor_dim"]
    """L2-normalised float32 descriptors, 128 columns for both ALIKED graphs."""
    scores_bn: Float32[ndarray, "batch num_keypoints"]
    """Per-keypoint scores; see `NativeModel.score_meaning`."""

    @classmethod
    def from_bindings(cls, outputs: Mapping[str, ndarray], batch_size: int) -> FeatureBatch:
        """Reshape one execution's output bindings onto a batch axis.

        Args:
            outputs: The engine's `keypoints`, `descriptors` and `scores` arrays,
                at whatever rank the graph declares them.
            batch_size: Images in this execution.

        Returns:
            The same numbers, as `[batch, ...]`.
        """
        scores_bn: Float32[ndarray, "batch num_keypoints"] = np.asarray(outputs["scores"], dtype=np.float32).reshape(
            batch_size, -1
        )
        return cls(
            keypoints_bn2=np.asarray(outputs["keypoints"], dtype=np.float32).reshape(batch_size, -1, 2),
            descriptors_bnd=np.asarray(outputs["descriptors"], dtype=np.float32).reshape(
                batch_size, scores_bn.shape[1], -1
            ),
            scores_bn=scores_bn,
        )


@dataclass(frozen=True, slots=True)
class ExtractionGroup:
    """The images one engine runs, at one input size."""

    engine_path: Path
    """The serialised engine these images execute on, already resolved or built."""
    tasks: Sequence[ImageTask]
    """The images, in the order their results are wanted."""
    network_height: int | None = None
    """Height every image is resized to; `None` takes the engine's declared input height."""
    network_width: int | None = None
    """Width every image is resized to; `None` takes the engine's declared input width."""


def to_network_bchw(resized_bgr_hwc: ResizedImageBGR) -> NetworkImage:
    """The blob's host-side conversion: BGR HWC uint8 to planar RGB float32 in [0, 1].

    Kept as its own function because it is the reference the GPU preprocessing
    engine is checked against, bit for bit
    (`tests/colsfm/test_features_trt_gpu_preprocess.py`).

    Args:
        resized_bgr_hwc: One decoded image at the engine's input size.

    Returns:
        Float32 `[1, 3, height, width]` planar RGB in [0, 1].
    """
    network_chw: Float32[ndarray, "3 network_height network_width"] = np.ascontiguousarray(
        resized_bgr_hwc[..., ::-1].transpose(2, 0, 1), dtype=np.float32
    )
    network_chw /= np.float32(255.0)
    return network_chw[None]


def preprocess_image(
    task: ImageTask,
    network_height: int,
    network_width: int,
    convert_on_host: bool = False,
) -> PreparedImage:
    """Decode one image and reproduce the blob's OpenCV preprocessing.

    The `cv2.resize` is skipped when the file is already the engine's size, which
    every Galileo frame is; OpenCV's own resize does not shortcut that case and
    the copy costs 2.4 ms per image (docs/raco-speed-investigation.md §4).
    `INTER_LINEAR` at a 1:1 scale is an identity, so skipping it changes nothing.

    Args:
        task: The image to read and the size its camera declares.
        network_height: The engine's input height.
        network_width: The engine's input width.
        convert_on_host: Also build the float32 planar RGB tensor here, in this
            worker thread. The GPU path leaves it unset and uploads the uint8.

    Returns:
        The decoded image at the engine's input size.

    Raises:
        ValueError: When OpenCV cannot decode the file, or when the file's size
            disagrees with the camera the database holds for it — which would
            silently misplace every keypoint.
    """
    bgr_hwc: ndarray | None = cv2.imread(str(task.image_path), cv2.IMREAD_COLOR)
    if bgr_hwc is None:
        raise ValueError(f"OpenCV could not decode {task.image_path}")
    if bgr_hwc.shape[:2] != (task.image_height, task.image_width):
        raise ValueError(
            f"The database camera says {task.image_width}x{task.image_height}, but "
            f"{task.image_path} is {bgr_hwc.shape[1]}x{bgr_hwc.shape[0]}"
        )
    if bgr_hwc.shape[:2] == (network_height, network_width):
        resized_bgr_hwc: ResizedImageBGR = bgr_hwc
    else:
        resized_bgr_hwc = cv2.resize(bgr_hwc, (network_width, network_height), interpolation=cv2.INTER_LINEAR)
    return PreparedImage(
        task=task,
        resized_bgr_hwc=resized_bgr_hwc,
        network_bchw=to_network_bchw(resized_bgr_hwc) if convert_on_host else None,
    )


def prepared_stream(
    tasks: Sequence[ImageTask], workers: int, network_height: int, network_width: int, convert_on_host: bool = False
) -> Iterator[PreparedImage]:
    """Decode ahead of the GPU without letting the queue grow without bound.

    Args:
        tasks: The images, in the order their results are wanted.
        workers: Decode threads.
        network_height: The engine's input height.
        network_width: The engine's input width.
        convert_on_host: Whether the workers also build the float32 tensor.

    Yields:
        Prepared images in `tasks` order.
    """
    window: int = max(workers, 1) * PREPROCESSING_LOOKAHEAD
    remaining: Iterator[ImageTask] = iter(tasks)
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        pending: deque[Future[PreparedImage]] = deque(
            pool.submit(preprocess_image, task, network_height, network_width, convert_on_host)
            for task in islice(remaining, window)
        )
        while pending:
            yield pending.popleft().result()
            for task in islice(remaining, 1):
                pending.append(pool.submit(preprocess_image, task, network_height, network_width, convert_on_host))


def batched_images(prepared: Iterator[PreparedImage], batch_size: int) -> Iterator[list[PreparedImage]]:
    """Group a prepared-image stream into fixed-size batches, the last one short.

    Args:
        prepared: Prepared images, in the order their results are wanted.
        batch_size: Images per batch.

    Yields:
        Non-empty lists of at most `batch_size` prepared images.
    """
    while batch := list(islice(prepared, max(batch_size, 1))):
        yield batch


def normalized_to_pixels(normalized_xy: NormalizedKeypoints, image_width: int, image_height: int) -> KeypointsXY:
    """Map ALIKED's normalised coordinates onto the original image.

    `(k + 1) * 0.5 * (size - 1)`, the blob's own mapping
    (feature_extractor_main.md §6.3), which targets the **original** image size
    rather than the network's — the stretch is undone here and nowhere else.

    Args:
        normalized_xy: Float32 `[num_keypoints, 2]` coordinates in [-1, 1].
        image_width: Original image width in pixels.
        image_height: Original image height in pixels.

    Returns:
        Float32 `[num_keypoints, 2]` xy pixel coordinates.
    """
    scale_xy: Float32[ndarray, " 2"] = np.asarray(
        [(image_width - 1) / 2.0, (image_height - 1) / 2.0], dtype=np.float32
    )
    return np.ascontiguousarray((normalized_xy + np.float32(1.0)) * scale_xy, dtype=np.float32)


def select_keypoints(scores: KeypointScores, *, min_score: float, max_num_features: int) -> Bool[ndarray, " num_keypoints"]:
    """Gate on the score, then keep the strongest of what survived.

    Both steps in that order, which is the order both backends have always used.
    The gate is `>=`, as the blob's own post-processing is
    (feature_extractor_main.md §6.3); a `>` would drop a point sitting exactly on
    `detector_threshold`. The truncation ranks with `argsort` over the gated
    scores, so a keypoint the gate removed cannot come back.

    Args:
        scores: Float32 `[num_keypoints]` scores; see `NativeModel.score_meaning`.
        min_score: The gate, applied with `>=`.
        max_num_features: Keypoint ceiling; 0 or less keeps everything the gate left.

    Returns:
        A `[num_keypoints]` mask of the keypoints to store.
    """
    kept: Bool[ndarray, " num_keypoints"] = scores >= np.float32(min_score)
    if max_num_features > 0 and int(kept.sum()) > max_num_features:
        # A partition, not a full sort: the answer is a mask, so which order the top
        # `max_num_features` come back in cannot reach it, and ALIKED offers 20 000
        # candidates per image where the cap keeps 4096.
        gated: Float32[ndarray, " num_keypoints"] = np.where(kept, scores, -np.inf)
        strongest: Int64[ndarray, " num_selected"] = np.argpartition(gated, -max_num_features)[-max_num_features:]
        kept = np.zeros_like(kept)
        kept[strongest] = True
    return kept


def store_feature_batch(
    database: pycolmap.Database,
    batch: Sequence[PreparedImage],
    features: FeatureBatch,
    *,
    min_score: float,
    max_num_features: int,
    descriptor_type: pycolmap.FeatureExtractorType = DESCRIPTOR_TYPE,
) -> None:
    """Write one executed batch's keypoints and descriptors into the database.

    Only the two feature columns are touched. The rows themselves were written by
    `colsfm.database.create_database`, so `image_id`, `camera_id` and `frame_id`
    are left exactly as the pipeline assigned them — the same contract COLMAP's
    own `ImageReader` gets on the pycolmap backend.

    Args:
        database: An open COLMAP database whose image rows already exist.
        batch: The prepared images that made up this execution, in binding order.
        features: What the engine returned for them.
        min_score: Gate on the scores; see `select_keypoints`.
        max_num_features: Keypoint ceiling per image; 0 or less keeps everything
            the gate left.
        descriptor_type: The label the stored descriptor blob carries.
    """
    # One transaction for the batch: an image's keypoints and its descriptors describe
    # each other, and an interrupted extraction that had left one without the other
    # would be a contradiction to every reader downstream rather than damage it could
    # see. `colsfm.database.database_transaction` documents what it does and does not
    # promise.
    with database_transaction(database):
        for index, item in enumerate(batch):
            scores: KeypointScores = features.scores_bn[index].reshape(-1)
            kept: Bool[ndarray, " num_keypoints"] = select_keypoints(
                scores, min_score=min_score, max_num_features=max_num_features
            )
            keypoints_xy: KeypointsXY = normalized_to_pixels(
                features.keypoints_bn2[index].reshape(-1, 2)[kept], item.task.image_width, item.task.image_height
            )
            descriptors: Descriptors = np.ascontiguousarray(
                features.descriptors_bnd[index].reshape(len(scores), -1)[kept]
            )
            database.write_keypoints(item.task.image_id, keypoints_xy)
            database.write_descriptors(
                item.task.image_id,
                pycolmap.FeatureDescriptorsFloat(data=descriptors, type=descriptor_type).to_bytes(),
            )


def image_tasks(database_path: Path, image_root: Path, image_names: Sequence[str]) -> list[ImageTask]:
    """Pair each requested image name with its database row and its camera size.

    Args:
        database_path: An existing database holding the image and camera rows.
        image_root: Directory the names are relative to.
        image_names: Images to extract, as stored in the database.

    Returns:
        One task per name, in the order given.

    Raises:
        KeyError: When a name has no image row.
        FileNotFoundError: When an image file is missing.
    """
    # One open and one walk of the image table: `image_ids_by_name` opened the database
    # to read every row for the names, and this function then opened it again to read
    # every row for the camera ids.
    with pycolmap.Database.open(database_path) as database:
        cameras: dict[int, pycolmap.Camera] = {camera.camera_id: camera for camera in database.read_all_cameras()}
        camera_by_name: dict[str, tuple[int, pycolmap.Camera]] = {
            image.name: (image.image_id, cameras[image.camera_id]) for image in database.read_all_images()
        }
    tasks: list[ImageTask] = []
    for name in image_names:
        if name not in camera_by_name:
            raise KeyError(f"The database holds no image named {name}")
        image_id: int
        camera: pycolmap.Camera
        image_id, camera = camera_by_name[name]
        tasks.append(
            ImageTask(
                image_id=image_id,
                image_path=image_root / name,
                image_width=int(camera.width),
                image_height=int(camera.height),
            )
        )
    missing: list[Path] = [task.image_path for task in tasks if not task.image_path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images are missing; the first is {missing[0]}")
    return tasks


def run_native_extraction(
    database_path: Path,
    groups: Sequence[ExtractionGroup],
    model: NativeModel,
    *,
    min_score: float,
    max_num_features: int,
    batch_size: int,
    preprocessing_workers: int = DEFAULT_PREPROCESSING_WORKERS,
    gpu_preprocessing: bool = DEFAULT_GPU_PREPROCESSING,
) -> None:
    """Execute every group and store what came back, reusing sessions and buffers.

    One `TensorRTSession` per distinct engine and one `GpuPreprocessor` per
    engine and input size, both held open for the whole pass: a mixed-camera set
    that alternates between two size groups pays two engine loads, not two per
    group.

    Args:
        database_path: An existing COLMAP database whose image rows are written.
        groups: The engines to run and the images to run through each; empty
            groups are skipped.
        model: The graph's bindings, descriptor label and score meaning.
        min_score: Gate on the scores; see `select_keypoints`.
        max_num_features: Keypoint ceiling per image.
        batch_size: Images executed together. The caller validates it against its
            own engines' profiles, because those are the caller's to know.
        preprocessing_workers: Decode-and-resize threads.
        gpu_preprocessing: Stage the decoded uint8 frames in one page-locked
            batch buffer and let a TensorRT preprocessing engine transpose, widen
            and scale them. `False` does that arithmetic on the host instead.

    Raises:
        RuntimeError: When an engine cannot be deserialised or fails to execute.
    """
    with ExitStack() as stack:
        database: pycolmap.Database = stack.enter_context(pycolmap.Database.open(database_path))
        sessions: dict[Path, TensorRTSession] = {}
        preprocessors: dict[tuple[Path, int, int], GpuPreprocessor] = {}
        for group in groups:
            if not group.tasks:
                continue
            if group.engine_path not in sessions:
                sessions[group.engine_path] = stack.enter_context(
                    TensorRTSession(group.engine_path, reuse_output_buffers=True)
                )
            session: TensorRTSession = sessions[group.engine_path]
            declared_shape: tuple[int, ...] = tuple(session.engine.get_tensor_shape(model.image_binding))
            network_height: int = group.network_height if group.network_height is not None else int(declared_shape[2])
            network_width: int = group.network_width if group.network_width is not None else int(declared_shape[3])
            preprocessor: GpuPreprocessor | None = None
            if gpu_preprocessing:
                preprocessor_key: tuple[Path, int, int] = (group.engine_path, network_height, network_width)
                if preprocessor_key not in preprocessors:
                    preprocessors[preprocessor_key] = stack.enter_context(
                        GpuPreprocessor(
                            height=network_height,
                            width=network_width,
                            max_batch=batch_size,
                            optimal_batch=batch_size,
                            stream=session.stream,
                        )
                    )
                preprocessor = preprocessors[preprocessor_key]
            prepared: Iterator[PreparedImage] = prepared_stream(
                group.tasks, preprocessing_workers, network_height, network_width, not gpu_preprocessing
            )
            for batch in batched_images(prepared, batch_size):
                network_input: NetworkBatch | NetworkImage | DeviceTensor
                if preprocessor is None:
                    network_input = _host_batch(batch)
                else:
                    for slot, item in enumerate(batch):
                        preprocessor.staging_bhwc[slot] = item.resized_bgr_hwc
                    network_input = preprocessor.run(len(batch))
                outputs: dict[str, ndarray] = session.run({model.image_binding: network_input})
                store_feature_batch(
                    database,
                    batch,
                    FeatureBatch.from_bindings(outputs, len(batch)),
                    min_score=min_score,
                    max_num_features=max_num_features,
                    descriptor_type=model.descriptor_type,
                )


def _host_batch(batch: Sequence[PreparedImage]) -> NetworkBatch | NetworkImage:
    """Assemble the host-converted tensors of one batch into the engine's input.

    A single image is passed straight through: `np.concatenate` over a one-item
    list would copy 27.6 MB for nothing.

    Args:
        batch: Prepared images whose workers built the float32 tensor.

    Returns:
        Float32 `[batch, 3, height, width]` planar RGB in [0, 1], contiguous.
    """
    if len(batch) == 1:
        return batch[0].network_image()
    return np.ascontiguousarray(np.concatenate([item.network_image() for item in batch], axis=0), dtype=np.float32)
