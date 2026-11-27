"""The TensorRT extraction loop RaCo-ALIKED runs through (`colsfm.features_raco`).

Frames come from the run's `colsfm.frames.FrameSource` as GPU tensors; `_device_batch` stretches them to the engine's
input size on the engine's own stream and binds the batch in place. `FeatureBatch` names what came back off the
bindings, `select_keypoints` is the score gate plus the top-k, and `store_feature_batch` writes the two database
columns. `run_native_extraction` sequences them over a list of `ExtractionGroup`s (one per engine and network size),
reusing one trtkit `TensorRtDynamicRuntime` (direct binding) per engine.

**Subsampling order**: gate first, then rank the survivors and keep the strongest `max_num_features`; the SSC thinning
that selects points for LightGlue+ happens later, on its scores (`colsfm.matching_trt`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, Self, TypeAlias, runtime_checkable

import numpy as np
import pycolmap
import torch
from jaxtyping import Bool, Float32, Int64
from numpy import ndarray
from torch import Tensor
from trtkit.tensorrt_runtime import TensorRtDynamicRuntime

from colsfm.database import Descriptors, KeypointsXY, database_transaction
from colsfm.frames import FrameSource
from colsfm.keypoint_masks import KeypointMask, MaskSource, keypoints_outside_mask

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
"""The image input the RaCo-ALIKED graphs declare."""

DESCRIPTOR_TYPE: Final[pycolmap.FeatureExtractorType] = pycolmap.FeatureExtractorType.ALIKED_N16ROT
"""What the descriptor blob is labelled as in the database.

The blob's `aliked.onnx` is ALIKED-n16 (`colsfm.features` module docstring), and
COLMAP's nearest first-class descriptor type is `ALIKED_N16ROT`. The label only
travels with the blob; nothing in this pipeline dispatches on it, and the
descriptors are recovered bit-exactly either way
(`docs/spec/pycolmap-capabilities.md` §4)."""


@runtime_checkable
class EngineRuntime(Protocol):
    """What extraction uses of a runtime: trtkit's direct-binding `TensorRtDynamicRuntime`, or its `OnnxDynamicRuntime`."""

    max_input_shapes: dict[str, tuple[int, ...]]

    def __call__(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Execute once and return output views valid until the next call."""
        ...

    def __enter__(self) -> Self: ...

    def __exit__(self, *_: object) -> None:
        """Release the context, engine and buffers (`close()`)."""
        ...


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
    """The graph's image input."""
    descriptor_type: pycolmap.FeatureExtractorType
    """The label the stored descriptor blob carries."""
    score_meaning: ScoreMeaning
    """What the graph's `scores` output is; see `ScoreMeaning`."""


@dataclass(frozen=True, slots=True)
class ImageTask:
    """One image to extract, and where its row already lives."""

    image_id: int
    """The database `image_id`, owned by `colsfm.database.create_database`."""
    name: str
    """The image name, as the database and the frame source key it."""
    image_width: int
    """Original width in pixels, from the image's camera."""
    image_height: int
    """Original height in pixels, from the image's camera."""


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
        scores_bn: Float32[ndarray, "batch num_keypoints"] = np.asarray(outputs["scores"], dtype=np.float32).reshape(batch_size, -1)
        return cls(
            keypoints_bn2=np.asarray(outputs["keypoints"], dtype=np.float32).reshape(batch_size, -1, 2),
            descriptors_bnd=np.asarray(outputs["descriptors"], dtype=np.float32).reshape(batch_size, scores_bn.shape[1], -1),
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
    maximum_batch_size: int | None = None
    """The group's engine's batch ceiling, when below the run's batch size (a large shape-dynamic group)."""


def normalized_to_pixels(normalized_xy: NormalizedKeypoints, image_width: int, image_height: int) -> KeypointsXY:
    """Map ALIKED's normalised coordinates onto the original image.

    `(k + 1) * 0.5 * size - 0.5`, the inverse of the `align_corners=False` normalisation the RaCo graph emits
    (`colsfm.raco_export`, metadata `cusfm_keypoints`), into 0-based pixel centres. It targets the **original** image
    size rather than the network's — the stretch is undone here and nowhere else. The blob's own mapping,
    `(k + 1) * 0.5 * (size - 1)` (feature_extractor_main.md §6.3), is the `align_corners=True` inverse: on RaCo's
    output it scaled every keypoint by `1 - 1 / size` about the centre (half a pixel at the edges).

    Args:
        normalized_xy: Float32 `[num_keypoints, 2]` coordinates in [-1, 1].
        image_width: Original image width in pixels.
        image_height: Original image height in pixels.

    Returns:
        Float32 `[num_keypoints, 2]` xy pixel coordinates.
    """
    scale_xy: Float32[ndarray, " 2"] = np.asarray([image_width / 2.0, image_height / 2.0], dtype=np.float32)
    return np.ascontiguousarray((normalized_xy + np.float32(1.0)) * scale_xy - np.float32(0.5), dtype=np.float32)


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
    batch: Sequence[ImageTask],
    features: FeatureBatch,
    *,
    min_score: float,
    max_num_features: int,
    descriptor_type: pycolmap.FeatureExtractorType = DESCRIPTOR_TYPE,
    masks: MaskSource | None = None,
) -> None:
    """Write one executed batch's keypoints and descriptors into the database.

    Only the two feature columns are touched. The rows themselves were written by
    `colsfm.database.create_database`, so `image_id`, `camera_id` and `frame_id`
    are left exactly as the pipeline assigned them — the same contract COLMAP's
    own `ImageReader` gets on the pycolmap backend.

    Args:
        database: An open COLMAP database whose image rows already exist.
        batch: The images that made up this execution, in binding order.
        features: What the engine returned for them.
        min_score: Gate on the scores; see `select_keypoints`.
        max_num_features: Keypoint ceiling per image; 0 or less keeps everything
            the gate left.
        descriptor_type: The label the stored descriptor blob carries.
        masks: Per-image keypoint masks (`colsfm.keypoint_masks`), applied after the gate and the top-k: a keypoint on
            a masked pixel is not stored, the rest keep their order and descriptors. None = unmasked.
    """
    # One transaction for the batch: an image's keypoints and its descriptors describe
    # each other, and an interrupted extraction that had left one without the other
    # would be a contradiction to every reader downstream rather than damage it could
    # see. `colsfm.database.database_transaction` documents what it does and does not
    # promise.
    with database_transaction(database):
        for index, task in enumerate(batch):
            scores: KeypointScores = features.scores_bn[index].reshape(-1)
            kept: Bool[ndarray, " num_keypoints"] = select_keypoints(scores, min_score=min_score, max_num_features=max_num_features)
            keypoints_xy: KeypointsXY = normalized_to_pixels(features.keypoints_bn2[index].reshape(-1, 2)[kept], task.image_width, task.image_height)
            descriptors: Descriptors = np.ascontiguousarray(features.descriptors_bnd[index].reshape(len(scores), -1)[kept])
            mask: KeypointMask | None = None if masks is None else masks.mask(task.name, task.image_width, task.image_height)
            if mask is not None:
                outside: Bool[ndarray, " num_kept"] = keypoints_outside_mask(keypoints_xy, mask)
                keypoints_xy, descriptors = np.ascontiguousarray(keypoints_xy[outside]), np.ascontiguousarray(descriptors[outside])
            database.write_keypoints(task.image_id, keypoints_xy)
            database.write_descriptors(
                task.image_id,
                pycolmap.FeatureDescriptorsFloat(data=descriptors, type=descriptor_type).to_bytes(),
            )


def image_tasks(database_path: Path, image_names: Sequence[str]) -> list[ImageTask]:
    """Pair each requested image name with its database row and its camera size.

    Args:
        database_path: An existing database holding the image and camera rows.
        image_root: Directory the names are relative to.
        image_names: Images to extract, as stored in the database.

    Returns:
        One task per name, in the order given.

    Raises:
        KeyError: When a name has no image row.
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
                name=name,
                image_width=int(camera.width),
                image_height=int(camera.height),
            )
        )
    return tasks


def run_native_extraction(
    database_path: Path,
    groups: Sequence[ExtractionGroup],
    model: NativeModel,
    frames: FrameSource,
    *,
    min_score: float,
    max_num_features: int,
    batch_size: int,
    masks: MaskSource | None = None,
    open_runtime: Callable[[Path], EngineRuntime] | None = None,
) -> None:
    """Execute every group on the frames' device and store what came back, one runtime at a time.

    A group's runtime is released, and torch's cached blocks returned to the driver, before the next group's is
    created: an execution context reserves activation memory for its profile maximum (3.8 GB for eight 480x800 cube
    faces), and a rig with three image sizes that kept all three alive needed their sum.

    Args:
        database_path: An existing COLMAP database whose image rows are written.
        groups: The engines to run and the images to run through each; empty groups are skipped.
        model: The graph's bindings, descriptor label and score meaning.
        frames: Where the pixels come from (`colsfm.frames`), on the device `open_runtime`'s runtime takes.
        min_score: Gate on the scores; see `select_keypoints`.
        max_num_features: Keypoint ceiling per image.
        batch_size: Images executed together. The caller validates it against its own engines' profiles.
        masks: Per-image keypoint masks; see `store_feature_batch`.
        open_runtime: Opens a group's `engine_path`: a direct-binding `TensorRtDynamicRuntime` when None,
            `trtkit.OnnxDynamicRuntime` for an ONNX graph (tests pass a fake).

    Raises:
        RuntimeError: When an engine cannot be deserialised or fails to execute.
    """
    with ExitStack() as stack:
        database: pycolmap.Database = stack.enter_context(pycolmap.Database.open(database_path))
        for group in groups:
            if not group.tasks:
                continue
            # One group's context, engine and buffers at a time: the block closes them before the next group opens its own.
            with open_runtime(group.engine_path) if open_runtime is not None else TensorRtDynamicRuntime(group.engine_path, direct_binding=True) as runtime:
                group_batch: int = min(batch_size, group.maximum_batch_size or batch_size)
                declared_shape: tuple[int, ...] = runtime.max_input_shapes[model.image_binding]
                network_height: int = group.network_height if group.network_height is not None else int(declared_shape[2])
                network_width: int = group.network_width if group.network_width is not None else int(declared_shape[3])
                for first in range(0, len(group.tasks), group_batch):
                    tasks: list[ImageTask] = list(group.tasks[first : first + group_batch])
                    network_tensor: Tensor = _device_batch(frames, tasks, network_height, network_width)
                    # The engine's views are valid until the next call: the database write is the one place they become host.
                    outputs: dict[str, ndarray] = {name: value.cpu().numpy() for name, value in runtime({model.image_binding: network_tensor}).items()}
                    store_feature_batch(
                        database,
                        tasks,
                        FeatureBatch.from_bindings(outputs, len(tasks)),
                        min_score=min_score,
                        max_num_features=max_num_features,
                        descriptor_type=model.descriptor_type,
                        masks=masks,
                    )
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()


def _device_batch(frames: FrameSource, tasks: Sequence[ImageTask], network_height: int, network_width: int) -> Tensor:
    """One runtime input built on the frames' device from frame-source pixels (on the GPU, on torch's current stream: the
    runtime waits on it).

    RGB, stretched to the network size with bilinear interpolation only when the image is not already that size, scaled
    to [0, 1], float32 `[batch, 3, h, w]` — what the RaCo graph was exported against.

    Args:
        frames: The run's frame source.
        tasks: The batch's images.
        network_height: The engine's input height.
        network_width: The engine's input width.

    Returns:
        The batch, contiguous, bound by the runtime with no copy.

    Raises:
        ValueError: When a frame's size disagrees with its database camera (which would misplace every keypoint).
    """
    resized: list[Tensor] = []
    for task, (name, rgb) in zip(tasks, frames.frames([task.name for task in tasks]), strict=True):
        if (int(rgb.shape[1]), int(rgb.shape[2])) != (task.image_height, task.image_width):
            raise ValueError(f"The database camera says {task.image_width}x{task.image_height}, but frame {name} is {rgb.shape[2]}x{rgb.shape[1]}")
        image: Tensor = rgb[None].float()
        if (task.image_height, task.image_width) != (network_height, network_width):
            image = torch.nn.functional.interpolate(
                image, size=(network_height, network_width), mode="bilinear", align_corners=False, antialias=False
            )
        resized.append(image)
    return torch.cat(resized).div_(255.0)
