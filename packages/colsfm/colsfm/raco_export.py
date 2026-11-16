"""Export fabio-sim's RaCo-ALIKED and LightGlue+ to the ONNX graphs the default colsfm run needs.

The torch modules come from `recipes/lightglue-onnx-src` (LightGlue-ONNX @ d12b4ba); the weights download through
`torch.hub` from the RaCo and ALIKED GitHub releases on first use. `colsfm.apis.raco_export` writes every graph
`colsfm.model_assets` names; the engines are built from them on first use, per machine (`colsfm.tensorrt_runtime`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx
import torch
from lightglue_dynamo.config import Extractor, RankerMode
from lightglue_dynamo.models import LightGlue, RaCoALIKED
from onnxscript import opset20 as onnx_op
from onnxscript import optimizer as onnx_optimizer
from torch import nn
from torch.nn import functional


def export_shape_as_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """A tensor's shape as an int64 tensor, built from its (possibly symbolic) sizes.

    LightGlue-ONNX calls `torch._shape_as_tensor` while exporting, which upstream runs on torch 2.13; on the workspace's
    torch 2.12 that op fails on the exporter's fake tensors ("data is not allocated yet"). Stacking scalar tensors of the
    sizes exports to the same `Shape`-derived graph and keeps every dynamic axis symbolic.

    Args:
        tensor: Any tensor.

    Returns:
        `int64[rank]` holding the shape.
    """
    return torch.stack(
        [
            torch.tensor(size, dtype=torch.int64, device=tensor.device)
            if isinstance(size, int)
            else torch.scalar_tensor(size, dtype=torch.int64, device=tensor.device)
            for size in tensor.shape
        ]
    )


def _install_shape_helper() -> None:
    """Point every loaded LightGlue-ONNX module that imported `shape_as_tensor` at `export_shape_as_tensor`."""
    import sys

    import lightglue_dynamo.models  # noqa: F401  (loads every model module)

    for name, module in list(sys.modules.items()):
        if name.startswith("lightglue_dynamo") and hasattr(module, "shape_as_tensor"):
            module.shape_as_tensor = export_shape_as_tensor  # pyrefly: ignore[missing-attribute]


_install_shape_helper()

UPSTREAM_COMMIT: str = "d12b4ba1632f558234e3f084e1f3d8bdf9147890"
"""Pinned fabio-sim/LightGlue-ONNX revision used by the Pixi environment."""

INPUT_DIM_DIVISOR: int = 32
"""`Extractor.raco_aliked.input_dim_divisor`: RaCo's four-level pyramid pools by 32.

Upstream's own CLI refuses a static height or width that is not a multiple of
this, and builds its dynamic spatial dimensions as `32 * Dim(...)` for the same
reason. `colsfm.features_raco` rounds every frame size up to it."""

MINIMUM_DYNAMIC_SIDE: int = 64
"""Smallest height or width the boundary ranker accepts, from upstream's CLI check."""


@dataclass
class ExportConfig:
    """The export settings of the RaCo-ALIKED extractors and the LightGlue+ matcher."""

    image_height: int = 1200
    """Static padded image height expected by cuSFM."""
    image_width: int = 1920
    """Static image width expected by cuSFM."""
    num_keypoints: int = 2048
    """Fixed extractor keypoint budget."""
    opset: int = 20
    """ONNX opset required by upstream's Dynamo RaCo export."""
    minimum_batch_size: int = 1
    """Minimum batch accepted by a dynamic extractor."""
    optimal_batch_size: int = 8
    """Preferred TensorRT optimization-profile batch."""
    maximum_batch_size: int = 16
    """Maximum batch accepted by a dynamic extractor."""
    portable_deform_conv: bool = True
    """Decompose DeformConv into GridSample nodes instead of its native ONNX operator."""
    dynamic_shape: bool = False
    """When set, also mark height and width dynamic so one engine serves every frame size.

    The graph then takes the image at its *network* size directly and drops the
    in-graph `interpolate`: a spatial resample whose target is a symbolic
    `ceil(size / 32) * 32` cannot be expressed without leaking the rounding into
    the graph, and the host already resizes. Callers must therefore hand it a
    height and a width that are multiples of `INPUT_DIM_DIVISOR`."""


class CusfmExtractor(nn.Module):
    """Adapt fabio-sim's pixel-coordinate extractor to cuSFM's ONNX seam."""

    extractor: RaCoALIKED

    def __init__(self, config: ExportConfig) -> None:
        super().__init__()
        processing_height: int = (config.image_height + 31) // 32 * 32
        self.extractor = RaCoALIKED(
            num_keypoints=config.num_keypoints,
            portable_deform_conv=config.portable_deform_conv,
            topk_chunk_size=65536,
            ranker_mode=RankerMode.boundary,
        )
        image_size_xy: torch.Tensor = torch.tensor(
            [config.image_width, processing_height], dtype=torch.float32
        )
        self.register_buffer("image_size_xy", image_size_xy, persistent=False)
        selection_scores: torch.Tensor = torch.arange(
            config.num_keypoints,
            0,
            -1,
            dtype=torch.float32,
        ) / config.num_keypoints
        self.register_buffer("selection_scores", selection_scores, persistent=False)
        self.processing_height: int = processing_height
        self.processing_width: int = config.image_width

    def extract_batch(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract a batch of normalized keypoints, descriptors, and scores.

        Args:
            image: Float32 images with shape ``[batch, 3, 1200, 1920]`` and values in ``[0, 1]``.

        Returns:
            Keypoints ``[batch, 2048, 2]``, descriptors ``[batch, 2048, 128]``,
            and scores ``[batch, 2048]``.
        """
        processing_image_bchw: torch.Tensor = functional.interpolate(
            image,
            size=(self.processing_height, self.processing_width),
            mode="bilinear",
            align_corners=False,
        )
        keypoints_bnc, descriptors_bnd = self.extractor.extract_for_matching(processing_image_bchw)

        # RaCo returns pixel centers (+0.5). cuSFM stores each coordinate with
        # align_corners=False normalization, so pixel centers remain inside
        # the closed [-1, 1] interval after its own inverse conversion.
        normalized_bnc: torch.Tensor = 2.0 * keypoints_bnc / self.image_size_xy - 1.0
        # Boundary ranking returns its selected points in priority order but does
        # not materialize standalone dense scores. Preserve that order as a
        # strictly monotonic score for cuSFM's downstream top-k/NMS, without
        # pulling the expensive dense rank map back into this graph.
        scores_bn: torch.Tensor = self.selection_scores[None].expand(image.shape[0], -1)
        return normalized_bnc, descriptors_bnd, scores_bn

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract cuSFM's legacy batch-one, unbatched output tensors.

        Args:
            image: Float32 image with shape ``[1, 3, 1200, 1920]`` and values in ``[0, 1]``.

        Returns:
            Keypoints ``[2048, 2]``, descriptors ``[2048, 128]``, and scores ``[2048]``.
        """
        batch_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = self.extract_batch(image)
        return batch_outputs[0][0], batch_outputs[1][0], batch_outputs[2][0]


class BatchedCusfmExtractor(CusfmExtractor):
    """Expose RaCo features with an explicit dynamic batch dimension."""

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract batch-preserving cuSFM tensors.

        Args:
            image: Float32 images with shape ``[batch, 3, 1200, 1920]`` and values in ``[0, 1]``.

        Returns:
            Keypoints ``[batch, 2048, 2]``, descriptors ``[batch, 2048, 128]``,
            and scores ``[batch, 2048]``.
        """
        return self.extract_batch(image)


class DynamicShapeCusfmExtractor(CusfmExtractor):
    """Expose RaCo features with a dynamic batch **and** a dynamic input size.

    The batched graph resamples every input to a fixed 1216x1920 inside itself,
    so a 1226x370 KITTI frame is inferred over five times the pixels it has. This
    variant deletes that resample and normalises against the shape it is handed,
    which makes the network size the caller's choice and therefore the frame's
    own. The one thing the caller inherits is RaCo's pooling: height and width
    must be multiples of `INPUT_DIM_DIVISOR`.
    """

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract batch-preserving cuSFM tensors at the shape supplied.

        Args:
            image: Float32 images with shape ``[batch, 3, height, width]`` and
                values in ``[0, 1]``; height and width multiples of 32.

        Returns:
            Keypoints ``[batch, 2048, 2]``, descriptors ``[batch, 2048, 128]``,
            and scores ``[batch, 2048]``.
        """
        keypoints_bnc, descriptors_bnd = self.extractor.extract_for_matching(image)
        # The static graph divides by a baked `[width, processing_height]`. Here
        # the two sides are symbolic, and `torch.stack` of the two scaled columns
        # is the one form that keeps them symbolic through the Dynamo export
        # rather than freezing the example shape into a constant.
        normalized_x_bn: torch.Tensor = 2.0 * keypoints_bnc[..., 0] / image.shape[3] - 1.0
        normalized_y_bn: torch.Tensor = 2.0 * keypoints_bnc[..., 1] / image.shape[2] - 1.0
        normalized_bnc: torch.Tensor = torch.stack((normalized_x_bn, normalized_y_bn), dim=-1)
        scores_bn: torch.Tensor = self.selection_scores[None].expand(image.shape[0], -1)
        return normalized_bnc, descriptors_bnd, scores_bn






class CusfmLightGlue(nn.Module):
    """Adapt fabio-sim's interleaved RaCo LightGlue to cuSFM's four inputs."""

    matcher: LightGlue

    def __init__(self, config: ExportConfig) -> None:
        super().__init__()
        self.matcher = LightGlue(**Extractor.raco_aliked.lightglue_config)
        processing_height: int = (config.image_height + 31) // 32 * 32
        long_edge_scale_xy: torch.Tensor = torch.tensor(
            [1.0, processing_height / config.image_width], dtype=torch.float32
        )
        self.register_buffer("long_edge_scale_xy", long_edge_scale_xy, persistent=False)

    @staticmethod
    def _pad_points(tensor_bnc: torch.Tensor, point_count: int | torch.SymInt) -> torch.Tensor:
        """Pad a ``[1, n, c]`` tensor to a shared point count."""
        padding: int | torch.SymInt = point_count - tensor_bnc.shape[1]
        return functional.pad(tensor_bnc, (0, 0, 0, padding))

    def forward(
        self,
        kpts0: torch.Tensor,
        kpts1: torch.Tensor,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match two cuSFM feature sets.

        Args:
            kpts0: Float32 normalized keypoints with shape ``[1, n0, 2]``.
            kpts1: Float32 normalized keypoints with shape ``[1, n1, 2]``.
            desc0: Float32 descriptors with shape ``[1, n0, 128]``.
            desc1: Float32 descriptors with shape ``[1, n1, 128]``.

        Returns:
            Int32 match indices ``[n1, 2]`` and Float32 scores ``[n1]``.
            Unused rows have score zero, matching cuSFM's legacy matcher.
        """
        count0: int | torch.SymInt = kpts0.shape[1]
        count1: int | torch.SymInt = kpts1.shape[1]
        shared_count: int | torch.SymInt = torch.sym_max(count0, count1)

        points0_bnc: torch.Tensor = self._pad_points(kpts0 * self.long_edge_scale_xy, shared_count)
        points1_bnc: torch.Tensor = self._pad_points(kpts1 * self.long_edge_scale_xy, shared_count)
        descriptors0_bnd: torch.Tensor = self._pad_points(desc0, shared_count)
        descriptors1_bnd: torch.Tensor = self._pad_points(desc1, shared_count)
        points_2bnc: torch.Tensor = torch.cat((points0_bnc, points1_bnc), dim=0)
        descriptors_2bnd: torch.Tensor = torch.cat((descriptors0_bnd, descriptors1_bnd), dim=0)

        match_results: tuple[torch.Tensor, torch.Tensor] = self.matcher(points_2bnc, descriptors_2bnd)
        matches_m3: torch.Tensor = match_results[0]
        scores_m: torch.Tensor = match_results[1]
        valid_m: torch.Tensor = (matches_m3[:, 1] < count0) & (matches_m3[:, 2] < count1)
        matches_m2: torch.Tensor = torch.where(
            valid_m[:, None],
            matches_m3[:, 1:],
            torch.zeros_like(matches_m3[:, 1:]),
        ).to(torch.int32)
        valid_scores_m: torch.Tensor = torch.where(valid_m, scores_m, torch.zeros_like(scores_m))

        # cuSFM's frozen TensorRTBufferManager cannot allocate data-dependent
        # outputs. The legacy graph likewise scatters sparse matches into a
        # shape-resolved buffer and uses zero scores as the unused-row sentinel.
        destination_m2: torch.Tensor = matches_m3[:, 2:3].expand(-1, 2)
        destination_m: torch.Tensor = matches_m3[:, 2]
        matches_s2: torch.Tensor = torch.zeros(
            (shared_count, 2), dtype=torch.int32, device=matches_m3.device
        ).scatter(0, destination_m2, matches_m2)
        scores_s: torch.Tensor = torch.zeros(
            (shared_count,), dtype=scores_m.dtype, device=scores_m.device
        ).scatter(0, destination_m, valid_scores_m)
        return matches_s2[:count1], scores_s[:count1]


def _integer_division(self: object, other: object, rounding_mode: str | None = None) -> object:
    """Use ONNX integer Div to prevent TensorRT FP16 index overflow."""
    if rounding_mode not in {"floor", "trunc"}:
        raise ValueError(f"Unsupported integer division mode: {rounding_mode}")
    return onnx_op.Div(self, other)


def _stamp_model(path: Path, artifact: str) -> None:
    """Record the pinned upstream source and adapter semantics in ONNX metadata."""
    model: onnx.ModelProto = onnx.load(path)
    model = onnx_optimizer.optimize(model, num_iterations=2)
    metadata: dict[str, str] = {
        "artifact": artifact,
        "source": "https://github.com/fabio-sim/LightGlue-ONNX",
        "source_commit": UPSTREAM_COMMIT,
        "cusfm_keypoints": "align_corners_false_normalized_xy",
        "cusfm_scores": "monotonic_upstream_selection_priority",
    }
    for key, value in metadata.items():
        property_entry: onnx.StringStringEntryProto = model.metadata_props.add()
        property_entry.key = key
        property_entry.value = value
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path, save_as_external_data=False)




def export_batched_extractor(config: ExportConfig, output_path: Path) -> None:
    """Export a batch-dynamic RaCo-ALIKED graph for the direct writer."""
    if config.dynamic_shape:
        export_dynamic_shape_extractor(config, output_path)
        return
    if not (
        1 <= config.minimum_batch_size
        <= config.optimal_batch_size
        <= config.maximum_batch_size
    ):
        raise ValueError(
            "Batch bounds must satisfy 1 <= minimum <= optimal <= maximum, got "
            f"{config.minimum_batch_size}, {config.optimal_batch_size}, {config.maximum_batch_size}"
        )
    module: BatchedCusfmExtractor = BatchedCusfmExtractor(config).eval()
    module.extractor.fuse_batch_norm()
    example_image_bchw: torch.Tensor = torch.zeros(
        config.optimal_batch_size,
        3,
        config.image_height,
        config.image_width,
        dtype=torch.float32,
    )
    batch_dimension: torch.export.Dim = torch.export.Dim(
        "batch",
        min=config.minimum_batch_size,
        max=config.maximum_batch_size,
    )
    dynamic_shapes: tuple[dict[int, object], ...] = ({0: batch_dimension},)
    translation_table: dict[Any, Any] = {torch.ops.aten.div.Tensor_mode: _integer_division}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        (example_image_bchw,),
        str(output_path),
        input_names=["image"],
        output_names=["keypoints", "descriptors", "scores"],
        opset_version=config.opset,
        dynamic_shapes=dynamic_shapes,
        dynamo=True,
        external_data=False,
        optimize=False,
        custom_translation_table=translation_table,
    )
    _stamp_model(output_path, "cusfm-raco-aliked-batched-extractor")


def export_dynamic_shape_extractor(config: ExportConfig, output_path: Path) -> None:
    """Export a batch- and shape-dynamic RaCo-ALIKED graph.

    Every optimisation the fixed-shape export uses is kept: the boundary ranker,
    the portable DeformConv decomposition, folded BatchNorm, the 65536-element
    top-k chunk, upstream's default 2x candidate multiplier, and the integer-Div
    translation that stops TensorRT lowering flattened indices through FP16.

    Args:
        config: Export settings; `image_height` and `image_width` become the
            Dynamo trace's example shape rather than the graph's shape.
        output_path: Where to write the ONNX graph.

    Raises:
        ValueError: When the batch bounds are unordered or the example shape is
            not a usable multiple of `INPUT_DIM_DIVISOR`.
    """
    if not 1 <= config.minimum_batch_size <= config.optimal_batch_size <= config.maximum_batch_size:
        raise ValueError(
            "Batch bounds must satisfy 1 <= minimum <= optimal <= maximum, got "
            f"{config.minimum_batch_size}, {config.optimal_batch_size}, {config.maximum_batch_size}"
        )
    example_height: int = (config.image_height + INPUT_DIM_DIVISOR - 1) // INPUT_DIM_DIVISOR * INPUT_DIM_DIVISOR
    example_width: int = (config.image_width + INPUT_DIM_DIVISOR - 1) // INPUT_DIM_DIVISOR * INPUT_DIM_DIVISOR
    if min(example_height, example_width) < MINIMUM_DYNAMIC_SIDE:
        raise ValueError(f"Boundary ranking needs both sides at least {MINIMUM_DYNAMIC_SIDE}, got {example_height}x{example_width}")

    module: DynamicShapeCusfmExtractor = DynamicShapeCusfmExtractor(config).eval()
    module.extractor.fuse_batch_norm()
    example_image_bchw: torch.Tensor = torch.zeros(
        config.optimal_batch_size, 3, example_height, example_width, dtype=torch.float32
    )
    minimum_factor: int = MINIMUM_DYNAMIC_SIDE // INPUT_DIM_DIVISOR
    batch_dimension: torch.export.Dim = torch.export.Dim(
        "batch", min=config.minimum_batch_size, max=config.maximum_batch_size
    )
    height_factor: torch.export.Dim = torch.export.Dim("height_factor", min=minimum_factor)
    width_factor: torch.export.Dim = torch.export.Dim("width_factor", min=minimum_factor)
    dynamic_shapes: tuple[dict[int, object], ...] = (
        {0: batch_dimension, 2: INPUT_DIM_DIVISOR * height_factor, 3: INPUT_DIM_DIVISOR * width_factor},
    )
    translation_table: dict[Any, Any] = {torch.ops.aten.div.Tensor_mode: _integer_division}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        (example_image_bchw,),
        str(output_path),
        input_names=["image"],
        output_names=["keypoints", "descriptors", "scores"],
        opset_version=config.opset,
        dynamic_shapes=dynamic_shapes,
        dynamo=True,
        external_data=False,
        optimize=False,
        custom_translation_table=translation_table,
    )
    _stamp_model(output_path, "cusfm-raco-aliked-dynamic-shape-extractor")




def export_matcher(config: ExportConfig, output_path: Path) -> None:
    """Export the cuSFM-compatible RaCo-trained LightGlue graph."""
    module: CusfmLightGlue = CusfmLightGlue(config).eval()
    example_points0_bnc: torch.Tensor = torch.zeros(1, config.num_keypoints, 2, dtype=torch.float32)
    example_points1_bnc: torch.Tensor = torch.zeros(1, config.num_keypoints, 2, dtype=torch.float32)
    example_descriptors0_bnd: torch.Tensor = torch.zeros(
        1, config.num_keypoints, 128, dtype=torch.float32
    )
    example_descriptors1_bnd: torch.Tensor = torch.zeros(
        1, config.num_keypoints, 128, dtype=torch.float32
    )
    count0: torch.export.Dim = torch.export.Dim("num_keypoints0", min=128, max=4096)
    count1: torch.export.Dim = torch.export.Dim("num_keypoints1", min=128, max=4096)
    dynamic_shapes: tuple[dict[int, object], ...] = (
        {1: count0},
        {1: count1},
        {1: count0},
        {1: count1},
    )
    torch.onnx.export(
        module,
        (example_points0_bnc, example_points1_bnc, example_descriptors0_bnd, example_descriptors1_bnd),
        str(output_path),
        input_names=["kpts0", "kpts1", "desc0", "desc1"],
        output_names=["matches0", "mscores0"],
        opset_version=config.opset,
        dynamic_shapes=dynamic_shapes,
        dynamo=True,
        external_data=False,
        optimize=False,
    )
    _stamp_model(output_path, "cusfm-raco-lightglue-matcher")




