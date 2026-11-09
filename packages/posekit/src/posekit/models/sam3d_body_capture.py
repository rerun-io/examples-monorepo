"""Capture-safe execution of the released SAM body decoders."""

from collections.abc import Callable
from functools import wraps
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from sam_3d_body.models.decoders.prompt_encoder import PromptEncoder
from sam_3d_body.models.heads.camera_head import PerspectiveHead
from sam_3d_body.models.meta_arch.sam3d_body import SAM3DBody
from torch import Tensor, nn
from torch.overrides import TorchFunctionMode

from posekit.models.sam3d_body import Sam3dBodyStage, _StageClock
from posekit.models.sam3d_body_head import CaptureMhrHead
from posekit.models.sam3d_body_keypoints import sample_keypoint_features
from posekit.ops.compile import compile_unless_dev
from posekit.ops.cuda_graph import CapturedGraph, GraphPool

PoseFields: TypeAlias = dict[str, Tensor | np.ndarray | None]
DecoderOutput: TypeAlias = tuple[Tensor, list[PoseFields]]
def _compile_geometry[**Parameters, Result](function: Callable[Parameters, Result]) -> Callable[Parameters, Result]:
    """Keep the decoder's temporary index adapter outside compiled geometry."""
    compiled = compile_unless_dev(function)
    if compiled is function:
        # Eager dev execution still needs the enclosing static-index adapter.
        return function

    @wraps(function)
    def run(*args: Parameters.args, **kwargs: Parameters.kwargs) -> Result:
        with torch._C.DisableTorchFunction():
            return compiled(*args, **kwargs)

    return run


def prepare_decoder_projection(model: SAM3DBody) -> None:
    """Compile released camera math and omit vertex projections unused by the GPU role."""
    for name in ("head_camera", "head_camera_hand"):
        head = model.get_submodule(name)
        assert isinstance(head, PerspectiveHead)
        object.__setattr__(head, "perspective_projection", _compile_geometry(head.perspective_projection))
    model._full_to_crop = _compile_geometry(model._full_to_crop)

    def keypoints_only(project: Callable[[PoseFields, dict[str, Tensor]], PoseFields]) -> Callable[[PoseFields, dict[str, Tensor]], PoseFields]:
        def run(pose_output: PoseFields, batch: dict[str, Tensor]) -> PoseFields:
            vertices = pose_output.pop("pred_vertices")
            try:
                return project(pose_output, batch)
            finally:
                pose_output["pred_vertices"] = vertices

        return run

    object.__setattr__(model, "camera_project", keypoints_only(model.camera_project))
    object.__setattr__(model, "camera_project_hand", keypoints_only(model.camera_project_hand))


class _DecoderTensorMode(TorchFunctionMode):
    """Resolve static indices and sparse feature sampling before graph capture.

    PyTorch's eager list indexing makes a pageable host tensor on every call.
    Warm-up populates this cache; capture then reads the same device indices.
    The mode applies only while recording the released decoder, never on replay.
    """

    def __init__(self) -> None:
        self.indices: dict[tuple[torch.device, tuple[int, ...]], Tensor] = {}

    def __torch_function__(
        self, func: Callable, types: tuple[type, ...], args: tuple = (), kwargs: dict | None = None,
    ):
        kwargs = {} if kwargs is None else kwargs
        if (
            func is F.grid_sample and args[0].ndim == 4 and args[1].shape[-2] == 1
            and kwargs.get("mode", "bilinear") == "bilinear" and kwargs.get("padding_mode", "zeros") == "zeros"
            and kwargs.get("align_corners") is False
        ):
            with torch._C.DisableTorchFunction():
                sampled: Tensor = sample_keypoint_features(args[0], args[1].squeeze(2))
            return sampled.transpose(1, 2).unsqueeze(-1)
        if func in (Tensor.__getitem__, Tensor.__setitem__) and args[0].is_cuda:
            index = args[1]
            axes: tuple = index if isinstance(index, tuple) else (index,)
            converted: list = []
            for axis in axes:
                if isinstance(axis, list) and all(isinstance(value, int) for value in axis):
                    key: tuple[torch.device, tuple[int, ...]] = (args[0].device, tuple(axis))
                    if key not in self.indices:
                        self.indices[key] = torch.tensor(axis, dtype=torch.int64, device=args[0].device)
                    axis = self.indices[key]
                converted.append(axis)
            args = (args[0], tuple(converted) if isinstance(index, tuple) else converted[0], *args[2:])
        return func(*args, **kwargs)


@runtime_checkable
class DecoderForward(Protocol):
    """The released decoder entry point, before capture."""

    def __call__(
        self, image_embeddings: Tensor, init_estimate: Tensor | None = None,
        keypoints: Tensor | None = None, prev_estimate: Tensor | None = None,
        condition_info: Tensor | None = None, batch: dict[str, Tensor] | None = None,
    ) -> DecoderOutput: ...


class CapturePromptEncoder(PromptEncoder):
    """Gather learned prompt embeddings without data-dependent indexing or host checks."""

    def __init__(self, encoder: PromptEncoder) -> None:
        nn.Module.__init__(self)
        self.encoder: PromptEncoder = encoder
        self.labels: Tensor = torch.cat([
            encoder.invalid_point_embed.weight, encoder.not_a_point_embed.weight,
            *(entry.get_parameter("weight") for entry in encoder.point_embeddings),
        ])
        self.dense: dict[tuple[int, int], Tensor] = {}

    def forward(self, keypoints: Float[Tensor, "b p 3"] | None, boxes: Tensor | None = None, masks: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Encode already normalized coordinates and labels -2, -1, or a joint index."""
        assert keypoints is not None and boxes is None and masks is None
        labels: Tensor = keypoints[..., 2].long()
        positional: Tensor = self.encoder.pe_layer._pe_encoding(keypoints[..., :2].float())
        embedding: Tensor = torch.where((labels >= 0)[..., None], positional, 0.0) + self.labels[(labels + 2).clamp_min(0)]
        return embedding, labels > -2

    def get_dense_pe(self, size: tuple[int, int]) -> Tensor:
        """Cache only the position encoding, which depends on feature shape and fixed weights."""
        if size not in self.dense:
            self.dense[size] = self.encoder.get_dense_pe(size)
        return self.dense[size]

    # Upstream annotates PromptEncoder.get_mask_embeddings as returning a Tensor; it returns (mask, no-mask) embeddings.
    def get_mask_embeddings(self, masks: Tensor | None = None, bs: int = 1, size: tuple[int, int] = (16, 16)) -> tuple[Tensor, Tensor]:  # pyrefly: ignore[bad-override]
        """Keep the released mask conditioning."""
        return self.encoder.get_mask_embeddings(masks, bs, size)  # pyrefly: ignore[bad-return]


class CapturedBodyDecoder:
    """One graph per batch size and prompt shape, independent of prompt values."""

    def __init__(
        self, forward: DecoderForward, pool: GraphPool, clock: _StageClock, *, hand: bool = False,
        full_mesh: CaptureMhrHead | None = None, prompt_decoder: nn.Module | None = None, capture: bool = True,
    ) -> None:
        self.forward: DecoderForward = forward
        self.pool: GraphPool = pool
        self.hand: bool = hand
        self.clock: _StageClock = clock
        self.full_mesh: CaptureMhrHead | None = full_mesh
        self.capture: bool = capture
        self.prompt_valid: Tensor | None = None
        self.pad_prompts: bool = prompt_decoder is not None
        if prompt_decoder is not None:
            prompt_decoder.register_forward_pre_hook(self._mask_prompts)
        self.index_mode: _DecoderTensorMode = _DecoderTensorMode()
        self.graphs: dict[tuple[int, int, bool], CapturedGraph[tuple[Tensor, PoseFields]]] = {}

    def _mask_prompts(self, module: nn.Module, args: tuple) -> tuple | None:
        """Exclude padded prompt tokens from self-attention and image-to-token attention."""
        if self.prompt_valid is None:
            return None
        tokens: Tensor = args[0]
        mask: Tensor = tokens.new_ones(tokens.shape[:2])
        mask[:, 2 : 2 + self.prompt_valid.shape[1]] = self.prompt_valid
        return (*args[:4], mask, *args[5:])

    def run(
        self, image_embeddings: Tensor, keypoints: Tensor, condition_info: Tensor,
        batch: dict[str, Tensor], prev_estimate: Tensor | None,
    ) -> DecoderOutput:
        """Execute capture-safe decoder math, also usable inside a whole-frame graph."""
        self.prompt_valid = keypoints[..., 2] != -3 if self.pad_prompts and prev_estimate is not None else None
        with self.index_mode:
            output = self.forward(image_embeddings, keypoints=keypoints, condition_info=condition_info, prev_estimate=prev_estimate, batch=batch)
        pose = output[1][-1]
        if self.full_mesh is not None:
            self.full_mesh.complete_mesh(pose)
        return output[0], [pose]

    def __call__(
        self, image_embeddings: Tensor, init_estimate: Tensor | None = None,
        keypoints: Tensor | None = None, prev_estimate: Tensor | None = None,
        condition_info: Tensor | None = None, batch: dict[str, Tensor] | None = None,
    ) -> DecoderOutput:
        """Replay the decoder and return owned outputs, safe across the next replay."""
        assert init_estimate is None and keypoints is not None and condition_info is not None and batch is not None
        if not self.capture:
            return self.run(image_embeddings, keypoints, condition_info, batch, prev_estimate)
        names: tuple[str, ...] = (
            "bbox_center", "bbox_scale", "ori_img_size", "cam_int", "affine_trans", "img_size",
            "ray_cond_hand" if self.hand else "ray_cond",
        )
        values: tuple[Tensor, ...] = (image_embeddings, keypoints, condition_info, *(batch[name] for name in names))
        if prev_estimate is not None:
            values += (prev_estimate,)
        key: tuple[int, int, bool] = (len(image_embeddings), keypoints.shape[1], prev_estimate is not None)
        if key not in self.graphs:
            storage: tuple[Tensor, ...] = tuple(value.clone() for value in values)
            captured_batch: dict[str, Tensor] = dict(zip(names, storage[3 : 3 + len(names)], strict=True))
            # The decoder reads only the person dimension of img, never its pixels.
            captured_batch["img"] = batch["img"]

            def run() -> tuple[Tensor, PoseFields]:
                output = self.run(storage[0], storage[1], storage[2], captured_batch, storage[-1] if prev_estimate is not None else None)
                return output[0], output[1][-1]

            enabled: bool = self.clock.enabled
            self.clock.enabled = False
            try:
                self.graphs[key] = CapturedGraph(run, storage, pool=self.pool, keep=(self.forward, captured_batch))
            finally:
                self.clock.enabled = enabled
        events: tuple[torch.cuda.Event, torch.cuda.Event] | None = (
            (torch.cuda.Event(enable_timing=True, external=True), torch.cuda.Event(enable_timing=True, external=True)) if self.clock.enabled else None
        )
        if events is not None:
            events[0].record()
        tokens, pose = self.graphs[key].replay(*values)
        result: DecoderOutput = tokens.clone(), [{name: value.clone() if isinstance(value, Tensor) else value for name, value in pose.items()}]
        if events is not None:
            events[1].record()
            labels: dict[str, Sam3dBodyStage] = {
                "body": "decoder_body", "left_hand": "decoder_left_hand", "right_hand": "decoder_right_hand", "prompt": "decoder_prompt",
                "hands": "decoder_hands",
            }
            self.clock.events.append((labels[self.clock.current], *events))
        return result
