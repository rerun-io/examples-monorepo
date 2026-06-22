"""TensorRT backend for MammaNet (export -> engine -> CUDA-graph-wrapped runner).

Follows the proven sapiens2-pose pattern in this monorepo: `tensorrt-cu12`
python API (not torch-tensorrt, whose torch-2.10 wheels are cu13-only), static
batch, FP16 builder flag, and one captured ``execute_async_v3`` launch replayed
per call — which composes cleanly with the fitter's manual CUDA graph.

Engines are machine-local artifacts (sm-specific): built once into
``.trt_cache/`` by ``tools/build_trt_engine.py``, never committed.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch
from jaxtyping import Float32

from mamma.landmarks.config import DEFAULT_MAMMANET_CONFIG, MammaNetConfig
from mamma.landmarks.mammanet import MammaNet

ENGINE_BATCH: int = 4
"""Static batch baked into the engine (cameras x persons rounded up)."""


class _ExportWrapper(torch.nn.Module):
    """Tuple-returning wrapper (ONNX needs flat outputs, not a dict)."""

    def __init__(self, model: MammaNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, masks: torch.Tensor):
        out = self.model(x, masks)
        return out["joints2d"], out["visibility"], out["contact"], out["floor_contact"]


def export_mammanet_onnx(model: MammaNet, onnx_path: Path, config: MammaNetConfig = DEFAULT_MAMMANET_CONFIG) -> None:
    """Export MammaNet to a static-batch ONNX graph.

    The dynamo exporter's output failed TRT 10.13's parser ("Failed to import
    initializer"), so this uses the legacy TorchScript exporter at opset 17 —
    the same combination sapiens2-pose ships.
    """
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _ExportWrapper(model).eval().cuda()
    x = torch.randn(ENGINE_BATCH, 3, config.crop_height, config.crop_width, device="cuda")
    masks = torch.rand(ENGINE_BATCH, 1, config.crop_height, config.crop_width, device="cuda")
    torch.onnx.export(
        wrapper,
        (x, masks),
        str(onnx_path),
        input_names=["crops", "masks"],
        output_names=["joints2d", "visibility", "contact", "floor_contact"],
        opset_version=17,
        dynamo=False,
    )


def build_engine(onnx_path: Path, engine_path: Path) -> None:
    """Build an FP16 TensorRT engine from the exported ONNX graph."""
    trt: Any = importlib.import_module("tensorrt")  # no type stubs shipped

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{errors}")
    builder_config = builder.create_builder_config()
    if hasattr(trt.BuilderFlag, "TF32"):
        builder_config.clear_flag(trt.BuilderFlag.TF32)
    builder_config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, builder_config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))


class MammaNetTrtRunner:
    """Static-batch TRT inference with a captured launch (torch-tensor I/O)."""

    def __init__(self, engine_path: Path, config: MammaNetConfig = DEFAULT_MAMMANET_CONFIG) -> None:
        trt: Any = importlib.import_module("tensorrt")  # no type stubs shipped

        self.config = config
        logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(logger)
        engine = self._runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"could not deserialize engine: {engine_path}")
        self._engine: Any = engine
        self._context: Any = engine.create_execution_context()
        self._trt = trt
        dev = torch.device("cuda")
        self._in_crops: Float32[torch.Tensor, "b 3 ch cw"] = torch.empty(
            ENGINE_BATCH, 3, config.crop_height, config.crop_width, device=dev
        )
        self._in_masks: Float32[torch.Tensor, "b 1 ch cw"] = torch.empty(
            ENGINE_BATCH, 1, config.crop_height, config.crop_width, device=dev
        )
        n = config.num_landmarks
        self._outs: dict[str, torch.Tensor] = {
            "joints2d": torch.empty(ENGINE_BATCH, n, 3, device=dev),
            "visibility": torch.empty(ENGINE_BATCH, n, 1, device=dev),
            "contact": torch.empty(ENGINE_BATCH, n, 1, device=dev),
            "floor_contact": torch.empty(ENGINE_BATCH, n, 1, device=dev),
        }
        self._context.set_tensor_address("crops", int(self._in_crops.data_ptr()))
        self._context.set_tensor_address("masks", int(self._in_masks.data_ptr()))
        for name, tensor in self._outs.items():
            self._context.set_tensor_address(name, int(tensor.data_ptr()))
        self._graph: torch.cuda.CUDAGraph | None = None

    def _capture(self) -> None:
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            ok = bool(self._context.execute_async_v3(stream_handle=int(warmup.cuda_stream)))
        if not ok:
            raise RuntimeError("TRT warmup launch failed")
        torch.cuda.current_stream().wait_stream(warmup)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            stream = torch.cuda.current_stream()
            if not bool(self._context.execute_async_v3(stream_handle=int(stream.cuda_stream))):
                raise RuntimeError("TRT capture launch failed")
        self._graph = graph

    def __call__(
        self,
        crops: Float32[torch.Tensor, "k 3 ch cw"],
        masks: Float32[torch.Tensor, "k 1 ch cw"],
    ) -> dict[str, torch.Tensor]:
        """Run up to ENGINE_BATCH crops; rows beyond ``k`` are padding."""
        k: int = crops.shape[0]
        if k > ENGINE_BATCH:
            raise ValueError(f"engine batch is {ENGINE_BATCH}, got {k} crops — chunk upstream")
        if self._graph is None:
            self._capture()
        self._in_crops[:k].copy_(crops)
        self._in_masks[:k].copy_(masks)
        assert self._graph is not None
        self._graph.replay()
        return {name: tensor[:k] for name, tensor in self._outs.items()}
