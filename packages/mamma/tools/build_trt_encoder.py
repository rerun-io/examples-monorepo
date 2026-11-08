"""Build the machine-local EfficientTAM image-encoder TensorRT engine and check it against eager.

Exports ``predictor.image_encoder`` (fp16-typed ONNX, static camera batch),
builds a strongly-typed engine into ``.trt_cache/``, then prints per-level
parity against the eager bf16-autocast ``encode_image`` and the latency of
both paths. The pipeline loads the engine via ``--tracker.trt-encoder-engine``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import tyro

from mamma.tracking.tracker import TrackerConfig
from mamma.tracking.trt_encoder import ENCODER_BATCH, TrtImageEncoder, build_encoder_engine, export_encoder_onnx


@dataclass
class BuildEncoderConfig:
    tracker: TrackerConfig
    """Tracker settings naming the SAM2-family config and checkpoint to export."""
    cache_dir: Path = Path(".trt_cache")
    """Machine-local engine cache."""
    frames: Path | None = None
    """Optional ``torch.save``'d uint8 ``[n, b, 3, h, w]`` camera batches for parity on real frames."""
    frame_hw: tuple[int, int] = (720, 1280)
    """Random-frame size when ``frames`` is not given (the pipeline's engine resolution)."""
    force: bool = False
    """Rebuild even if the engine exists."""


def engine_path_for(cache_dir: Path, config_name: str) -> Path:
    """Engine cache path keyed by model config, batch, TensorRT version, and compute capability."""
    import tensorrt as trt

    sm: str = "sm{}{}".format(*torch.cuda.get_device_capability())
    stem: str = Path(config_name).stem
    return cache_dir / f"{stem}_encoder_b{ENCODER_BATCH}_fp16_trt{trt.__version__.replace('.', '')}_{sm}.plan"


def _time_ms(fn, batch: torch.Tensor, iters: int = 50) -> tuple[float, float]:
    """(GPU ms via CUDA events, host wall ms) per call after warmup."""
    for _ in range(5):
        fn(batch)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall0: float = time.perf_counter()
    start.record()
    for _ in range(iters):
        fn(batch)
    end.record()
    wall_ms: float = (time.perf_counter() - wall0) / iters * 1000
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, wall_ms


def main(config: BuildEncoderConfig) -> None:
    from sam2.build_sam import build_sam2_generic_video_predictor

    predictor = build_sam2_generic_video_predictor(config.tracker.sam2_config, str(config.tracker.sam2_checkpoint), device="cuda")
    engine_path: Path = engine_path_for(config.cache_dir, config.tracker.sam2_config)
    if not engine_path.exists() or config.force:
        onnx_path: Path = engine_path.with_suffix(".onnx")
        print("exporting ONNX (dynamo exporter, fp16-typed graph)...")
        export_encoder_onnx(predictor, onnx_path)
        print(f"building strongly-typed engine ({engine_path.name})...")
        t0: float = time.perf_counter()
        build_encoder_engine(onnx_path, engine_path)
        print(f"built in {time.perf_counter() - t0:.0f}s -> {engine_path}")
    else:
        print(f"engine exists: {engine_path}")

    encoder: TrtImageEncoder = TrtImageEncoder(predictor, engine_path)
    if config.frames is not None:
        batches: torch.Tensor = torch.load(config.frames)
    else:
        torch.manual_seed(0)
        batches = torch.randint(0, 256, (4, ENCODER_BATCH, 3, *config.frame_hw), dtype=torch.uint8)

    def eager(batch: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return cast("tuple[list[torch.Tensor], list[torch.Tensor]]", predictor.encode_image(batch))

    def trt(batch: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return encoder(batch)

    def eager_fp32(batch: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        with torch.inference_mode():
            return cast("tuple[list[torch.Tensor], list[torch.Tensor]]", predictor.encode_image(batch))

    # Per level: (max abs, rel L2) of eager bf16 and TensorRT fp16 vs eager fp32.
    stats: dict[str, list[tuple[float, float]]] = {"eager bf16": [], "TensorRT fp16": []}
    for batch in batches:
        batch = batch.cuda()
        ref_emb, ref_pos = eager_fp32(batch)
        for label, fn in (("eager bf16", eager), ("TensorRT fp16", trt)):
            out_emb, out_pos = fn(batch)
            assert [t.shape for t in out_emb] == [t.shape for t in ref_emb] and [t.dtype for t in out_emb] == [t.dtype for t in ref_emb]
            assert all(torch.allclose(a, b) for a, b in zip(out_pos, ref_pos, strict=True)), "position encodings differ"
            stats[label].extend(
                ((out - ref).abs().max().item(), ((out - ref).norm() / ref.norm()).item()) for ref, out in zip(ref_emb, out_emb, strict=True)
            )
    for label, rows in stats.items():
        print(
            f"{label:14s} vs eager fp32 ({len(rows)} level-batches): max abs {max(r[0] for r in rows):.4f}, "
            f"rel L2 mean {sum(r[1] for r in rows) / len(rows):.4f} max {max(r[1] for r in rows):.4f}"
        )

    sample: torch.Tensor = batches[0].cuda()
    for label, fn in (("eager bf16", eager), ("TensorRT fp16", trt)):
        gpu_ms, wall_ms = _time_ms(fn, sample)
        print(f"{label:14s}: {gpu_ms:6.2f} ms GPU (stream-timed) / {wall_ms:6.2f} ms host per {ENCODER_BATCH}-camera call")


if __name__ == "__main__":
    main(tyro.cli(BuildEncoderConfig))
