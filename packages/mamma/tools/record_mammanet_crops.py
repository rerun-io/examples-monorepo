"""Record real MammaNet engine inputs from the streaming benchmark (FP8 calibration and real-crop parity).

Runs ``tools/benchmark.py`` unchanged (same CLI, needs ``--trt-engine``) and
keeps every ``--every``-th full 4-crop batch handed to the TensorRT runtime,
saved as fp16 ``{"crops": [n, 4, 3, h, w], "masks": [n, 4, 1, h, w]}``.
Consumed by ``tools/build_trt_engine.py --calib-crops``.

    python tools/record_mammanet_crops.py --out .trt_cache/mammanet_calib_crops.pt -- <benchmark args>
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro


@dataclass
class RecordConfig:
    out: Path = Path(".trt_cache/mammanet_calib_crops.pt")
    """Output file."""
    every: int = 6
    """Keep one full batch out of this many engine calls."""


def main(config: RecordConfig, benchmark_args: list[str]) -> int:
    from trtkit import tensorrt_runtime

    sys.path.insert(0, str(Path(__file__).parent))
    import benchmark  # pyrefly: ignore[missing-import]  # sibling tool script

    crops: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    calls: list[int] = [0]
    run = tensorrt_runtime.TensorRtRuntime.__call__

    def recording_call(self: tensorrt_runtime.TensorRtRuntime, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "crops" in inputs and inputs["crops"].shape[0] == self.spec.max_batch_size:
            calls[0] += 1
            if calls[0] % config.every == 0:
                crops.append(inputs["crops"].detach().half().cpu())
                masks.append(inputs["masks"].detach().half().cpu())
        return run(self, inputs)

    tensorrt_runtime.TensorRtRuntime.__call__ = recording_call  # pyrefly: ignore[bad-assignment]
    status: int = benchmark.main(tyro.cli(benchmark.BenchmarkConfig, args=benchmark_args))
    config.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"crops": torch.stack(crops), "masks": torch.stack(masks)}, config.out)
    print(f"saved {len(crops)} batches (of {calls[0]} full engine calls) -> {config.out}")
    return status


if __name__ == "__main__":
    split: int = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    sys.exit(main(tyro.cli(RecordConfig, args=sys.argv[1:split]), sys.argv[split + 1 :]))
