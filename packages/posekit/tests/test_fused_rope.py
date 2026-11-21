"""sam2-streaming fused RoPE: Triton finds the environment's ptxas without activation, and falls back if it cannot build."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("sam2", reason="the streaming SAM 2 fork is in the posekit envs only")

from sam2.modeling import fused_rope  # noqa: E402  (after the importorskip)

_PTXAS_KEYS: tuple[str, str] = ("TRITON_PTXAS_PATH", "TRITON_PTXAS_BLACKWELL_PATH")


def test_use_env_ptxas_sets_unset_paths_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "ptxas").write_text("")
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.delenv("TRITON_PTXAS_PATH", raising=False)
    monkeypatch.setenv("TRITON_PTXAS_BLACKWELL_PATH", "/opt/cuda/bin/ptxas")
    fused_rope.use_env_ptxas()
    assert os.environ["TRITON_PTXAS_PATH"] == str(tmp_path / "bin" / "ptxas")
    assert os.environ["TRITON_PTXAS_BLACKWELL_PATH"] == "/opt/cuda/bin/ptxas"


def test_rotate_turns_fused_path_off_with_one_warning_when_the_kernel_cannot_build(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_ptxas(*args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Cannot find ptxas-blackwell")

    monkeypatch.setattr(fused_rope, "ENABLED", True)
    monkeypatch.setattr(fused_rope, "_built", False)
    monkeypatch.setattr(fused_rope, "rope_", no_ptxas)
    monkeypatch.setattr(fused_rope, "use_env_ptxas", lambda: None)
    q: torch.Tensor = torch.randn(1, 2, 16, 16)
    k: torch.Tensor = torch.randn(1, 2, 32, 16)
    k_before: torch.Tensor = k.clone()
    with pytest.warns(RuntimeWarning, match="fused RoPE is off.*Cannot find ptxas-blackwell"):
        assert fused_rope.rotate_qk_(q, k[:, :, :16], torch.ones(16, 8, dtype=torch.complex64)) is None
    assert fused_rope.ENABLED is False
    assert fused_rope._built is False
    torch.testing.assert_close(k, k_before, rtol=0, atol=0)


_CHILD = r"""
import json, sys, warnings
import torch
from sam2.modeling import fused_rope
from sam2.modeling.sam2_forgetful_memory import SAM2ForgetfulObjectMemoryBank
from sam2.modeling.sam2_prompt import SAM2Prompt
from sam2.sam2_generic_video_predictor import SAM2GenericVideoPredictorState
from posekit.models.sam2_video import Sam2VideoSegmenterConfig

warnings.simplefilter("always")
predictor = Sam2VideoSegmenterConfig(variant="efficienttam-ti-512", fused_rope=True).setup().predictor
assert fused_rope.ENABLED
device = torch.device("cuda")


def scene(step):
    frames = torch.full((2, 256, 256, 3), 40, dtype=torch.uint8)
    frames[0, 80:150, 30 + 4 * step : 100 + 4 * step] = torch.tensor([220, 60, 60], dtype=torch.uint8)
    yy, xx = torch.meshgrid(torch.arange(256), torch.arange(256), indexing="ij")
    frames[1][(yy - 60 - 5 * step) ** 2 + (xx - 128) ** 2 < 35**2] = torch.tensor([60, 200, 90], dtype=torch.uint8)
    return frames.to(device).permute(0, 3, 1, 2)


def run():
    states = [SAM2GenericVideoPredictorState.create((256, 256), SAM2ForgetfulObjectMemoryBank(storage_device=device)) for _ in range(2)]
    first = [[SAM2Prompt(obj_id=0, boxes=torch.tensor([[30.0, 80.0, 100.0, 150.0]], device=device))],
             [SAM2Prompt(obj_id=0, boxes=torch.tensor([[93.0, 25.0, 163.0, 95.0]], device=device))]]
    masks = []
    with torch.inference_mode():
        for step in range(6):
            out = predictor.forward_batched(states, step, scene(step), first if step == 0 else [[], []], [False, False])
            masks.append([out[s][0].masks_logits > 0 for s in range(2)])
    return masks


fused = run()
built = fused_rope._built
fused_rope.ENABLED = False
unfused = run()
ious = [float((a & b).sum() / (a | b).sum().clamp_min(1)) for fs, us in zip(fused, unfused) for a, b in zip(fs, us)]
nonempty = all(bool(m.any()) for ms in unfused for m in ms)
print(json.dumps({"built": built, "ious": ious, "equal": all(bool(torch.equal(a, b)) for fs, us in zip(fused, unfused) for a, b in zip(fs, us)), "nonempty": nonempty}))
"""


@pytest.mark.integration
def test_fused_rope_builds_without_env_activation_and_matches_unfused_masks(tmp_path: Path) -> None:
    """The pipeline driver runs ``<env>/bin/python`` with no ``CONDA_PREFIX``; Triton then misses ptxas-blackwell."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU (EfficientTAM-ti from the Hugging Face hub)")
    env_bin: str = str(Path(sys.prefix) / "bin")
    env: dict[str, str] = {k: v for k, v in os.environ.items() if k != "CONDA_PREFIX" and k not in _PTXAS_KEYS}
    env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep) if p != env_bin)
    env["TRITON_CACHE_DIR"] = str(tmp_path / "triton-cache")  # an empty cache, so the kernel compiles here
    result = subprocess.run([sys.executable, "-c", _CHILD], env=env, capture_output=True, text=True, timeout=600, check=False)
    assert result.returncode == 0, result.stderr[-4000:]
    report: dict = json.loads(result.stdout.strip().splitlines()[-1])
    if (Path(env_bin) / "ptxas").is_file():
        assert report["built"], result.stderr[-4000:]
        assert "fused RoPE is off" not in result.stderr
    else:
        assert result.stderr.count("fused RoPE is off") == 1, result.stderr[-4000:]
    assert report["nonempty"]
    assert min(report["ious"]) > 0.999, report
