"""One-time conversion of the MammaNet PyTorch-Lightning checkpoint to safetensors.

Run this inside the ORIGINAL mamma repo's pixi environment (python 3.11,
torch 2.10, pytorch-lightning installed) — NOT in the monorepo env — so the
Lightning pickle can always be unpickled:

    /home/pablo/0Dev/repos/mamma/.claude/worktrees/optimize-10x/.pixi/envs/default/bin/python \
        tools/convert_ckpt.py \
        --ckpt /home/pablo/.codex/worktrees/c737/mamma/data/weights/ma_2d/mamma_mask_full_cvpr.ckpt

Outputs, next to the input ckpt:
  - mamma_mask_full_cvpr.safetensors  (flat state_dict, fp32, contiguous)
  - mamma_mask_full_cvpr.manifest.json (key -> shape/dtype map + hparams snapshot)

The monorepo's `mamma.landmarks.mammanet.load_mammanet` loads the safetensors
file directly and never touches the Lightning pickle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to mamma_mask_full_cvpr.ckpt")
    args = parser.parse_args()

    ckpt_path: Path = args.ckpt
    try:
        ckpt: dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        load_mode: str = "weights_only=True"
    except Exception as exc:  # noqa: BLE001 — fall back to full unpickle in the frozen env
        print(f"weights_only load failed ({type(exc).__name__}); retrying with full unpickle")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        load_mode = "weights_only=False"

    state_dict: dict[str, torch.Tensor] = ckpt.get("state_dict", ckpt)
    prefixes: set[str] = {k.split(".", 1)[0] for k in state_dict}
    print(f"loaded via {load_mode}; {len(state_dict)} tensors; top-level prefixes: {sorted(prefixes)}")

    # Keep keys verbatim — the ported MammaNet module mirrors the original
    # LightningModule attribute names, so no prefix stripping is needed.
    flat: dict[str, torch.Tensor] = {k: v.contiguous().clone() for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    skipped: list[str] = [k for k, v in state_dict.items() if not isinstance(v, torch.Tensor)]
    if skipped:
        print(f"skipped {len(skipped)} non-tensor entries: {skipped[:5]}")

    from safetensors.torch import save_file

    out_st: Path = ckpt_path.with_suffix(".safetensors")
    save_file(flat, str(out_st))

    hparams = ckpt.get("hyper_parameters") or ckpt.get("hparams")
    manifest: dict = {
        "source_ckpt": str(ckpt_path),
        "load_mode": load_mode,
        "num_tensors": len(flat),
        "keys": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in flat.items()},
        "hyper_parameters_repr": repr(hparams)[:20000] if hparams is not None else None,
        "extra_ckpt_keys": [k for k in ckpt if k != "state_dict"],
    }
    out_manifest: Path = ckpt_path.with_suffix(".manifest.json")
    out_manifest.write_text(json.dumps(manifest, indent=2))

    size_gb: float = out_st.stat().st_size / 1e9
    print(f"wrote {out_st} ({size_gb:.2f} GB) and {out_manifest}")


if __name__ == "__main__":
    main()
