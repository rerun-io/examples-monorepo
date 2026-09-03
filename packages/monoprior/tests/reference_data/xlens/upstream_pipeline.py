"""XLensInference: load a released checkpoint and run inference.

A single Stage-3 checkpoint serves all three camera regimes; the mode is chosen
purely by the inputs you pass (see ``xlens.inference.preprocess``):

    pinhole        cam_types = 1, d_cam from K
    fisheye        cam_types = 0, d_cam from a calibration LUT
    heterogeneous  a per-view mix of the two

The model architecture (backbone, calibration tokens, distortion bias, ...) is
read from a ``.pth`` checkpoint itself; for a bare ``.safetensors`` (weights only)
pass the accompanying arch config YAML via ``config=``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class XLensInference:
    def __init__(self, checkpoint_path: str, device: str = "cuda",
                 amp_dtype: str = "bf16", config: Optional[str] = None):
        from ..models import XLensNet

        self.device = torch.device(device)
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

        if str(checkpoint_path).endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(str(checkpoint_path))
            cfg: Dict[str, Any] = {}
        else:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
            state = ckpt.get("model_state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
        # Optional external arch config (YAML). Required for a bare .safetensors (which
        # carries no embedded config); merged over any embedded config when provided.
        if config is not None:
            import yaml
            with open(config) as _f:
                cfg = {**cfg, **(yaml.safe_load(_f) or {})}
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

        # Infer heterogeneous-camera arch from the calibration-token weights so a
        # pinhole-only checkpoint and a full stage-3 checkpoint both load cleanly.
        n_cam_types = cfg.get("n_cam_types", 0)
        calib_per_type = cfg.get("calib_tokens_per_type", 4)
        tok = state.get("backbone.pretrained.calib_tokens.tokens")
        if tok is not None:                                   # (L, n_cam_types, K, dim)
            n_cam_types, calib_per_type = tok.shape[1], tok.shape[2]

        model = XLensNet(
            backbone_name=cfg.get("backbone", "vitl"),
            checkpoint_dir=cfg.get("checkpoint_dir", "checkpoints"),
            head_features=cfg.get("head_features", 256),
            head_out_channels=tuple(cfg.get("head_out_channels", [256, 512, 1024, 1024])),
            freeze_backbone=False,
            predict_mask=cfg.get("predict_mask", False),
            scale_head_mode=cfg.get("scale_head_mode", "cls"),
            scale_head_num_queries=cfg.get("scale_head_num_queries", 4),
            scale_head_num_heads=cfg.get("scale_head_num_heads", 8),
            use_dwc=cfg.get("use_dwc", False),
            dwc_kernel_size=cfg.get("dwc_kernel_size", 3),
            n_cam_types=n_cam_types,
            use_calib_tokens=cfg.get("use_calib_tokens", False),
            calib_tokens_per_type=calib_per_type,
            calib_inject_types=tuple(cfg.get("calib_token_inject_types", [0])),
            use_distortion_bias=cfg.get("use_distortion_bias", False),
            distortion_bias_layers=cfg.get("distortion_bias_layers", "global_only"),
            distortion_bias_hidden_dim=cfg.get("distortion_bias_hidden_dim", 64),
            distortion_bias_chunk_size=cfg.get("distortion_bias_chunk_size", 1024),
        ).to(self.device)

        info = model.load_state_dict(state, strict=False)
        if info.missing_keys:
            logger.warning("missing keys: %d", len(info.missing_keys))
        if info.unexpected_keys:
            logger.warning("unexpected keys: %d", len(info.unexpected_keys))
        model.eval()
        self.model = model
        logger.info("loaded %s (%.1fM params, n_cam_types=%d)", checkpoint_path,
                    sum(p.numel() for p in model.parameters()) / 1e6, n_cam_types)

    @torch.no_grad()
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Run the model. ``batch`` is what ``preprocess.assemble_batch`` returns.

        Returns the raw model output dict, with tensors moved to CPU float32:
            depth            (B, S, H, W)   normalized-space depth
            depth_metric     (B, S, H, W)   metric depth = depth * scale
            depth_conf       (B, S, H, W)
            metric_scaling_factor (B,)
            ray_direction / ray_origin / ray_conf  (world-frame ray head)
            mask             (optional)
        """
        with torch.autocast("cuda", enabled=self.device.type == "cuda", dtype=self.amp_dtype):
            out = self.model(
                batch["images"],
                ray_map=batch.get("ray_map"),
                d_cam=batch.get("d_cam"),
                cam_types=batch.get("cam_types"),
            )
        return {k: (v.detach().float().cpu() if torch.is_tensor(v) else v) for k, v in out.items()}
