import os
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from monopriors.third_party.promptda.model.config import model_configs
from monopriors.third_party.promptda.model.dpt import DPTHead


class PromptDA(nn.Module):
    patch_size = 14  # patch size of the pretrained dinov2 model
    use_bn = False
    use_clstoken = False
    output_act = "sigmoid"

    def __init__(self, encoder="vitl", ckpt_path="checkpoints/promptda_vitl.ckpt"):
        super().__init__()
        model_config = model_configs[encoder]

        self.encoder = encoder
        self.model_config = model_config
        # get current working directory using pathlib
        file_dir = Path(__file__).parent
        dinov2_path = file_dir / "torchhub/facebookresearch_dinov2_main"
        assert dinov2_path.exists(), "torchub path not found"
        self.pretrained = torch.hub.load(
            str(dinov2_path),
            f"dinov2_{encoder}14",
            source="local",
            pretrained=False,
        )
        dim = self.pretrained.blocks[0].attn.qkv.in_features
        self.depth_head = DPTHead(
            nclass=1,
            in_channels=dim,
            features=model_config["features"],
            out_channels=model_config["out_channels"],
            use_bn=self.use_bn,
            use_clstoken=self.use_clstoken,
            output_act=self.output_act,
        )

        # mean and std of the pretrained dinov2 model
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.load_checkpoint(ckpt_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, model_kwargs=None, **hf_kwargs):
        """
        Load a model from a checkpoint file.
        ### Parameters:
        - `pretrained_model_name_or_path`: path to the checkpoint file or repo id.
        - `model_kwargs`: additional keyword arguments to override the parameters in the checkpoint.
        - `hf_kwargs`: additional keyword arguments to pass to the `hf_hub_download` function. Ignored if `pretrained_model_name_or_path` is a local path.
        ### Returns:
        - A new instance of `MoGe` with the parameters loaded from the checkpoint.
        """
        ckpt_path = None
        if Path(pretrained_model_name_or_path).exists():
            ckpt_path = pretrained_model_name_or_path
        else:
            cached_checkpoint_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                repo_type="model",
                filename="model.ckpt",
                **hf_kwargs,
            )
            ckpt_path = cached_checkpoint_path
        # model_config = checkpoint['model_config']
        # if model_kwargs is not None:
        #     model_config.update(model_kwargs)
        if model_kwargs is None:
            model_kwargs = {}
        model_kwargs.update({"ckpt_path": ckpt_path})
        model = cls(**model_kwargs)
        return model

    def load_checkpoint(self, ckpt_path):
        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            self.load_state_dict({k[9:]: v for k, v in checkpoint["state_dict"].items()})
        else:
            print(f"Checkpoint {ckpt_path} not found")

    def forward(self, x, prompt_depth=None):
        assert prompt_depth is not None, "prompt_depth is required"
        prompt_depth, min_val, max_val = self.normalize(prompt_depth)
        h, w = x.shape[-2:]
        features = self.pretrained.get_intermediate_layers(
            (x - self._mean) / self._std,
            self.model_config["layer_idxs"],
            return_class_token=True,
        )
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        depth = self.depth_head(features, patch_h, patch_w, prompt_depth)
        depth = self.denormalize(depth, min_val, max_val)
        return depth

    @torch.no_grad()
    def predict(self, image: torch.Tensor, prompt_depth: torch.Tensor):
        return self.forward(image, prompt_depth)

    def normalize(self, prompt_depth: torch.Tensor):
        B, C, H, W = prompt_depth.shape
        min_val = torch.quantile(prompt_depth.reshape(B, -1), 0.0, dim=1, keepdim=True)[:, :, None, None]
        max_val = torch.quantile(prompt_depth.reshape(B, -1), 1.0, dim=1, keepdim=True)[:, :, None, None]
        prompt_depth = (prompt_depth - min_val) / (max_val - min_val)
        return prompt_depth, min_val, max_val

    def denormalize(self, depth: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor):
        return depth * (max_val - min_val) + min_val
