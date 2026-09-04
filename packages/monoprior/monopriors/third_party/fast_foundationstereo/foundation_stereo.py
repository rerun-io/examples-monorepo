"""Owned Fast-FoundationStereo inference architecture."""

from typing import Literal, TypeAlias

import torch
import torch.nn.functional as F
from jaxtyping import Float, Float32, Int8
from torch import Tensor, nn

from monopriors.third_party.fast_foundationstereo.extractor import ConfigLike, ContextNetOutput, ContextNetSharedBackbone, Feature, FeaturePyramid
from monopriors.third_party.fast_foundationstereo.geometry import Combined_Geo_Encoding_Volume
from monopriors.third_party.fast_foundationstereo.submodule import (
    BasicConv,
    BasicConv_IN,
    ChannelAttentionEnhancement,
    Conv2x,
    Conv3dNormActReduced,
    CostVolumeDisparityAttention,
    FeatureAtt,
    ResnetBasicBlock3D,
    SpatialAttentionExtractor,
    build_concat_volume_optimized_pytorch1,
    build_gwc_volume_optimized_pytorch1,
    build_gwc_volume_triton,
    context_upsample,
    disparity_regression,
)
from monopriors.third_party.fast_foundationstereo.update import BasicSelectiveMultiUpdateBlock, TensorPyramid, UpdateBlockOutput
from monopriors.third_party.fast_foundationstereo.utils import InputPadder

AMP_DTYPE: torch.dtype = torch.float16
VolumeBuilder: TypeAlias = Literal["pytorch1", "triton"]


class FoundationStereo(nn.Module):
    """Historical checkpoint base-class name retained for pickle compatibility."""


def normalize_image(image_b3hw: Float[Tensor, "b 3 h w"]) -> Float[Tensor, "b 3 h w"]:
    """Normalize RGB images in the 0-255 range with ImageNet statistics.

    Args:
        image_b3hw: Floating-point RGB tensor with shape ``(batch, 3, height, width)``.

    Returns:
        Normalized RGB tensor with shape ``(batch, 3, height, width)``.
    """
    mean_1311: Float[Tensor, "1 3 1 1"] = image_b3hw.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std_1311: Float[Tensor, "1 3 1 1"] = image_b3hw.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    normalized_b3hw: Float[Tensor, "b 3 h w"] = (image_b3hw / 255.0 - mean_1311) / std_1311
    return normalized_b3hw


class hourglass(nn.Module):
    """Aggregate a combined cost volume with a 3D encoder-decoder."""

    def __init__(self, cfg: ConfigLike, in_channels: int, feat_dims: list[int]) -> None:
        """Initialize the cost-volume hourglass.

        Args:
            cfg: Fast-FoundationStereo inference configuration.
            in_channels: Input and output volume channel count.
            feat_dims: Image-feature channels from quarter to thirty-second resolution.
        """
        super().__init__()
        self.cfg: ConfigLike = cfg
        self.conv1: nn.Sequential = nn.Sequential(
            BasicConv(
                in_channels,
                in_channels * 2,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=2,
                dilation=1,
            ),
            Conv3dNormActReduced(in_channels * 2, in_channels * 2, kernel_size=3, kernel_disp=17),
        )
        self.conv2: nn.Sequential = nn.Sequential(
            BasicConv(
                in_channels * 2,
                in_channels * 4,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=2,
                dilation=1,
            ),
            Conv3dNormActReduced(in_channels * 4, in_channels * 4, kernel_size=3, kernel_disp=17),
        )
        self.conv3: nn.Sequential = nn.Sequential(
            BasicConv(
                in_channels * 4,
                in_channels * 6,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=2,
                dilation=1,
            ),
            Conv3dNormActReduced(in_channels * 6, in_channels * 6, kernel_size=3, kernel_disp=17),
        )
        self.conv3_up: BasicConv = BasicConv(
            in_channels * 6,
            in_channels * 4,
            deconv=True,
            is_3d=True,
            bn=True,
            relu=True,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )
        self.conv2_up: BasicConv = BasicConv(
            in_channels * 4,
            in_channels * 2,
            deconv=True,
            is_3d=True,
            bn=True,
            relu=True,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )
        self.conv1_up: BasicConv = BasicConv(
            in_channels * 2,
            in_channels,
            deconv=True,
            is_3d=True,
            bn=True,
            relu=True,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )
        self.conv_out: nn.Sequential = nn.Sequential(
            Conv3dNormActReduced(in_channels, in_channels, kernel_size=3, kernel_disp=17),
            Conv3dNormActReduced(in_channels, in_channels, kernel_size=3, kernel_disp=17),
        )
        self.agg_0: nn.Sequential = nn.Sequential(
            BasicConv(in_channels * 8, in_channels * 4, is_3d=True, kernel_size=1, padding=0, stride=1),
            Conv3dNormActReduced(in_channels * 4, in_channels * 4, kernel_size=3, kernel_disp=17),
            Conv3dNormActReduced(in_channels * 4, in_channels * 4, kernel_size=3, kernel_disp=17),
        )
        self.agg_1: nn.Sequential = nn.Sequential(
            BasicConv(in_channels * 4, in_channels * 2, is_3d=True, kernel_size=1, padding=0, stride=1),
            Conv3dNormActReduced(in_channels * 2, in_channels * 2, kernel_size=3, kernel_disp=17),
            Conv3dNormActReduced(in_channels * 2, in_channels * 2, kernel_size=3, kernel_disp=17),
        )
        self.atts: nn.ModuleDict = nn.ModuleDict(
            {
                "4": CostVolumeDisparityAttention(
                    d_model=in_channels,
                    nhead=4,
                    dim_feedforward=in_channels,
                    norm_first=False,
                    num_transformer=4,
                    max_len=int(cfg["max_disp"]) // 16,
                )
            }
        )
        self.conv_patch: nn.Sequential = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=4, stride=4, padding=0, groups=in_channels),
            nn.BatchNorm3d(in_channels),
        )
        self.feature_att_8: FeatureAtt = FeatureAtt(in_channels * 2, feat_dims[1])
        self.feature_att_16: FeatureAtt = FeatureAtt(in_channels * 4, feat_dims[2])
        self.feature_att_32: FeatureAtt = FeatureAtt(in_channels * 6, feat_dims[3])
        self.feature_att_up_16: FeatureAtt = FeatureAtt(in_channels * 4, feat_dims[2])
        self.feature_att_up_8: FeatureAtt = FeatureAtt(in_channels * 2, feat_dims[1])
        self.post32_to_16: nn.Module | None = None
        self.post16_to_8: nn.Module | None = None
        self.post8_to_4: nn.Module | None = None

    def forward(
        self,
        x_bcdhw: Float[Tensor, "b channels disparities h4 w4"],
        features_left: FeaturePyramid,
    ) -> Float[Tensor, "b channels disparities h4 w4"]:
        """Aggregate one combined cost volume.

        Args:
            x_bcdhw: Combined volume with shape ``(batch, channels, disparities, height, width)``.
            features_left: Left-image feature pyramid.

        Returns:
            Aggregated volume with shape ``(batch, channels, disparities, height, width)``.
        """
        conv1_bcdhw: Float[Tensor, "b channels2 disparities2 h8 w8"] = self.feature_att_8(self.conv1(x_bcdhw), features_left[1])
        conv2_bcdhw: Float[Tensor, "b channels4 disparities4 h16 w16"] = self.feature_att_16(self.conv2(conv1_bcdhw), features_left[2])
        conv3_bcdhw: Float[Tensor, "b channels6 disparities8 h32 w32"] = self.feature_att_32(self.conv3(conv2_bcdhw), features_left[3])

        if self.post32_to_16 is None:
            conv3_up_bcdhw: Float[Tensor, "b channels4 disparities4 h16 w16"] = self.conv3_up(conv3_bcdhw)
            combined2_bcdhw: Float[Tensor, "b channels8 disparities4 h16 w16"] = torch.cat((conv3_up_bcdhw, conv2_bcdhw), dim=1)
            conv2_bcdhw = self.feature_att_up_16(self.agg_0(combined2_bcdhw), features_left[2])
        else:
            conv2_bcdhw = self.post32_to_16(conv2_bcdhw, conv3_bcdhw, features_left[2])

        if self.post16_to_8 is None:
            conv2_up_bcdhw: Float[Tensor, "b channels2 disparities2 h8 w8"] = self.conv2_up(conv2_bcdhw)
            combined1_bcdhw: Float[Tensor, "b channels4 disparities2 h8 w8"] = torch.cat((conv2_up_bcdhw, conv1_bcdhw), dim=1)
            conv1_bcdhw = self.feature_att_up_8(self.agg_1(combined1_bcdhw), features_left[1])
        else:
            conv1_bcdhw = self.post16_to_8(conv1_bcdhw, conv2_bcdhw, features_left[1])

        conv_bcdhw: Float[Tensor, "b channels disparities h4 w4"] = self.conv1_up(conv1_bcdhw)
        if self.post8_to_4 is None:
            patched_bcdhw: Float[Tensor, "b channels disparities4 h16 w16"] = self.conv_patch(x_bcdhw)
            attended_bcdhw: Float[Tensor, "b channels disparities4 h16 w16"] = self.atts["4"](patched_bcdhw)
            interpolated_bcdhw: Float[Tensor, "b channels disparities h4 w4"] = F.interpolate(
                attended_bcdhw,
                scale_factor=4,
                mode="trilinear",
                align_corners=False,
            )
            conv_bcdhw = self.conv_out(conv_bcdhw + interpolated_bcdhw)
        else:
            conv_bcdhw = self.post8_to_4(x_bcdhw, conv_bcdhw)
        return conv_bcdhw


class FastFoundationStereo(nn.Module):
    """Fast-FoundationStereo inference model used by config and released NAS checkpoints."""

    def __init__(self, args: ConfigLike) -> None:
        """Initialize the config-described architecture.

        The released checkpoint is a pickled NAS-pruned instance of this class and is not rebuilt by this constructor.

        Args:
            args: Fast-FoundationStereo inference configuration.
        """
        super().__init__()
        self.args: ConfigLike = args
        self.dtype: torch.dtype = torch.float32
        context_dims: list[int] = list(args["hidden_dims"])
        self.cv_group: int = int(args.get("cv_group", 8))
        self.concat_channel: int = 24
        self.volume_dim: int = int(args.get("volume_dim", 28))
        self.update_block: BasicSelectiveMultiUpdateBlock = BasicSelectiveMultiUpdateBlock(
            self.args,
            int(args["hidden_dims"][0]),
            volume_dim=self.volume_dim,
        )
        self.sam: SpatialAttentionExtractor = SpatialAttentionExtractor()
        self.cam: ChannelAttentionEnhancement = ChannelAttentionEnhancement(int(args["hidden_dims"][0]))
        self.context_zqr_convs: nn.ModuleList = nn.ModuleList(
            [
                nn.Conv2d(context_dims[index], int(args["hidden_dims"][index]) * 3, kernel_size=3, padding=1)
                for index in range(int(args["n_gru_layers"]))
            ]
        )
        self.feature: Feature = Feature(args)
        self.proj_cmb: nn.Conv2d = nn.Conv2d(self.feature.d_out[0], self.concat_channel // 2, kernel_size=1, padding=0)
        self.cnet: ContextNetSharedBackbone = ContextNetSharedBackbone(
            args,
            c04=self.feature.d_out[0],
            c08=self.feature.d_out[1],
            c16=self.feature.d_out[2],
            output_dim=[args["hidden_dims"], context_dims],
        )
        self.stem_2: nn.Sequential = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
        )
        self.spx_2_gru: Conv2x = Conv2x(32, 32, deconv=True, bn=False, concat=True)
        self.spx_gru: nn.Sequential = nn.Sequential(nn.ConvTranspose2d(64, 9, kernel_size=4, stride=2, padding=1))
        self.corr_stem: nn.Sequential = nn.Sequential(
            nn.Conv3d(self.proj_cmb.out_channels * 2 + self.cv_group, self.volume_dim, kernel_size=1),
            BasicConv(self.volume_dim, self.volume_dim, kernel_size=3, padding=1, is_3d=True),
            ResnetBasicBlock3D(self.volume_dim, self.volume_dim, kernel_size=3, stride=1, padding=1),
            ResnetBasicBlock3D(self.volume_dim, self.volume_dim, kernel_size=3, stride=1, padding=1),
        )
        self.corr_feature_att: FeatureAtt = FeatureAtt(self.volume_dim, self.feature.d_out[0])
        self.cost_agg: hourglass = hourglass(cfg=self.args, in_channels=self.volume_dim, feat_dims=self.feature.d_out)
        self.classifier: nn.Sequential = nn.Sequential(
            BasicConv(self.volume_dim, self.volume_dim // 2, kernel_size=3, padding=1, is_3d=True),
            ResnetBasicBlock3D(self.volume_dim // 2, self.volume_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(self.volume_dim // 2, 1, kernel_size=7, padding=3),
        )
        radius: int = int(args["corr_radius"])
        dx_11r1: Int8[Tensor, "1 1 radius_samples 1"] = torch.arange(
            -radius,
            radius + 1,
            requires_grad=False,
            dtype=torch.int8,
        ).reshape(1, 1, 2 * radius + 1, 1)
        self.dx: Int8[Tensor, "1 1 radius_samples 1"]
        self.register_buffer("dx", dx_11r1)

    def upsample_disp(
        self,
        disparity_b1hw: Float[Tensor, "b 1 h4 w4"],
        mask_features_bchw: Float[Tensor, "b mask_channels h4 w4"],
        stem_2x_bchw: Float[Tensor, "b stem_channels h2 w2"],
    ) -> Float32[Tensor, "b 1 h w"]:
        """Upsample quarter-resolution disparity with learned context weights.

        Args:
            disparity_b1hw: Disparity with shape ``(batch, 1, height / 4, width / 4)``.
            mask_features_bchw: Mask features with shape ``(batch, mask_channels, height / 4, width / 4)``.
            stem_2x_bchw: Stem features with shape ``(batch, stem_channels, height / 2, width / 2)``.

        Returns:
            Float32 disparity with shape ``(batch, 1, height, width)``.
        """
        with torch.amp.autocast("cuda", enabled=bool(self.args["mixed_precision"]), dtype=AMP_DTYPE):
            xspx_bchw: Float[Tensor, "b spx_channels h2 w2"] = self.spx_2_gru(mask_features_bchw, stem_2x_bchw)
            logits_b9hw: Float[Tensor, "b 9 h w"] = self.spx_gru(xspx_bchw)
            weights_b9hw: Float[Tensor, "b 9 h w"] = F.softmax(logits_b9hw, 1)
            disparity_up_b1hw: Float[Tensor, "b 1 h w"] = context_upsample(disparity_b1hw * 4.0, weights_b9hw).unsqueeze(1)
        return disparity_up_b1hw.to(self.dtype)

    def forward(
        self,
        image1_b3hw: Float32[Tensor, "b 3 h w"],
        image2_b3hw: Float32[Tensor, "b 3 h w"],
        iters: int = 12,
        test_mode: bool = False,
        init_disp: Float[Tensor, "b 1 h4 w4"] | None = None,
        optimize_build_volume: VolumeBuilder = "pytorch1",
    ) -> Float32[Tensor, "b 1 h w"]:
        """Estimate left-view disparity for a rectified stereo pair.

        Args:
            image1_b3hw: Float32 left RGB tensor with shape ``(batch, 3, height, width)`` and values in ``[0, 255]``.
            image2_b3hw: Float32 right RGB tensor with shape ``(batch, 3, height, width)`` and values in ``[0, 255]``.
            iters: Number of recurrent disparity updates.
            test_mode: Must be true because this owned fork retains inference outputs only.
            init_disp: Optional quarter-resolution initial disparity.
            optimize_build_volume: PyTorch or Triton groupwise-correlation implementation.

        Returns:
            Float32 disparity with shape ``(batch, 1, height, width)``.

        Raises:
            ValueError: If training output or fewer than one update is requested.
        """
        if not test_mode:
            raise ValueError("The owned Fast-FoundationStereo fork supports inference only; pass test_mode=True.")
        if iters < 1:
            raise ValueError("Fast-FoundationStereo inference requires at least one iteration.")
        batch_size: int = image1_b3hw.shape[0]
        normalized1_b3hw: Float[Tensor, "b 3 h w"] = normalize_image(image1_b3hw)
        normalized2_b3hw: Float[Tensor, "b 3 h w"] = normalize_image(image2_b3hw)
        with torch.amp.autocast("cuda", enabled=bool(self.args["mixed_precision"]), dtype=AMP_DTYPE):
            pair_features: FeaturePyramid = self.feature(torch.cat([normalized1_b3hw, normalized2_b3hw], dim=0))
            features_left: FeaturePyramid = [output[:batch_size] for output in pair_features]
            features_right: FeaturePyramid = [output[batch_size:] for output in pair_features]
            stem_2x_bchw: Float[Tensor, "b stem_channels h2 w2"] = self.stem_2(normalized1_b3hw)
            if optimize_build_volume == "pytorch1":
                gwc_volume_bgdhw: Float[Tensor, "b groups disparities h4 w4"] = build_gwc_volume_optimized_pytorch1(
                    features_left[0],
                    features_right[0],
                    int(self.args["max_disp"]) // 4,
                    self.cv_group,
                    normalize=bool(self.args["normalize"]),
                )
            elif optimize_build_volume == "triton":
                gwc_volume_bgdhw = build_gwc_volume_triton(
                    features_left[0],
                    features_right[0],
                    int(self.args["max_disp"]) // 4,
                    self.cv_group,
                    normalize=bool(self.args["normalize"]),
                )
            else:
                raise ValueError(f"Invalid optimize_build_volume: {optimize_build_volume}")
            left_projected_bchw: Float[Tensor, "b concat_channels h4 w4"] = self.proj_cmb(features_left[0])
            right_projected_bchw: Float[Tensor, "b concat_channels h4 w4"] = self.proj_cmb(features_right[0])
            concat_volume_bcdhw: Float[Tensor, "b double_concat_channels disparities h4 w4"] = build_concat_volume_optimized_pytorch1(
                left_projected_bchw,
                right_projected_bchw,
                maxdisp=int(self.args["max_disp"]) // 4,
            )
            combined_volume_bcdhw: Float[Tensor, "b volume_channels disparities h4 w4"] = torch.cat(
                [gwc_volume_bgdhw, concat_volume_bcdhw],
                dim=1,
            )
            combined_volume_bcdhw = self.corr_stem(combined_volume_bcdhw)
            combined_volume_bcdhw = self.corr_feature_att(combined_volume_bcdhw, features_left[0])
            combined_volume_bcdhw = self.cost_agg(combined_volume_bcdhw, features_left)
            logits_bdhw: Float[Tensor, "b disparities h4 w4"] = self.classifier(combined_volume_bcdhw).squeeze(1)
            probability_bdhw: Float[Tensor, "b disparities h4 w4"] = F.softmax(logits_bdhw, dim=1)
            initial_disparity_b1hw: Float[Tensor, "b 1 h4 w4"] = (
                disparity_regression(probability_bdhw, int(self.args["max_disp"]) // 4) if init_disp is None else init_disp
            )
            context_output: ContextNetOutput = self.cnet(features_left[0], features_left[1], features_left[2])
            context_list: list[list[Float[Tensor, "b hidden h4 w4"]]] = list(context_output)
            net_list: TensorPyramid = [torch.tanh(context[0]) for context in context_list]
            input_list: TensorPyramid = [torch.relu(context[1]) for context in context_list]
            input_list = [self.cam(context) * context for context in input_list]
            attention_list: TensorPyramid = [self.sam(context) for context in input_list]

        geo_volume: Combined_Geo_Encoding_Volume = Combined_Geo_Encoding_Volume(
            features_left[0].to(self.dtype),
            features_right[0].to(self.dtype),
            combined_volume_bcdhw.to(self.dtype),
            num_levels=int(self.args["corr_levels"]),
        )
        feature_height: int = features_left[0].shape[2]
        feature_width: int = features_left[0].shape[3]
        coordinates_111w: Float32[Tensor, "1 1 w 1"] = torch.arange(
            feature_width,
            dtype=torch.float32,
            device=initial_disparity_b1hw.device,
        ).reshape(1, 1, feature_width, 1)
        coordinates_bhw1: Float32[Tensor, "b h w 1"] = coordinates_111w.repeat(batch_size, feature_height, 1, 1)
        disparity_b1hw: Float32[Tensor, "b 1 h4 w4"] = initial_disparity_b1hw.to(self.dtype)
        mask_features_bchw: Float[Tensor, "b mask_channels h4 w4"] | None = None
        for _ in range(iters):
            disparity_b1hw = disparity_b1hw.detach()
            geo_features_bchw: Float32[Tensor, "b correlation_channels h4 w4"] = geo_volume(disparity_b1hw, coordinates_bhw1, self.dx)
            with torch.amp.autocast("cuda", enabled=bool(self.args["mixed_precision"]), dtype=AMP_DTYPE):
                update_output: UpdateBlockOutput = self.update_block(
                    net_list,
                    input_list,
                    geo_features_bchw.to(self.dtype),
                    disparity_b1hw,
                    attention_list,
                )
            net_list = update_output[0]
            mask_features_bchw = update_output[1]
            delta_disparity_b1hw: Float[Tensor, "b 1 h4 w4"] = update_output[2]
            disparity_b1hw = disparity_b1hw + delta_disparity_b1hw.to(self.dtype)
        if mask_features_bchw is None:
            raise RuntimeError("Fast-FoundationStereo did not produce mask features.")
        disparity_up_b1hw: Float32[Tensor, "b 1 h w"] = self.upsample_disp(
            disparity_b1hw,
            mask_features_bchw.to(self.dtype),
            stem_2x_bchw.to(self.dtype),
        )
        return disparity_up_b1hw

    def run_hierachical(
        self,
        image1_b3hw: Float32[Tensor, "b 3 h w"],
        image2_b3hw: Float32[Tensor, "b 3 h w"],
        iters: int = 12,
        test_mode: bool = False,
        small_ratio: float = 0.5,
    ) -> Float32[Tensor, "b 1 h w"]:
        """Run coarse-to-fine hierarchical stereo inference.

        Args:
            image1_b3hw: Float32 left RGB tensor with shape ``(batch, 3, height, width)``.
            image2_b3hw: Float32 right RGB tensor with shape ``(batch, 3, height, width)``.
            iters: Number of recurrent updates at each scale.
            test_mode: Must be true because this owned fork retains inference outputs only.
            small_ratio: Coarse input scale.

        Returns:
            Float32 disparity with shape ``(batch, 1, height, width)``.

        Raises:
            ValueError: If training output is requested or the scale is not positive.
        """
        if not test_mode:
            raise ValueError("The owned Fast-FoundationStereo fork supports inference only; pass test_mode=True.")
        if small_ratio <= 0.0:
            raise ValueError("small_ratio must be positive.")
        height: int = image1_b3hw.shape[2]
        width: int = image1_b3hw.shape[3]
        image1_small_b3hw: Float32[Tensor, "b 3 small_h small_w"] = F.interpolate(
            image1_b3hw,
            scale_factor=small_ratio,
            align_corners=False,
            mode="bilinear",
        )
        image2_small_b3hw: Float32[Tensor, "b 3 small_h small_w"] = F.interpolate(
            image2_b3hw,
            scale_factor=small_ratio,
            align_corners=False,
            mode="bilinear",
        )
        small_padder: InputPadder = InputPadder(image1_small_b3hw.shape[-2:], divis_by=32, force_square=False)
        padded_small: list[Float[Tensor, "b channels padded_h padded_w"]] = small_padder.pad(image1_small_b3hw, image2_small_b3hw)
        disparity_small_b1hw: Float32[Tensor, "b 1 padded_h padded_w"] = self.forward(
            padded_small[0],
            padded_small[1],
            test_mode=True,
            iters=iters,
        )
        disparity_small_b1hw = small_padder.unpad(disparity_small_b1hw)
        disparity_small_up_b1hw: Float32[Tensor, "b 1 h w"] = F.interpolate(
            disparity_small_b1hw,
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        ) * (1.0 / small_ratio)
        disparity_small_up_b1hw = disparity_small_up_b1hw.clip(0, None)

        full_padder: InputPadder = InputPadder(image1_b3hw.shape[-2:], divis_by=32, force_square=False)
        padded_full: list[Float[Tensor, "b channels padded_h padded_w"]] = full_padder.pad(
            image1_b3hw,
            image2_b3hw,
            disparity_small_up_b1hw,
        )
        padded_disparity_b1hw: Float32[Tensor, "b 1 padded_h padded_w"] = padded_full[2]
        padded_disparity_b1hw += full_padder._pad[0]
        initial_disparity_b1hw: Float32[Tensor, "b 1 h4 w4"] = F.interpolate(
            padded_disparity_b1hw,
            scale_factor=0.25,
            mode="bilinear",
            align_corners=True,
        ) * 0.25
        disparity_b1hw: Float32[Tensor, "b 1 padded_h padded_w"] = self.forward(
            padded_full[0],
            padded_full[1],
            iters=iters,
            test_mode=True,
            init_disp=initial_disparity_b1hw,
        )
        output_b1hw: Float32[Tensor, "b 1 h w"] = full_padder.unpad(disparity_b1hw)
        return output_b1hw


FoundationStereoLite: type[FastFoundationStereo] = FastFoundationStereo
