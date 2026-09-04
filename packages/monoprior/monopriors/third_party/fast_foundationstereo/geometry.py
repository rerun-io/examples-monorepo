"""Geometric and all-pairs correlation pyramids for Fast-FoundationStereo."""

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float, Int8
from torch import Tensor

from monopriors.third_party.fast_foundationstereo.utils import bilinear_sampler


class Combined_Geo_Encoding_Volume:
    """Build and sample geometric and all-pairs correlation pyramids."""

    def __init__(
        self,
        init_fmap1_bchw: Float[Tensor, "b channels h w"],
        init_fmap2_bchw: Float[Tensor, "b channels h w2"],
        geo_volume_bgdhw: Float[Tensor, "b groups disparities h w"],
        num_levels: int = 2,
    ) -> None:
        """Initialize the correlation pyramids.

        Args:
            init_fmap1_bchw: Floating-point left features with shape ``(batch, channels, height, width)``.
            init_fmap2_bchw: Floating-point right features with shape ``(batch, channels, height, right_width)``.
            geo_volume_bgdhw: Floating-point geometric volume with shape ``(batch, groups, disparities, height, width)``.
            num_levels: Number of horizontal average-pooling levels.
        """
        self.num_levels: int = num_levels
        self.geo_volume_pyramid: list[Float[Tensor, "samples groups 1 _disparities"]] = []
        self.init_corr_pyramid: list[Float[Tensor, "samples 1 1 _right_width"]] = []

        init_corr_bhw1v: Float[Tensor, "b h w 1 w2"] = self.corr(init_fmap1_bchw, init_fmap2_bchw)
        geo_volume_ng1d: Float[Tensor, "samples groups 1 disparities"] = rearrange(
            geo_volume_bgdhw,
            "b groups disparities h w -> (b h w) groups 1 disparities",
        )
        init_corr_n11v: Float[Tensor, "samples 1 1 right_width"] = rearrange(
            init_corr_bhw1v,
            "b h w one right_width -> (b h w) one 1 right_width",
        )
        self.geo_volume_pyramid.append(geo_volume_ng1d)
        self.init_corr_pyramid.append(init_corr_n11v)
        for _ in range(self.num_levels - 1):
            geo_volume_ng1d = F.avg_pool2d(geo_volume_ng1d, [1, 2], stride=[1, 2])
            self.geo_volume_pyramid.append(geo_volume_ng1d)
        for _ in range(self.num_levels - 1):
            init_corr_n11v = F.avg_pool2d(init_corr_n11v, [1, 2], stride=[1, 2])
            self.init_corr_pyramid.append(init_corr_n11v)

    def __call__(
        self,
        disparity_b1hw: Float[Tensor, "b 1 h w"],
        coordinates_bhw1: Float[Tensor, "b h w 1"],
        dx_11r1: Int8[Tensor, "1 1 radius_samples 1"],
    ) -> Float[Tensor, "b correlation_features h w"]:
        """Sample both pyramids around a disparity estimate.

        Args:
            disparity_b1hw: Floating-point disparity with shape ``(batch, 1, height, width)``.
            coordinates_bhw1: Floating-point x coordinates with shape ``(batch, height, width, 1)``.
            dx_11r1: Floating-point or integer local offsets with shape ``(1, 1, radius_samples, 1)``.

        Returns:
            Floating-point encoded correlation with shape ``(batch, correlation_features, height, width)``.
        """
        batch_size: int = disparity_b1hw.shape[0]
        height: int = disparity_b1hw.shape[2]
        width: int = disparity_b1hw.shape[3]
        outputs_bhwf: list[Float[Tensor, "b h w _features"]] = []
        for level in range(self.num_levels):
            scale: int = 2**level
            disparity_n111: Float[Tensor, "samples 1 1 1"] = disparity_b1hw.view(batch_size * height * width, 1, 1, 1) / scale
            geo_x_n1r1: Float[Tensor, "samples 1 radius_samples 1"] = dx_11r1 + disparity_n111
            y_n1r1: Float[Tensor, "samples 1 radius_samples 1"] = torch.zeros_like(geo_x_n1r1)
            geo_grid_n1r2: Float[Tensor, "samples 1 radius_samples 2"] = torch.cat([geo_x_n1r1, y_n1r1], dim=-1)
            geo_sample_ng1r: Float[Tensor, "samples groups 1 radius_samples"] = bilinear_sampler(
                self.geo_volume_pyramid[level],
                geo_grid_n1r2,
            )
            geo_bhwf: Float[Tensor, "b h w geo_features"] = geo_sample_ng1r.view(batch_size, height, width, -1)

            init_x_n1r1: Float[Tensor, "samples 1 radius_samples 1"] = (
                coordinates_bhw1.view(batch_size * height * width, 1, 1, 1) / scale - disparity_n111 + dx_11r1
            )
            corr_grid_n1r2: Float[Tensor, "samples 1 radius_samples 2"] = torch.cat([init_x_n1r1, y_n1r1], dim=-1)
            corr_sample_n11r: Float[Tensor, "samples 1 1 radius_samples"] = bilinear_sampler(
                self.init_corr_pyramid[level],
                corr_grid_n1r2,
            )
            corr_bhwf: Float[Tensor, "b h w corr_features"] = corr_sample_n11r.view(batch_size, height, width, -1)
            outputs_bhwf.extend((geo_bhwf, corr_bhwf))

        combined_bhwf: Float[Tensor, "b h w correlation_features"] = torch.cat(outputs_bhwf, dim=-1)
        combined_bfhw: Float[Tensor, "b correlation_features h w"] = rearrange(combined_bhwf, "b h w features -> b features h w")
        return combined_bfhw

    @staticmethod
    def corr(
        fmap1_bchw: Float[Tensor, "b channels h w"],
        fmap2_bchw: Float[Tensor, "b channels h w2"],
        normalize: bool = True,
    ) -> Float[Tensor, "b h w 1 w2"]:
        """Compute normalized all-pairs horizontal correlation.

        Args:
            fmap1_bchw: Floating-point left features with shape ``(batch, channels, height, width)``.
            fmap2_bchw: Floating-point right features with shape ``(batch, channels, height, right_width)``.
            normalize: Must remain true for the released inference path.

        Returns:
            Floating-point correlation with shape ``(batch, height, width, 1, right_width)``.

        Raises:
            ValueError: If the removed unnormalized path is requested.
        """
        if not normalize:
            raise ValueError("Fast-FoundationStereo inference requires normalized all-pairs correlation.")
        with torch.amp.autocast("cuda", enabled=False):
            correlation_bhwv: Float[Tensor, "b h w w2"] = torch.einsum(
                "bchw,bchv->bhwv",
                F.normalize(fmap1_bchw.float(), dim=1),
                F.normalize(fmap2_bchw.float(), dim=1),
            )
        correlation_bhw1v: Float[Tensor, "b h w 1 w2"] = rearrange(correlation_bhwv, "b h w w2 -> b h w 1 w2").to(fmap1_bchw.dtype)
        return correlation_bhw1v
