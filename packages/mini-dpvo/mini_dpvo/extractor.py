"""CNN backbone feature extractors for DPVO.

Provides lightweight ResNet-style encoders that produce dense feature maps
at reduced spatial resolution.  DPVO uses two instances of
:class:`BasicEncoder4` (stride-4):

1. **Feature network** (``fnet``): Produces 128-dim features for computing
   local correlation volumes.  Uses instance normalization.
2. **Context / injection network** (``inet``): Produces DIM-dim (384)
   features injected into the GRU hidden state.  Uses no normalization.

:class:`BasicEncoder` is a deeper stride-8 variant used in related work
(e.g. RAFT) but not by default in DPVO.

All encoders accept batched multi-frame input ``(b, n, 3, h, w)`` and
return ``(b, n, output_dim, h // stride, w // stride)``.
"""

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class ResidualBlock(nn.Module):
    """Standard two-layer residual block with configurable normalization.

    Architecture: ``conv3x3 -> norm -> relu -> conv3x3 -> norm -> relu``
    with a skip connection.  When ``stride > 1`` a learned 1x1 convolution
    downsamples the skip path.

    Attributes:
        conv1: First 3x3 convolution (may have stride > 1 for downsampling).
        conv2: Second 3x3 convolution (always stride 1).
        relu: Shared ReLU activation.
        norm1: Normalization after conv1.
        norm2: Normalization after conv2.
        downsample: 1x1 conv + norm for the skip path when stride > 1,
            or ``None`` when stride == 1.
    """

    def __init__(self, in_planes: int, planes: int, norm_fn: str = 'group', stride: int = 1) -> None:
        super().__init__()

        self.conv1: nn.Conv2d = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2: nn.Conv2d = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)

        num_groups: int = planes // 8

        if norm_fn == 'group':
            self.norm1: nn.Module = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2: nn.Module = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3: nn.Module = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample: nn.Sequential | None = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x: Float[Tensor, "batch channels height width"]) -> Float[Tensor, "batch planes h2 w2"]:
        """Apply residual block.

        Args:
            x: Input feature map.

        Returns:
            Output feature map.  Spatial dimensions are reduced by
            ``stride`` if ``stride > 1``.
        """
        y: Float[Tensor, "..."] = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class BottleneckBlock(nn.Module):
    """Three-layer bottleneck residual block (1x1 -> 3x3 -> 1x1).

    The bottleneck reduces the channel count to ``planes // 4`` in the
    middle layer, then expands back to ``planes``.  This is more
    parameter-efficient than :class:`ResidualBlock` for high channel counts
    but is not used by default in DPVO.

    Attributes:
        conv1: 1x1 reduce convolution.
        conv2: 3x3 spatial convolution (may have stride > 1).
        conv3: 1x1 expand convolution.
        relu: Shared ReLU activation.
        norm1: Normalization after conv1.
        norm2: Normalization after conv2.
        norm3: Normalization after conv3.
        downsample: 1x1 conv + norm for the skip path when stride > 1,
            or ``None``.
    """

    def __init__(self, in_planes: int, planes: int, norm_fn: str = 'group', stride: int = 1) -> None:
        super().__init__()

        self.conv1: nn.Conv2d = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        self.conv2: nn.Conv2d = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        self.conv3: nn.Conv2d = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)

        num_groups: int = planes // 8

        if norm_fn == 'group':
            self.norm1: nn.Module = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2: nn.Module = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3: nn.Module = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4: nn.Module = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample: nn.Sequential | None = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)

    def forward(self, x: Float[Tensor, "batch channels height width"]) -> Float[Tensor, "batch planes h2 w2"]:
        """Apply bottleneck residual block.

        Args:
            x: Input feature map.

        Returns:
            Output feature map.
        """
        y: Float[Tensor, "..."] = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


DIM: int = 32
"""Base channel width for the encoder architectures.  Layers are multiples
of this value (DIM, 2*DIM, 4*DIM, ...)."""


class BasicEncoder(nn.Module):
    """Stride-8 feature encoder (conv7x2 -> layer1 -> layer2/s2 -> layer3/s2 -> 1x1).

    Produces a feature map at 1/8 the input spatial resolution.  This is
    the deeper encoder used in RAFT; DPVO uses :class:`BasicEncoder4`
    instead for its stride-4 output.

    The architecture is::

        conv7x7/s2 -> norm -> relu
        -> 2x ResBlock(DIM,   s1)       # layer1: 1/2 resolution
        -> 2x ResBlock(2*DIM, s2)       # layer2: 1/4 resolution
        -> 2x ResBlock(4*DIM, s2)       # layer3: 1/8 resolution
        -> conv1x1 -> output_dim        # projection

    Weights are initialised with Kaiming normal (conv) and constant
    (norm layers).

    Attributes:
        norm_fn: Normalization type (``'group'``, ``'batch'``,
            ``'instance'``, or ``'none'``).
        multidim: If True, builds additional layers for a multi-scale
            FPN-like architecture (not used in standard DPVO).
    """

    def __init__(self, output_dim: int = 128, norm_fn: str = 'batch', dropout: float = 0.0, multidim: bool = False) -> None:
        super().__init__()
        self.norm_fn: str = norm_fn
        self.multidim: bool = multidim

        if self.norm_fn == 'group':
            self.norm1: nn.Module = nn.GroupNorm(num_groups=8, num_channels=DIM)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(DIM)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(DIM)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1: nn.Conv2d = nn.Conv2d(3, DIM, kernel_size=7, stride=2, padding=3)
        self.relu1: nn.ReLU = nn.ReLU(inplace=True)

        self.in_planes: int = DIM
        self.layer1: nn.Sequential = self._make_layer(DIM,  stride=1)
        self.layer2: nn.Sequential = self._make_layer(2*DIM, stride=2)
        self.layer3: nn.Sequential = self._make_layer(4*DIM, stride=2)

        # 1x1 projection to the desired output dimensionality
        self.conv2: nn.Conv2d = nn.Conv2d(4*DIM, output_dim, kernel_size=1)

        if self.multidim:
            self.layer4: nn.Sequential = self._make_layer(256, stride=2)
            self.layer5: nn.Sequential = self._make_layer(512, stride=2)

            self.in_planes = 256
            self.layer6: nn.Sequential = self._make_layer(256, stride=1)

            self.in_planes = 128
            self.layer7: nn.Sequential = self._make_layer(128, stride=1)

            self.up1: nn.Conv2d = nn.Conv2d(512, 256, 1)
            self.up2: nn.Conv2d = nn.Conv2d(256, 128, 1)
            self.conv3: nn.Conv2d = nn.Conv2d(128, output_dim, kernel_size=1)

        if dropout > 0:
            self.dropout: nn.Dropout2d | None = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        # Kaiming initialization for stable training
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim: int, stride: int = 1) -> nn.Sequential:
        """Build a layer of two sequential :class:`ResidualBlock` instances.

        Args:
            dim: Output channel count for both blocks.
            stride: Stride for the first block (the second is always stride-1).

        Returns:
            Sequential module containing both blocks.
        """
        layer1: ResidualBlock = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2: ResidualBlock = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers: tuple[ResidualBlock, ResidualBlock] = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x: Float[Tensor, "b n c1 h1 w1"]) -> Float[Tensor, "b n output_dim h2 w2"]:
        """Extract stride-8 features from a batch of image sequences.

        Args:
            x: Input images, shape ``(b, n, 3, h, w)``.

        Returns:
            Feature maps of shape ``(b, n, output_dim, h//8, w//8)``.
        """
        b: int
        n: int
        c1: int
        h1: int
        w1: int
        b, n, c1, h1, w1 = x.shape
        x = x.view(b*n, c1, h1, w1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        c2: int
        h2: int
        w2: int
        _, c2, h2, w2 = x.shape
        return x.view(b, n, c2, h2, w2)


class BasicEncoder4(nn.Module):
    """Stride-4 feature encoder used by DPVO for both feature and context extraction.

    A shallower variant of :class:`BasicEncoder` producing features at 1/4
    the input spatial resolution.  The architecture is::

        conv7x7/s2 -> norm -> relu
        -> 2x ResBlock(DIM,   s1)       # layer1: 1/2 resolution
        -> 2x ResBlock(2*DIM, s2)       # layer2: 1/4 resolution
        -> conv1x1 -> output_dim        # projection

    DPVO instantiates two copies:

    - ``fnet``: ``BasicEncoder4(output_dim=128, norm_fn='instance')`` --
      produces correlation features.
    - ``inet``: ``BasicEncoder4(output_dim=DIM=384, norm_fn='none')`` --
      produces context features injected into the GRU.

    Attributes:
        norm_fn: Normalization type.
        multidim: Reserved for FPN-like extensions (not used).
    """

    def __init__(self, output_dim: int = 128, norm_fn: str = 'batch', dropout: float = 0.0, multidim: bool = False) -> None:
        super().__init__()
        self.norm_fn: str = norm_fn
        self.multidim: bool = multidim

        if self.norm_fn == 'group':
            self.norm1: nn.Module = nn.GroupNorm(num_groups=8, num_channels=DIM)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(DIM)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(DIM)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1: nn.Conv2d = nn.Conv2d(3, DIM, kernel_size=7, stride=2, padding=3)
        self.relu1: nn.ReLU = nn.ReLU(inplace=True)

        self.in_planes: int = DIM
        self.layer1: nn.Sequential = self._make_layer(DIM,  stride=1)
        self.layer2: nn.Sequential = self._make_layer(2*DIM, stride=2)

        # 1x1 projection to the desired output dimensionality
        self.conv2: nn.Conv2d = nn.Conv2d(2*DIM, output_dim, kernel_size=1)

        if dropout > 0:
            self.dropout: nn.Dropout2d | None = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        # Kaiming initialization for stable training
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim: int, stride: int = 1) -> nn.Sequential:
        """Build a layer of two sequential :class:`ResidualBlock` instances.

        Args:
            dim: Output channel count.
            stride: Stride for the first block.

        Returns:
            Sequential module containing both blocks.
        """
        layer1: ResidualBlock = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2: ResidualBlock = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers: tuple[ResidualBlock, ResidualBlock] = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x: Float[Tensor, "b n c1 h1 w1"]) -> Float[Tensor, "b n output_dim h2 w2"]:
        """Extract stride-4 features from a batch of image sequences.

        Args:
            x: Input images, shape ``(b, n, 3, h, w)``.

        Returns:
            Feature maps of shape ``(b, n, output_dim, h//4, w//4)``.
        """
        b: int
        n: int
        c1: int
        h1: int
        w1: int
        b, n, c1, h1, w1 = x.shape
        x = x.view(b*n, c1, h1, w1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.conv2(x)

        c2: int
        h2: int
        w2: int
        _, c2, h2, w2 = x.shape
        return x.view(b, n, c2, h2, w2)
