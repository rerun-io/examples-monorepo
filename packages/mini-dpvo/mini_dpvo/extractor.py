import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class ResidualBlock(nn.Module):
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
        y: Float[Tensor, "..."] = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class BottleneckBlock(nn.Module):
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
        y: Float[Tensor, "..."] = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

DIM: int = 32

class BasicEncoder(nn.Module):
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

        # output convolution
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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim: int, stride: int = 1) -> nn.Sequential:
        layer1: ResidualBlock = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2: ResidualBlock = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers: tuple[ResidualBlock, ResidualBlock] = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x: Float[Tensor, "b n c1 h1 w1"]) -> Float[Tensor, "b n output_dim h2 w2"]:
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

        # output convolution
        self.conv2: nn.Conv2d = nn.Conv2d(2*DIM, output_dim, kernel_size=1)

        if dropout > 0:
            self.dropout: nn.Dropout2d | None = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim: int, stride: int = 1) -> nn.Sequential:
        layer1: ResidualBlock = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2: ResidualBlock = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers: tuple[ResidualBlock, ResidualBlock] = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x: Float[Tensor, "b n c1 h1 w1"]) -> Float[Tensor, "b n output_dim h2 w2"]:
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
