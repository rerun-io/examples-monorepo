import torch
import torch.nn as nn
import torch.nn.functional as F


class Aggregation(nn.Module):
    def __init__(self, in_channels, left_att, blocks, expanse_ratio, backbone_channels):
        super(Aggregation, self).__init__()

        self.left_att = left_att
        self.expanse_ratio = expanse_ratio

        conv0 = [
            FasterNetResidual(in_channels, in_channels, stride=1, mlp_ratio=self.expanse_ratio)
            for _ in range(blocks[0])
        ]
        self.conv0 = nn.Sequential(*conv0)

        self.conv1 = FasterNetResidual(in_channels, in_channels * 2, stride=2, mlp_ratio=self.expanse_ratio)
        conv2_add = [
            FasterNetResidual(in_channels * 2, in_channels * 2, stride=1, mlp_ratio=self.expanse_ratio)
            for _ in range(blocks[1] - 1)
        ]
        self.conv2 = nn.Sequential(*conv2_add)

        self.conv3 = FasterNetResidual(in_channels * 2, in_channels * 4, stride=2, mlp_ratio=self.expanse_ratio)
        conv4_add = [
            FasterNetResidual(in_channels * 4, in_channels * 4, stride=1, mlp_ratio=self.expanse_ratio)
            for _ in range(blocks[2] - 1)
        ]
        self.conv4 = nn.Sequential(*conv4_add)

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels * 4,
                in_channels * 2,
                3,
                padding=1,
                output_padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels * 2),
        )

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels * 2,
                in_channels,
                3,
                padding=1,
                output_padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
        )

        self.redir1 = FasterNetResidual(in_channels, in_channels, stride=1, mlp_ratio=self.expanse_ratio)
        self.redir2 = FasterNetResidual(in_channels * 2, in_channels * 2, stride=1, mlp_ratio=self.expanse_ratio)

        if self.left_att:
            self.att0 = AttentionModule(in_channels, backbone_channels[0])
            self.att2 = AttentionModule(in_channels * 2, backbone_channels[1])
            self.att4 = AttentionModule(in_channels * 4, backbone_channels[2])

    def forward(self, x, features_left):
        x = self.conv0(x)
        if self.left_att:
            x = self.att0(x, features_left[0])

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        if self.left_att:
            conv2 = self.att2(conv2, features_left[1])

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        if self.left_att:
            conv4 = self.att4(conv4, features_left[2])

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return conv6


class FasterNetResidual(nn.Module):
    def __init__(self, inp, oup, stride, mlp_ratio, n_div=4, act_layer=nn.GELU):
        super(FasterNetResidual, self).__init__()
        assert stride in [1, 2]

        if stride == 1 and inp == oup:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
            )

        self.block = FasterNetBlock(oup, mlp_ratio=mlp_ratio, n_div=n_div, act_layer=act_layer)

    def forward(self, x):
        x = self.proj(x)
        return self.block(x)


class FasterNetBlock(nn.Module):
    def __init__(self, dim, mlp_ratio, n_div=4, act_layer=nn.GELU, layer_scale_init_value=0.0):
        super(FasterNetBlock, self).__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.spatial_mixing = PartialConv3(dim, n_div)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            act_layer(),
            nn.Conv2d(hidden_dim, dim, 1, bias=False),
        )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        else:
            self.layer_scale = None

    def forward(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = self.mlp(x)
        if self.layer_scale is not None:
            x = self.layer_scale.view(1, -1, 1, 1) * x
        return shortcut + x


class PartialConv3(nn.Module):
    def __init__(self, dim, n_div=4):
        super(PartialConv3, self).__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        return torch.cat((x1, x2), dim=1)


class AttentionModule(nn.Module):
    def __init__(self, dim, img_feat_dim):
        super().__init__()
        self.conv0 = nn.Conv2d(img_feat_dim, dim, 1)

        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, cost, x):
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        return attn * cost
