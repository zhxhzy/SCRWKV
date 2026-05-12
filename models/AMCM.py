import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.channel_format = data_format
        if self.channel_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.channel_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.gamma, self.beta, self.eps)
        elif self.channel_format == "channels_first":
            chan_mean = x.mean(1, keepdim=True)
            chan_var = (x - chan_mean).pow(2).mean(1, keepdim=True)
            x_norm = (x - chan_mean) / torch.sqrt(chan_var + self.eps)
            x_out = self.gamma[:, None, None] * x_norm + self.beta[:, None, None]
            return x_out


class SpatialAttention(nn.Module):

    def __init__(self, in_channels, grid_size=7, reduction=4):
        super().__init__()
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size

        self.patch_pool = nn.AdaptiveAvgPool2d(grid_size)

        self.spatial_weight_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, 1)
        )
        self.spatial_bias_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, 1)
        )

        self.sp_attn = nn.Parameter(torch.eye(self.num_patches))
        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        patches = self.patch_pool(x)

        patches_flat = patches.view(B, C, self.num_patches).permute(0, 2, 1)

        weight = self.spatial_weight_fc(patches_flat).squeeze(-1)
        bias = self.spatial_bias_fc(patches_flat).squeeze(-1)

        attn_matrix = F.softmax((weight.unsqueeze(1) * self.sp_attn) / self.temperature, dim=-1)

        attn_out = torch.bmm(attn_matrix, (weight + bias).unsqueeze(-1)).squeeze(-1)

        spatial_attn_map = attn_out.view(B, 1, self.grid_size, self.grid_size)
        spatial_attn_map = F.interpolate(spatial_attn_map, size=(H, W), mode='bilinear', align_corners=False)

        spatial_attn_map = self.sigmoid(spatial_attn_map)

        return x * spatial_attn_map


def _make_dw_conv_block(channels, kernel_size):
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels),
        nn.GELU(),
        nn.Conv2d(channels, channels, 1),
    )


class AMCM(nn.Module):
    def __init__(self, dim,
                 sa_grid_size=5, sa_reduction=2,
                 multiscale_kernels=(5, 7, 9, 11)):
        super().__init__()
        self.channels_total = dim

        self.main_channels = math.ceil(dim / 3)
        self.remainder_channels = dim - 2 * self.main_channels

        self.stage2_in_channels = self.main_channels * 2

        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.pre_dw_refine = _make_dw_conv_block(dim // 2, kernel_size=3)
        self.pre_modulation = nn.Conv2d(dim, dim, 1)

        self.stage1_norm = self.pre_norm
        self.stage1_context = nn.Sequential(
            nn.Conv2d(self.main_channels, self.main_channels, 1),
            nn.GELU(),
            nn.Conv2d(self.main_channels, self.main_channels, 7, padding=3,
                      groups=self.main_channels),
        )
        self.stage1_gate = nn.Conv2d(self.main_channels, self.main_channels, 1)
        self.stage1_post = nn.Conv2d(self.main_channels, self.main_channels, 1)
        self.gate_main2 = nn.Conv2d(self.main_channels, self.main_channels, 1)
        self.gate_remainder = nn.Conv2d(self.remainder_channels, self.remainder_channels, 1)

        self.stage2_norm = LayerNorm(self.stage2_in_channels, eps=1e-6,
                                     data_format="channels_first")
        self.stage2_context = nn.Sequential(
            nn.Conv2d(self.stage2_in_channels, self.stage2_in_channels, 1),
            nn.GELU(),
            nn.Conv2d(self.stage2_in_channels, self.stage2_in_channels, 9, padding=4,
                      groups=self.stage2_in_channels),
        )
        self.stage2_gate = nn.Conv2d(self.stage2_in_channels, self.stage2_in_channels, 1)
        self.stage2_post = nn.Conv2d(self.stage2_in_channels, self.stage2_in_channels, 1)

        self.stage3_dilated_dw = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=2, dilation=2, groups=dim),
        )

        ms_channels = dim // 4
        self.multiscale_branches = nn.ModuleList([
            _make_dw_conv_block(ms_channels, k) for k in multiscale_kernels
        ])

        self.spatial_attention = SpatialAttention(dim, sa_grid_size, sa_reduction)
        self.final_pointwise = nn.Conv2d(dim, dim, 1, groups=dim)

    def forward(self, x):
        residual = x

        x = self.pre_norm(x)

        halves = torch.split(x, self.channels_total // 2, dim=1)
        refined_first_half = self.pre_dw_refine(halves[0])
        x = torch.cat([refined_first_half, halves[1]], dim=1)

        x = x * self.pre_modulation(residual)

        split_sizes = [self.main_channels, self.main_channels, self.remainder_channels]
        main_1, main_2, remainder = torch.split(x, split_sizes, dim=1)

        ctx_feat = self.stage1_context(main_1)
        gated_feat = ctx_feat * self.stage1_gate(main_1)
        gated_feat = self.stage1_post(gated_feat)

        stage1_out = torch.cat((self.gate_main2(main_2), gated_feat), dim=1)

        stage1_out = self.stage2_norm(stage1_out)
        ctx_feat = self.stage2_context(stage1_out)
        gated_feat = ctx_feat * self.stage2_gate(stage1_out)
        gated_feat = self.stage2_post(gated_feat)

        stage3_in = torch.cat((self.gate_remainder(remainder), gated_feat), dim=1)

        stage3_out = self.stage3_dilated_dw(stage3_in) + stage3_in

        chunks = torch.split(stage3_out, self.channels_total // 4, dim=1)
        ms_outs = [branch(chunk) for branch, chunk in zip(self.multiscale_branches, chunks)]
        x = torch.cat(ms_outs, dim=1)

        x = self.spatial_attention(x)
        out = self.final_pointwise(x)

        return out + residual
