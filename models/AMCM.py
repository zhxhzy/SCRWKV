'''
Author: Zhang Hanxu
Github: github.com/zhxhzy/SCRWKV
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))   # 缩放 γ
        self.beta = nn.Parameter(torch.zeros(normalized_shape))   # 平移 β
        self.eps = eps
        self.channel_format = data_format
        if self.channel_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.channel_format == "channels_last":  # [N, H, W, C]
            return F.layer_norm(x, self.normalized_shape, self.gamma, self.beta, self.eps)
        elif self.channel_format == "channels_first":  # [N, C, H, W]
            chan_mean = x.mean(1, keepdim=True)                          # μ_c
            chan_var = (x - chan_mean).pow(2).mean(1, keepdim=True)      # σ_c^2
            x_norm = (x - chan_mean) / torch.sqrt(chan_var + self.eps)   # 归一化
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
            nn.Linear(in_channels // reduction, 1)  # 输出维度为1
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

        # [B, C, H, W] -> [B, C, P, P] (P-grid_size)
        patches = self.patch_pool(x)

        # [B, C, P, P] -> [B, C, P*P] -> [B, P*P, C]
        patches_flat = patches.view(B, C, self.num_patches).permute(0, 2, 1)

        # [B, P*P, C] -> [B, P*P, 1] -> [B, P*P]
        weight = self.spatial_weight_fc(patches_flat).squeeze(-1)
        bias = self.spatial_bias_fc(patches_flat).squeeze(-1)

        # [B, 1, P*P]
        # [P*P, P*P]
        # [B, P*P, P*P]
        attn_matrix = F.softmax((weight.unsqueeze(1) * self.sp_attn) / self.temperature, dim=-1)

        # [B, P*P, P*P] @ [B, P*P, 1] -> [B, P*P, 1] -> [B, P*P]
        attn_out = torch.bmm(attn_matrix, (weight + bias).unsqueeze(-1)).squeeze(-1)

        # [B, P*P] -> [B, 1, P, P]
        spatial_attn_map = attn_out.view(B, 1, self.grid_size, self.grid_size)
        # [B, 1, P, P] -> [B, 1, H, W]
        spatial_attn_map = F.interpolate(spatial_attn_map, size=(H, W), mode='bilinear', align_corners=False)

        spatial_attn_map = self.sigmoid(spatial_attn_map)

        return x * spatial_attn_map


def _make_dw_conv_block(channels, kernel_size):
    """Depthwise conv -> GELU -> pointwise 1x1 conv. Padding keeps spatial size."""
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

        # Three-way channel split: two equal "main" parts and a (possibly smaller) "remainder".
        self.main_channels = math.ceil(dim / 3)               # was self.dim3
        self.remainder_channels = dim - 2 * self.main_channels  # was self.undim

        # Stage 2 operates on the two main parts concatenated together.
        self.stage2_in_channels = self.main_channels * 2

        # ---- Pre-mix (operates on the full dim, before the 3-way split) ----
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")           # was ln_stage1
        # Refines the first half of the channels with a depthwise 3x3 + pointwise 1x1.
        self.pre_dw_refine = _make_dw_conv_block(dim // 2, kernel_size=3)                # was con33
        # 1x1 used to produce a modulation signal from the residual input.
        self.pre_modulation = nn.Conv2d(dim, dim, 1)                                     # was con1

        # ---- Stage 1: gated 7x7 depthwise context on the 1st main part ----
        self.stage1_norm = self.pre_norm  # alias not used; kept logical
        self.stage1_context = nn.Sequential(                                             # was ctx_branch_s1
            nn.Conv2d(self.main_channels, self.main_channels, 1),
            nn.GELU(),
            nn.Conv2d(self.main_channels, self.main_channels, 7, padding=3,
                      groups=self.main_channels),
        )
        self.stage1_gate = nn.Conv2d(self.main_channels, self.main_channels, 1)          # was gate_branch_s1
        self.stage1_post = nn.Conv2d(self.main_channels, self.main_channels, 1)          # was post_gate_s1
        self.gate_main2 = nn.Conv2d(self.main_channels, self.main_channels, 1)           # was gate1
        self.gate_remainder = nn.Conv2d(self.remainder_channels, self.remainder_channels, 1)  # was gate2

        # ---- Stage 2: gated 9x9 depthwise context on the 2-main-parts concat ----
        self.stage2_norm = LayerNorm(self.stage2_in_channels, eps=1e-6,                   # was ln_stage2
                                     data_format="channels_first")
        self.stage2_context = nn.Sequential(                                              # was ctx_branch_s2
            nn.Conv2d(self.stage2_in_channels, self.stage2_in_channels, 1),
            nn.GELU(),
            nn.Conv2d(self.stage2_in_channels, self.stage2_in_channels, 9, padding=4,
                      groups=self.stage2_in_channels),
        )
        self.stage2_gate = nn.Conv2d(self.stage2_in_channels, self.stage2_in_channels, 1)  # was gate_branch_s2
        self.stage2_post = nn.Conv2d(self.stage2_in_channels, self.stage2_in_channels, 1)  # was post_gate_s2

        # ---- Stage 3: dilated depthwise 3x3 across the full dim (residual refinement) ----
        self.stage3_dilated_dw = nn.Sequential(                                           # was ctx_branch_s3
            nn.Conv2d(dim, dim, 3, padding=2, dilation=2, groups=dim),
        )

        # ---- Multi-scale depthwise branches (split full dim into 4 chunks) ----
        ms_channels = dim // 4
        # was: con55, con77, con99, con11 with kernel sizes 5,7,9,11
        self.multiscale_branches = nn.ModuleList([
            _make_dw_conv_block(ms_channels, k) for k in multiscale_kernels
        ])

        # ---- Spatial attention + final pointwise mixer ----
        self.spatial_attention = SpatialAttention(dim, sa_grid_size, sa_reduction)        # was SpatialAttention
        self.final_pointwise = nn.Conv2d(dim, dim, 1, groups=dim)                         # was conx1

    def forward(self, x):
        residual = x

        # ----- Pre-mix on the full feature map -----
        x = self.pre_norm(x)

        # Split full dim in half, refine the first half with a depthwise 3x3, then re-concat
        halves = torch.split(x, self.channels_total // 2, dim=1)
        refined_first_half = self.pre_dw_refine(halves[0])
        x = torch.cat([refined_first_half, halves[1]], dim=1)

        # Modulate with the residual through a 1x1
        x = x * self.pre_modulation(residual)

        # ----- 3-way split: [main_1, main_2, remainder] -----
        split_sizes = [self.main_channels, self.main_channels, self.remainder_channels]
        main_1, main_2, remainder = torch.split(x, split_sizes, dim=1)

        # ----- Stage 1: gate the 7x7 context with a learned 1x1 gate (on main_1) -----
        ctx_feat = self.stage1_context(main_1)
        gated_feat = ctx_feat * self.stage1_gate(main_1)
        gated_feat = self.stage1_post(gated_feat)

        # Keep main_2 (lightly gated) alongside the gated stage-1 output
        stage1_out = torch.cat((self.gate_main2(main_2), gated_feat), dim=1)

        # ----- Stage 2: gated 9x9 depthwise context on the 2-main-parts concat -----
        stage1_out = self.stage2_norm(stage1_out)
        ctx_feat = self.stage2_context(stage1_out)
        gated_feat = ctx_feat * self.stage2_gate(stage1_out)
        gated_feat = self.stage2_post(gated_feat)

        # Re-attach the (gated) remainder channels to recover full dim
        stage3_in = torch.cat((self.gate_remainder(remainder), gated_feat), dim=1)

        # ----- Stage 3: dilated DW residual refinement (full dim) -----
        stage3_out = self.stage3_dilated_dw(stage3_in) + stage3_in

        # ----- Multi-scale depthwise branches over the full dim (split into 4) -----
        chunks = torch.split(stage3_out, self.channels_total // 4, dim=1)
        ms_outs = [branch(chunk) for branch, chunk in zip(self.multiscale_branches, chunks)]
        x = torch.cat(ms_outs, dim=1)

        # ----- Spatial attention + final pointwise mixer + residual -----
        x = self.spatial_attention(x)
        out = self.final_pointwise(x)

        return out + residual