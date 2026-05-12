'''
Author: Zhang Hanxu
Github: github.com/zhxhzy/SCRWKV
'''

from typing import Sequence
import warnings
import math
from typing import Optional

import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp
import collections.abc
from itertools import repeat
from timm.models.layers import DropPath
from torch.utils.cpp_extension import load

import os

from models.AMCM import AMCM


this_dir = os.path.dirname(os.path.abspath(__file__))
wkv_cuda = load(name="bi_wkv", sources=[
    os.path.join(this_dir, "cuda_new", "bi_wkv.cpp"),
    os.path.join(this_dir, "cuda_new", "bi_wkv_kernel.cu"),
],
                verbose=True,
                extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3',
                                   '-gencode arch=compute_86,code=sm_86'])


def resize_pos_embed(pos_embed,
                     src_shape,
                     dst_shape,
                     mode='bicubic',
                     num_extra_tokens=1):
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    _, L, C = pos_embed.shape
    src_h, src_w = src_shape
    assert L == src_h * src_w + num_extra_tokens, \
        f"The length of `pos_embed` ({L}) doesn't match the expected " \
        f'shape ({src_h}*{src_w}+{num_extra_tokens}). Please check the' \
        '`img_size` argument.'
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)

    dst_weight = F.interpolate(
        src_weight, size=dst_shape, align_corners=False, mode=mode)
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)

    return torch.cat((extra_tokens, dst_weight), dim=1)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class PatchEmbed(nn.Module):
    """
    Patch Embedding Layer (mmcv-style):
    Uses Conv2d with kernel_size = stride = patch_size to embed non-overlapping patches.
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_channels=3,
            embed_dims=768,
            norm_layer: Optional[str] = None  # 'LN', 'BN', or None
    ):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.in_channels = in_channels
        self.embed_dims = embed_dims

        self.new_grid_size = (img_size // self.patch_size[0], img_size // self.patch_size[1])

        self.proj = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dims,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # Patch resolution = output H, W after patching
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1]
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]


        if norm_layer == "LN":
            self.norm = nn.LayerNorm(embed_dims)
        elif norm_layer == "BN":
            self.norm = nn.BatchNorm1d(embed_dims)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        new_grid_size = (H // self.patch_size[0], W // self.patch_size[1])

        x = self.proj(x)  # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim], where N = H'*W'

        if self.norm is not None:
            if isinstance(self.norm, nn.LayerNorm):
                x = self.norm(x)
            elif isinstance(self.norm, nn.BatchNorm1d):
                # reshape to [B*N, C] for BN
                x = self.norm(x.transpose(1, 2)).transpose(1, 2)

        return x, new_grid_size


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v):

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)

        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = wkv_cuda.bi_wkv_forward(w, u, k, v)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        w, u, k, v = ctx.saved_tensors
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        gw, gu, gk, gv = wkv_cuda.bi_wkv_backward(w.float().contiguous(),
                                                  u.float().contiguous(),
                                                  k.float().contiguous(),
                                                  v.float().contiguous(),
                                                  gy.float().contiguous())
        if half_mode:
            return (gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            return (gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            return (gw, gu, gk, gv)


def RUN_CUDA(w, u, k, v):
    return WKV.apply(w.cuda(), u.cuda(), k.cuda(), v.cuda())


def GBST(input, shift_pixel=1, gamma=1 / 4, patch_resolution=None):

    #  (B, C, H, W)
    B, N, C = input.shape
    input_reshaped = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    B, C, H, W = input_reshaped.shape


    output = torch.zeros_like(input_reshaped)


    C_half = C // 2
    enlarge_total = C_half

    C_enlarge_quarter = C_half // 4
    enlarge_rem = enlarge_total - 4 * C_enlarge_quarter

    start, end = 0, C_enlarge_quarter
    output[:, start:end, :, shift_pixel:W] = input_reshaped[:, start:end, :, 0:W - shift_pixel]


    start, end = C_enlarge_quarter, C_enlarge_quarter * 2
    output[:, start:end, :, 0:W - shift_pixel] = input_reshaped[:, start:end, :, shift_pixel:W]


    start, end = C_enlarge_quarter * 2, C_enlarge_quarter * 3
    output[:, start:end, shift_pixel:H, :] = input_reshaped[:, start:end, 0:H - shift_pixel, :]

    start = C_enlarge_quarter * 3

    output[:, start:C_half, 0:H - shift_pixel, :] = input_reshaped[:, start:C_half, shift_pixel:H, :]

    if enlarge_rem > 0:
        start = C_enlarge_quarter * 4
        end = start + enlarge_rem
        output[:, start:end] = input_reshaped[:, start:end]


    shrink_channels_start_idx = C_half
    C_shrink = C - C_half
    C_shrink_quarter = C_shrink // 4
    shrink_rem = shrink_channels_start_idx - 4 * C_shrink_quarter

    start, end = shrink_channels_start_idx, shrink_channels_start_idx + C_shrink_quarter
    output[:, start:end, :, 0:W - shift_pixel] = input_reshaped[:, start:end, :, shift_pixel:W]


    start, end = shrink_channels_start_idx + C_shrink_quarter, shrink_channels_start_idx + C_shrink_quarter * 2
    output[:, start:end, :, shift_pixel:W] = input_reshaped[:, start:end, :, 0:W - shift_pixel]


    start, end = shrink_channels_start_idx + C_shrink_quarter * 2, shrink_channels_start_idx + C_shrink_quarter * 3
    output[:, start:end, 0:H - shift_pixel, :] = input_reshaped[:, start:end, shift_pixel:H, :]


    start = shrink_channels_start_idx + C_shrink_quarter * 3
    output[:, start:C, shift_pixel:H, :] = input_reshaped[:, start:C, 0:H - shift_pixel, :]
    if shrink_rem > 0:
        start = C_half + C_shrink_quarter * 4
        end = start + shrink_rem
        output[:, start:end] = input_reshaped[:, start:end]
    #  (B, N, C)
    output = output.flatten(2).transpose(1, 2)

    return output



class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id,
                 shift_mode1='q_shift1',
                 channel_gamma=1 / 4, shift_pixel=1, init_mode='fancy',
                 key_norm=False):
        super().__init__()
        self.use_dynamic_scan = True  # 默认关闭，按需开启
        self.dynamic_w_conv = nn.Conv2d(n_embd, n_embd, kernel_size=1, bias=False)

        self._scan_cache = None

        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel

        if shift_pixel > 0:

            self.shift_func1 = eval(shift_mode1)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(attn_sz)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)
        # self.outputsum = nn.Linear(n_embd * 2, n_embd, bias=False)
        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad():  # fancy init
                ratio_0_to_1 = (self.layer_id / (self.n_layer - 1))  # 0 to 1
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))  # 1 to ~0

                # fancy time_decay
                decay_speed = torch.ones(self.n_embd)
                for h in range(self.n_embd):
                    decay_speed[h] = -5 + 8 * (h / (self.n_embd - 1)) ** (
                            0.7 + 1.3 * ratio_0_to_1)
                self.spatial_decay = nn.Parameter(decay_speed)  #


                zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(self.n_embd)]) * 0.5)
                self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)


                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd

                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        elif init_mode == 'local':
            self.spatial_decay = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == 'global':
            self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def jit_func(self, x, patch_resolution):

        # B, T, C = x.size()
        if self.shift_pixel > 0:

            xx1 = self.shift_func1(x, self.shift_pixel, self.channel_gamma, patch_resolution)

            xk1 = x * self.spatial_mix_k + xx1 * (1 - self.spatial_mix_k)
            xv1 = x * self.spatial_mix_v + xx1 * (1 - self.spatial_mix_v)
            xr1 = x * self.spatial_mix_r + xx1 * (1 - self.spatial_mix_r)
        else:
            xk1 = x
            xv1 = x
            xr1 = x

        k1 = self.key(xk1)
        v1 = self.value(xv1)

        r1 = self.receptance(xr1)
        sr1 = torch.sigmoid(r1)

        return sr1, k1, v1

    def forward(self, x, patch_resolution):
        B, T, C = x.size()

        H, W = patch_resolution
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)  # [B,C,H,W]

        dynamic_w = torch.sigmoid(self.dynamic_w_conv(x_2d))  # [B,C,H,W]
        dynamic_w = dynamic_w.reshape(B, C, H * W).transpose(1, 2)  # [B,T,C]

        self.device = x.device

        sr1, k1, v1 = self.jit_func(x, patch_resolution)



        effective_decay = (self.spatial_decay / T).unsqueeze(0).unsqueeze(1)  # [1,1,C]
        effective_decay = effective_decay * (1 - dynamic_w)  # [B,T,C]

        rwkv_1 = RUN_CUDA(effective_decay, self.spatial_first / T, k1, v1)

        rwkv_1 = sr1 * (rwkv_1 + x)

        rwkv_1 = self.output(rwkv_1)

        return rwkv_1


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode1='q_shift1',
                 channel_gamma=1 / 4, shift_pixel=1, hidden_rate=4, init_mode='fancy',
                 key_norm=False):
        super().__init__()

        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel

        if shift_pixel > 0:

            self.shift_func1 = eval(shift_mode1)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None


        self.dynamic_w = nn.Conv1d(
            in_channels=n_embd,
            out_channels=n_embd,
            kernel_size=1,
            groups=n_embd,
            bias=True
        )

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)

        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None

        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0




    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad():
                ratio = (1.0 - (self.layer_id / self.n_layer))
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd

                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio))

        elif init_mode == 'local':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))

        elif init_mode == 'global':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)

        else:
            raise NotImplementedError


    def forward(self, x, patch_resolution):


        dyn = torch.sigmoid(
            self.dynamic_w(x.transpose(1, 2)).transpose(1, 2)
        )  # shape [B,L,C],  ∈ [0,1]

        if self.shift_pixel > 0:
            xx1 = self.shift_func1(x, self.shift_pixel, self.channel_gamma, patch_resolution)

            xk1 = x * (self.spatial_mix_k * dyn) + xx1 * (1 - self.spatial_mix_k * dyn)
            xr1 = x * (self.spatial_mix_r * dyn) + xx1 * (1 - self.spatial_mix_r * dyn)


        k1 = self.key(xk1)
        k1 = torch.square(torch.relu(k1))

        if self.key_norm is not None:
            k1 = self.key_norm(k1)


        kv1 = self.value(k1)

        rkv1 = torch.sigmoid(self.receptance(xr1)) * (kv1 + x)

        return rkv1


class Block(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id,  shift_mode1='GBST',
                 channel_gamma=1 / 4, shift_pixel=1, drop_path=0., hidden_rate=4,
                 init_mode='fancy', init_values=None, post_norm=False,
                 key_norm=False, with_cp=False):
        super().__init__()
        # Internal channel widths derived from n_embd.
        # Defaults match the original behavior: n_embd=256 -> hidden_dim=64, out_dim=256.
        self.out_dim = n_embd
        self.hidden_dim = n_embd // 4

        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix(self.hidden_dim, n_layer, layer_id, shift_mode1,
                                    channel_gamma, shift_pixel, init_mode,
                                    key_norm=key_norm)

        self.ffn = VRWKV_ChannelMix(self.hidden_dim, n_layer, layer_id, shift_mode1,
                                    channel_gamma, shift_pixel, hidden_rate,
                                    init_mode, key_norm=key_norm)


        self.layer_scale = (init_values is not None)
        self.post_norm = post_norm
        if self.layer_scale:
            self.gamma1 = nn.Parameter(init_values * torch.ones((self.hidden_dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(init_values * torch.ones((self.hidden_dim)), requires_grad=True)
        self.with_cp = with_cp

        self.conv256to64 = nn.Sequential(
            nn.Conv2d(self.out_dim, self.hidden_dim, 3, 1, 1, groups=self.hidden_dim),
            nn.GroupNorm(num_groups=self.hidden_dim // 16, num_channels=self.hidden_dim),
            nn.GELU()
        )
        self.conv64to256 = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.out_dim, 1),
            nn.GroupNorm(num_groups=self.out_dim // 32, num_channels=self.out_dim),
            nn.GELU()

        )

        self.conv64to64 = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1, groups=self.hidden_dim),

        )


        self.GN_256 = nn.GroupNorm(num_channels=self.out_dim, num_groups=self.out_dim // 32)

        self.AMCM64 = AMCM(self.hidden_dim)
        #


        self.conlast256 = nn.Conv2d(self.out_dim, self.out_dim, 1)

    def forward(self, x, patch_resolution):
        def _inner_forward(x):
            mixed_x = x
            if self.layer_id == 0:
                x = self.ln0(x)

            if self.post_norm:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.gamma2 * self.ln2(self.ffn(x, patch_resolution)))
                else:
                    x = x + self.drop_path(self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.ln2(self.ffn(x, patch_resolution)))
            else:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.gamma2 * self.ffn(self.ln2(x), patch_resolution))
                else:
                    B, L, C = x.shape
                    H, W = patch_resolution
                    x = x.permute(0, 2, 1).reshape(B, C, H, W)  # [B,C,H,W]
                    x = self.conv256to64(x)

                    x = x.reshape(B, self.hidden_dim, H * W).permute(0, 2, 1)  #  [B,L,C]
                    x = x.permute(0, 2, 1).reshape(B, self.hidden_dim, H, W)  # [B,C,H,W]
                    x = self.AMCM64(x)
                    x = x.reshape(B, self.hidden_dim, H * W).permute(0, 2, 1)  #  [B,L,C

                    x = x + self.drop_path(self.ffn(self.ln2(x), patch_resolution))

                    x = x.permute(0, 2, 1).reshape(B, self.hidden_dim, H, W)  # [B,C,H,W]

                    x = self.conv64to64(x)
                    x = x.reshape(B, self.hidden_dim, H * W).permute(0, 2, 1)  #  [B,L,C]
                    x = x + self.drop_path(self.att(self.ln1(x), patch_resolution))

                    x = x.permute(0, 2, 1).reshape(B, self.hidden_dim, H, W)  # [B,C,H,W]

                    x = self.conv64to256(x)

                    x = x.reshape(B, self.out_dim, H * W).permute(0, 2,
                                                         1)  #  [B,L,C]

                    b, l, c = mixed_x.shape
                    h = w = int(math.sqrt(l))
                    mixed_x = mixed_x.permute(0, 2, 1).reshape(b, c, h, w)
                    x = x.permute(0, 2, 1).reshape(B, C, H, W)
                    x = self.conlast256(x) + mixed_x

                    x = self.GN_256(x).reshape(B, C, H * W).permute(0, 2, 1)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x