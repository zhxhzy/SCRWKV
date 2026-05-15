'''
Author: Zhang Hanxu
Github: github.com/zhxhzy/SCRWKV
'''

from typing import Sequence
import copy
import numpy as np
import torch
import torch.nn as nn
from sympy.codegen import Print
from timm.models.layers import DropPath, trunc_normal_
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
from mmcv.cnn import build_norm_layer
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner.base_module import ModuleList
from mmcls.models.builder import BACKBONES
from mmcls.models.utils import resize_pos_embed, to_2tuple
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.SFE_dev.models.modules.patch_embed import ConvPatchEmbed

# from models.GBC import BottConv


from models.AMCM import AMCM


from models.SCIU import Block

class BottConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BottConv, self).__init__()
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False)
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x



@BACKBONES.register_module()
class SFE(BaseBackbone):
    arch_zoo = {
        'Crack': {
            'patch_size': 4,
            'embed_dims': 256,
            'num_layers': 4,
            'num_convs_patch_embed': 2,
            'layers_with_dwconv': [],
        }
    }

    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 arch=None,
                 patch_size=16,
                 embed_dims=192,
                 num_layers=20,
                 num_convs_patch_embed=1,
                 with_pos_embed=True,
                 out_indices=[0, 1, 2, 3,4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 interpolate_mode='bicubic',
                 layer_cfgs=dict(),
                 layers_with_dwconv=[],
                 init_cfg=None,
                 test_cfg=dict(),
                 convert_syncbn=False,
                 freeze_patch_embed=False,
                 **kwargs):
        super(SFE, self).__init__(init_cfg)

        self.test_cfg = test_cfg
        self.img_size = to_2tuple(img_size)
        self.convert_syncbn = convert_syncbn
        self.arch = arch

        if self.arch is None:
            self.embed_dims = embed_dims
            self.num_layers = num_layers
            self.patch_size = patch_size
            self.num_convs_patch_embed = num_convs_patch_embed
            self.layers_with_dwconv = layers_with_dwconv
        else:
            assert self.arch in self.arch_zoo.keys()
            self.embed_dims = self.arch_zoo[self.arch]['embed_dims']
            self.num_layers = self.arch_zoo[self.arch]['num_layers']
            self.patch_size = self.arch_zoo[self.arch]['patch_size']
            self.num_convs_patch_embed = self.arch_zoo[self.arch]['num_convs_patch_embed']
            self.layers_with_dwconv = self.arch_zoo[self.arch]['layers_with_dwconv']



        self.with_pos_embed = with_pos_embed
        self.interpolate_mode = interpolate_mode
        self.freeze_patch_embed = freeze_patch_embed
        _drop_path_rate = drop_path_rate

        self.patch_embed = ConvPatchEmbed(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            num_convs=self.num_convs_patch_embed,
            patch_size=self.patch_size,
            stride=self.patch_size,
            dilation=1###21
        )
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        if with_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dims))
            trunc_normal_(self.pos_embed, std=0.02)
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        dpr = np.linspace(0, _drop_path_rate, self.num_layers)
        self.drop_path_rate = _drop_path_rate



        self.layers = ModuleList()
        for i in range(self.num_layers):
            if i in [0,1,2,3, self.num_layers - 1]:
                self.layers.append(
                    Block(
                        n_embd=self.embed_dims,
                        n_layer=self.num_layers,
                        layer_id=i,
                        channel_gamma=1/4,
                        shift_pixel=1,

                        shift_mode1='GBST',
                        hidden_rate=4,
                        drop_path=dpr[i],
                        init_mode='fancy',
                        init_values=None,
                        post_norm=False,
                        key_norm=False,
                        with_cp=False
                    ))

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

        for i in out_indices:
            if i != self.num_layers - 1:
                if norm_cfg is not None:
                    norm_layer = build_norm_layer(norm_cfg, self.embed_dims)[1]
                else:
                    norm_layer = nn.Identity()
                self.add_module(f'norm_layer{i}', norm_layer)
                     
        self.conv256to128 = BottConv(in_channels=256, out_channels=128, mid_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv256to64 = BottConv(in_channels=256, out_channels=64, mid_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv256to32 = BottConv(in_channels=256, out_channels=32, mid_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv256to16 = BottConv(in_channels=256, out_channels=16, mid_channels=8, kernel_size=1, stride=1, padding=0)


        self.gn128 = nn.GroupNorm(num_channels=128, num_groups=8)
        self.gn64 = nn.GroupNorm(num_channels=64, num_groups=4)
        self.gn32 = nn.GroupNorm(num_channels=32, num_groups=2)
        self.gn16 = nn.GroupNorm(num_channels=16, num_groups=2)
        self.AMCM=AMCM(256)





    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self):
        super(SFE, self).init_weights()
        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            if self.with_pos_embed:
                trunc_normal_(self.pos_embed, std=0.02)
        self.set_freeze_patch_embed()

    def set_freeze_patch_embed(self):
        if self.freeze_patch_embed:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

    def forward(self, x):
        x, patch_resolution = self.patch_embed(x)
        if self.with_pos_embed:
            pos_embed = resize_pos_embed(
                self.pos_embed,
                self.patch_resolution,
                patch_resolution,
                mode=self.interpolate_mode,
                num_extra_tokens=0
            )
            x = x + pos_embed


        x = self.drop_after_pos(x)
        B, L, C = x.shape
        H, W = patch_resolution
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        x = self.AMCM(x)

        x = x.reshape(B, C, H * W).permute(0, 2, 1)  #  [B,L,C]



        outs_before = []
        outs = []
        for i, layer in enumerate(self.layers):
            if i in [0,1,2,3, self.num_layers - 1] and isinstance(layer, Block):
                x = layer(x, patch_resolution)  # VRWKV Block
            if i in self.out_indices:
                B, _, C = x.shape
                patch_token = x.reshape(B, *patch_resolution, C)
                if i != self.num_layers - 1:
                    norm_layer = getattr(self, f'norm_layer{i}')
                    patch_token = norm_layer(patch_token)
                patch_token = patch_token.permute(0, 3, 1, 2)
                outs_before.append(patch_token)


                if i == self.out_indices[0]:

                    patch_token_mid=patch_token

                    patch_token_mid = self.gn128(self.conv256to128(patch_token_mid))

                    patch_token_mid = nn.Upsample(size=(64, 64), mode="bilinear")(patch_token_mid)

                    patch_token_mid = patch_token_mid
                    outs.append(patch_token_mid)
                elif i == self.out_indices[1]:

                    patch_token_mid = patch_token
                    patch_token_mid = self.gn64(self.conv256to64(patch_token_mid))

                    patch_token_mid = nn.Upsample(size=(128, 128), mode="bilinear")(patch_token_mid)

                    patch_token_mid = patch_token_mid
                    outs.append(patch_token_mid)
                elif i == self.out_indices[2]:
                    patch_token_mid = patch_token

                    patch_token_mid = self.gn32(self.conv256to32(patch_token_mid))

                    patch_token_mid = nn.Upsample(size=(256, 256), mode="bilinear")(patch_token_mid)

                    patch_token_mid = patch_token_mid
                    outs.append(patch_token_mid)
                elif i == self.out_indices[3]:
                    patch_token_mid = patch_token

                    patch_token_mid = self.gn16(self.conv256to16(patch_token_mid))

                    patch_token_mid = nn.Upsample(size=(512, 512), mode="bilinear")(patch_token_mid)

                    patch_token_mid = patch_token_mid
                    outs.append(patch_token_mid)
                else:
                    continue

        return outs
