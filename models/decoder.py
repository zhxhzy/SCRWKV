'''
Author: Zhang Hanxu
Github: github.com/zhxhzy/SCRWKV
'''

import torch
from torch import nn
from mmcls.SFE_dev.models.SFE.SFE import SFE
from models.CSHF import CSHF

class Decoder(nn.Module):
    def __init__(self, backbone, args=None):
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.CSHF = CSHF(8)

    def forward(self, samples):
        outs_SFE = self.backbone(samples)
        out = self.CSHF(outs_SFE)

        return out

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()

        return 1 - dc

class bce_dice(nn.Module):
    def __init__(self, args):
        super(bce_dice, self).__init__()
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.dice_fn = DiceLoss()
        self.args = args

    def forward(self, y_pred, y_true):
        bce = self.bce_fn(y_pred, y_true)
        dice = self.dice_fn(y_pred.sigmoid(), y_true)
        return self.args.BCELoss_ratio * bce + self.args.DiceLoss_ratio * dice

def build(args):
    device = torch.device(args.device)
    args.device = torch.device(args.device)

    backbone = SFE(arch='Crack',
                     out_indices=(0, 1, 2, 3,4),
                     drop_path_rate=0.2,
                     final_norm=True,
                     convert_syncbn=True)
    model = Decoder(backbone, args)
    criterion = bce_dice(args)
    criterion.to(device)

    return model, criterion