import torch
from torch import nn
from torch.nn import functional as F
import torch
from simvg.models import HEADS
from mmcv.cnn import ConvModule, build_norm_layer
from torch import Tensor
import pycocotools.mask as maskUtils
import numpy as np


def seg_loss(input, target):
    weight = torch.FloatTensor([0.9, 1.1]).cuda()
    return nn.functional.cross_entropy(input, target.long(), weight=weight)


@HEADS.register_module()
class DecoderHead(nn.Module):
    def __init__(self, input_channels, patch_size=16, input_size=256):
        super(DecoderHead, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.seg_conv = nn.Linear(input_channels, (self.input_size // self.patch_size) ** 2 * 2)

    def forward_train(self, x, targets, cls_feat=None, lan_feature=None, lan_mask=None):
        # transpose the cls_feat to the same h,w as x
        B, C, H, W = x.shape
        x = self.seg_conv(cls_feat)
        x = x.reshape(B, 2, H, W)
        # get target
        target_mask = torch.from_numpy(np.concatenate([maskUtils.decode(target)[None] for target in targets])).cuda()
        # calc loss
        x = F.interpolate(x, size=target_mask.shape[-2:], mode="bilinear", align_corners=True)
        loss_mask = seg_loss(x, target_mask)
        loss_dict = {"loss_mask": loss_mask}
        return loss_dict, x

    def forward_test(self, x, cls_feat=None, lan_feature=None, lan_mask=None):
        input_shape = [x.shape[-2] * self.patch_size, x.shape[-1] * self.patch_size]
        # transpose the cls_feat to the same h,w as x
        B, C, H, W = x.shape
        x = self.seg_conv(cls_feat)
        x = x.reshape(B, 2, H, W)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=True)
        return x

    def forward(self, input: Tensor) -> tuple:
        """Forward function.

        Args:
            inputs (Tensor): Features from the upstream network, 4D-tensor
        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        # build FPN
        inputs = []
        inputs.append(self.fpn1(input))
        inputs.append(self.fpn2(input))
        inputs.append(self.fpn3(input))
        inputs.append(self.fpn4(input))

        # build laterals
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_ins)]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            for i in range(self.num_outs - self.num_ins):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs)


class TransposeConvUpSample(nn.Module):
    def __init__(
        self,
        backbone_channel=768,
        norm_cfg=dict(type="LN2d", requires_grad=True),
    ):
        super(TransposeConvUpSample, self).__init__()

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(backbone_channel, backbone_channel // 2, 2, 2),
            build_norm_layer(norm_cfg, backbone_channel // 2)[1],
            nn.GELU(),
            nn.ConvTranspose2d(backbone_channel // 2, backbone_channel // 4, 2, 2),
        )

    def forward(self, x):
        return self.upsample(x)
