import torch
from torch import nn
from torch.nn import functional as F
import torch
from simvg.models import HEADS
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule
from torch import Tensor
import pycocotools.mask as maskUtils
import numpy as np
from mmcv.cnn.bricks.registry import NORM_LAYERS
from .projection import Projector
from .cgformer_head import CGDecoder


def seg_loss(input, target):
    weight = torch.FloatTensor([0.9, 1.1]).cuda()
    return nn.functional.cross_entropy(input, target.long(), weight=weight)


@NORM_LAYERS.register_module()
class LN2d(nn.Module):
    """A LayerNorm variant, popularized by Transformers, that performs
    pointwise mean and variance normalization over the channel dimension for
    inputs that have shape (batch_size, channels, height, width)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class PredictHead(nn.Module):
    def __init__(self, predict_head, fpn_output_channels, fpn="None"):
        super(PredictHead, self).__init__()
        self.name = predict_head
        if predict_head == "CRIS":
            self.prediction_head = Projector(word_dim=fpn_output_channels[-1], in_dim=fpn_output_channels[-2])
        elif predict_head == "CGFormer":
            self.prediction_head = CGDecoder(token_dim=fpn_output_channels[-2], input_dims=fpn_output_channels[::-1])
        elif predict_head == "Conv":
            if fpn == "None":
                self.prediction_head = nn.Conv2d(fpn_output_channels[-1], 2, 1)
            elif fpn == "SimpleFPN":
                self.prediction_head = nn.Conv2d(fpn_output_channels[-1] // 2, 2, 1)
            elif fpn== "TransposeConvUpSample":
                self.prediction_head = nn.Conv2d(fpn_output_channels[-1] // 4, 2, 1)
        else:
            raise TypeError()

    def forward(self, x, lan_feature=None, lan_mask=None):
        if self.name == "CRIS":
            lan_feature = torch.cat(list(map(lambda feat, mask: torch.max(feat[mask, :], dim=0, keepdim=True)[0], lan_feature, ~lan_mask)))
            x = self.prediction_head(x, lan_feature)
        elif self.name == "CGFormer":
            x = self.prediction_head(x, lan_feature.transpose(-1, -2), ~lan_mask.unsqueeze(-1))
        elif self.name == "Conv":
            x = self.prediction_head(x)
        return x


class Neck(nn.Module):
    def __init__(self, neck_name, input_channels, out_channels):
        super(Neck, self).__init__()
        self.name = neck_name
        if neck_name == "SimFPN":
            self.neck = SimpleFPN(backbone_channel=input_channels, out_channels=out_channels)
        elif neck_name == "TransposeConvUpSample":
            self.neck = TransposeConvUpSample(backbone_channel=input_channels)
        else:
            self.neck = nn.Identity()

    def forward(self, x):
        return self.neck(x)


@HEADS.register_module()
class UnetHead(nn.Module):
    def __init__(self, input_channels, patch_size, fpn_output_channels=[96, 192, 384, 768], predict_head="Conv", fpn="None"):
        super(UnetHead, self).__init__()
        self.unet_decoder = SimpleDecoding(fpn_output_channels[-1])
        self.fpn_name = fpn
        self.neck = Neck(neck_name=fpn, input_channels=input_channels, out_channels=fpn_output_channels)
        self.patch_size = patch_size
        self.predict_head = PredictHead(predict_head, fpn_output_channels, fpn=fpn)

    def forward_train(self, x, targets, cls_feat=None, lan_feature=None, lan_mask=None):
        target_mask = torch.from_numpy(np.concatenate([maskUtils.decode(target)[None] for target in targets])).cuda()
        input_shape = target_mask.shape[-2:]

        if self.fpn_name == "None":
            x = self.predict_head(x)
        elif self.fpn_name == "TransposeConvUpSample":
            x = self.neck(x)
            x = self.predict_head(x)
        else:
            x_c1, x_c2, x_c3, x_c4 = self.neck(x)
            if self.predict_head.name == "CRIS":
                x = self.unet_decoder(x_c4, x_c3, x_c2, x_c1)
                x = self.predict_head(x, lan_feature, lan_mask)
            elif self.predict_head.name == "CGFormer":
                x, maps = self.predict_head([x_c4, x_c3, x_c2, x_c1], lan_feature, lan_mask)
            else:
                raise TypeError()

        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=True)
        loss_mask = seg_loss(x, target_mask)
        loss_dict = {"loss_mask": loss_mask}
        return loss_dict, x

    def forward_test(self, x, cls_feat=None, lan_feature=None, lan_mask=None):
        input_shape = [x.shape[-2] * self.patch_size, x.shape[-1] * self.patch_size]
        
        if self.fpn_name == "None":
            x = self.predict_head(x)
        elif self.fpn_name == "TransposeConvUpSample":
            x = self.neck(x)
            x = self.predict_head(x)
        else:
            x_c1, x_c2, x_c3, x_c4 = self.neck(x)
            if self.predict_head.name == "CRIS":
                x = self.unet_decoder(x_c4, x_c3, x_c2, x_c1)
                x = self.predict_head(x, lan_feature, lan_mask)
            elif self.predict_head.name == "CGFormer":
                x, maps = self.predict_head([x_c4, x_c3, x_c2, x_c1], lan_feature, lan_mask)
            else:
                raise TypeError()

        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=True)
        return x


class SimpleFPN(BaseModule):
    """Simple Feature Pyramid Network for ViTDet."""

    def __init__(
        self,
        backbone_channel=768,
        in_channels=[192, 384, 768, 768],
        out_channels=[96, 192, 384, 768],
        num_outs=4,
        norm_cfg=dict(type="LN2d", requires_grad=True),
        conv_cfg=None,
        act_cfg=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.backbone_channel = backbone_channel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel, self.backbone_channel // 2, 2, 2),
            build_norm_layer(norm_cfg, self.backbone_channel // 2)[1],
            nn.GELU(),
            nn.ConvTranspose2d(self.backbone_channel // 2, self.backbone_channel // 4, 2, 2),
        )
        self.fpn2 = nn.Sequential(nn.ConvTranspose2d(self.backbone_channel, self.backbone_channel // 2, 2, 2))
        self.fpn3 = nn.Sequential(nn.Identity())
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.num_ins):
            l_conv = ConvModule(in_channels[i], out_channels[i], 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False)
            fpn_conv = ConvModule(out_channels[i], out_channels[i], 3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

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


class SimpleDecoding(nn.Module):
    def __init__(self, c4_dims, factor=2):
        super(SimpleDecoding, self).__init__()

        hidden_size = c4_dims // factor
        c4_size = c4_dims
        c3_size = c4_dims // (factor**1)
        c2_size = c4_dims // (factor**2)
        c1_size = c4_dims // (factor**3)

        self.conv1_4 = nn.Conv2d(c4_size + c3_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(hidden_size)
        self.relu1_4 = nn.ReLU()
        self.conv2_4 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_4 = nn.BatchNorm2d(hidden_size)
        self.relu2_4 = nn.ReLU()

        self.conv1_3 = nn.Conv2d(hidden_size + c2_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(hidden_size)
        self.relu1_3 = nn.ReLU()
        self.conv2_3 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_3 = nn.BatchNorm2d(hidden_size)
        self.relu2_3 = nn.ReLU()

        self.conv1_2 = nn.Conv2d(hidden_size + c1_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(hidden_size)
        self.relu1_2 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(hidden_size)
        self.relu2_2 = nn.ReLU()

    def forward(self, x_c4, x_c3, x_c2, x_c1):
        # fuse Y4 and Y3
        if x_c4.size(-2) < x_c3.size(-2) or x_c4.size(-1) < x_c3.size(-1):
            x_c4 = F.interpolate(input=x_c4, size=(x_c3.size(-2), x_c3.size(-1)), mode="bilinear", align_corners=True)
        x = torch.cat([x_c4, x_c3], dim=1)
        x = self.conv1_4(x)
        x = self.bn1_4(x)
        x = self.relu1_4(x)
        x = self.conv2_4(x)
        x = self.bn2_4(x)
        x = self.relu2_4(x)
        # fuse top-down features and Y2 features
        if x.size(-2) < x_c2.size(-2) or x.size(-1) < x_c2.size(-1):
            x = F.interpolate(input=x, size=(x_c2.size(-2), x_c2.size(-1)), mode="bilinear", align_corners=True)
        x = torch.cat([x, x_c2], dim=1)
        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu1_3(x)
        x = self.conv2_3(x)
        x = self.bn2_3(x)
        x = self.relu2_3(x)
        # fuse top-down features and Y1 features
        if x.size(-2) < x_c1.size(-2) or x.size(-1) < x_c1.size(-1):
            x = F.interpolate(input=x, size=(x_c1.size(-2), x_c1.size(-1)), mode="bilinear", align_corners=True)
        x = torch.cat([x, x_c1], dim=1)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)

        return x
