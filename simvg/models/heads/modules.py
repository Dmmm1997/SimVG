import torch
from torch import nn
from torch.nn import functional as F
import torch
from simvg.models import HEADS
import pycocotools.mask as maskUtils
import numpy as np
from simvg.models import MODELS, build_head
from .unet_head import *
from simvg.models.losses.boxloss import BoxLoss
from simvg.layers.mlp import MLP
from .tgqs_kd_detr_head.transformer import DetrTransformerDecoder, DetrTransformerEncoder
from simvg.layers.position_embedding import PositionEmbeddingSine, PositionEmbeddingSine1D
from torchvision.ops import RoIAlign
from ..utils import xywh_to_x1y1x2y2
from ..losses.contristiveloss import HardMiningTripletLoss
import einops
from torchvision.ops import roi_pool


class SegBranch(nn.Module):
    def __init__(self, input_channels, norm_cfg=dict(type="LN2d", requires_grad=True), upsample_rate=1):
        super(SegBranch, self).__init__()
        if upsample_rate == 4:
            self.neck = nn.Sequential(
                nn.ConvTranspose2d(input_channels, input_channels, 2, 2),
                build_norm_layer(norm_cfg, input_channels)[1],
                nn.GELU(),
                nn.ConvTranspose2d(input_channels, input_channels, 2, 2),
            )
            self.predict = nn.Conv2d(input_channels, 1, 1)
        elif upsample_rate == 1:
            self.neck = nn.Identity()
            self.predict = nn.Conv2d(input_channels, 1, 1)

    def forward(self, x):
        x = self.neck(x)
        x = self.predict(x)
        return x


class BoxBranch(nn.Module):
    def __init__(self, input_channels):
        super(BoxBranch, self).__init__()
        self.predict = MLP(input_channels, input_channels, 4, 3)

    def forward(self, x):
        x = self.predict(x).sigmoid()
        return x


class QueryAugment(nn.Module):
    def __init__(self, hidden_channels=256, num_queries=1):
        super().__init__()
        self.query2text_crossattn = DetrTransformerDecoder(
            embed_dim=hidden_channels,
            num_heads=8,
            attn_dropout=0.1,
            feedforward_dim=1024,
            ffn_dropout=0.1,
            num_layers=1,
            return_intermediate=False,
            post_norm=True,
        )
        self.query2img_crossattn = DetrTransformerDecoder(
            embed_dim=hidden_channels,
            num_heads=8,
            attn_dropout=0.1,
            feedforward_dim=1024,
            ffn_dropout=0.1,
            num_layers=3,
            return_intermediate=False,
            post_norm=True,
        )
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=hidden_channels // 2,
            temperature=10000,
            normalize=True,
        )
        self.position_embedding_1d = PositionEmbeddingSine1D(
            num_pos_feats=hidden_channels // 2,
            temperature=10000,
            normalize=True,
        )
        self.query_embed = nn.Embedding(num_queries, hidden_channels)

    def x_mask_pos_enc(self, x, img_shape):
        batch_size = x.size(0)
        input_img_h, input_img_w = img_shape
        x_mask = x.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_shape
            x_mask[img_id, :img_h, :img_w] = 0

        x_mask = F.interpolate(x_mask.unsqueeze(1), size=x.size()[-2:]).to(torch.bool).squeeze(1)
        x_pos_embeds = self.position_embedding(x_mask)
        return x_mask, x_pos_embeds

    def forward(self, box_feat, img_feat, lan_feat, lan_mask):
        # ! query 2 text cross attention
        query_embed_input = self.query_embed.weight.unsqueeze(0).repeat(lan_feat.shape[0], 1, 1).transpose(0, 1)
        query_embed_input = box_feat + query_embed_input
        text_pos_embed = self.position_embedding_1d(lan_feat).unsqueeze(0).repeat(lan_feat.shape[0], 1, 1).permute(1, 0, 2).cuda()
        text_feat_input = lan_feat.transpose(0, 1)
        query_embed = self.query2text_crossattn(
            query=torch.zeros_like(query_embed_input),
            key=text_feat_input,
            value=text_feat_input,
            key_pos=text_pos_embed,
            query_pos=query_embed_input,
            key_padding_mask=lan_mask.bool(),
        )[-1]
        # ! query 2 image cross attention
        img_masks, seg_pos_embed = self.x_mask_pos_enc(img_feat, img_feat.shape[-2:])
        seg_pos_embed = seg_pos_embed.view(seg_pos_embed.shape[0], seg_pos_embed.shape[1], -1).permute(2, 0, 1)
        img_feat_input = img_feat.view(img_feat.shape[0], img_feat.shape[1], -1).permute(2, 0, 1)
        query_embed = self.query2img_crossattn(
            query=torch.zeros_like(query_embed),
            key=img_feat_input,
            value=img_feat_input,
            key_pos=seg_pos_embed,
            query_pos=query_embed,
            # key_padding_mask=img_masks.bool(),
        )[-1]

        return query_embed[0]


class BoxSegPooler(nn.Module):
    def __init__(self, input_dim=768, sample_scale=1 / 4):
        super(BoxSegPooler, self).__init__()
        # self.box_align = RoIAlign(output_size=7, sampling_ratio=2, spatial_scale=sample_scale)
        self.box_pooler = nn.AdaptiveAvgPool2d((1, 1))
        self.seg_pooler = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, bbox, seg, img_feat, gt=False, img_pool=True):
        B, C, H, W = img_feat.shape
        # * box pooling
        bbox_x1y1x2y2 = xywh_to_x1y1x2y2(bbox)
        # ROI Align
        # bbox_feat = self.box_align(img_feat, [bbox_x1y1x2y2])
        # ROI Pooling
        batch_indices = torch.arange(B).unsqueeze(1).cuda()
        rois = torch.cat([batch_indices, bbox_x1y1x2y2], dim=1).float()
        bbox_feat = roi_pool(img_feat, rois, output_size=(1, 1))
        # avg pooling
        bbox_feat = self.box_pooler(bbox_feat).reshape(B, C)
        # * seg pooling
        # @TODO
        if gt:
            seg_mask = seg.squeeze(1)
        else:
            seg_mask = (seg.sigmoid() > 0.5).int().squeeze(1)
        seg_feat_pos, seg_feat_neg = self.masked_pool(img_feat, seg_mask)
        if img_pool:
            seg_feat_pos = self.seg_pooler(seg_feat_pos).reshape(B, C)
            seg_feat_neg = self.seg_pooler(seg_feat_neg).reshape(B, C)
        else:
            seg_feat_pos = seg_feat_pos.reshape(B, C, -1).transpose(1, 2)
            seg_feat_neg = seg_feat_neg.reshape(B, C, -1).transpose(1, 2)
        return bbox_feat, [seg_feat_pos, seg_feat_neg]

    def masked_pool(self, feature_map, mask):
        """
        Extract pixels from the feature map where the mask is True and compute the mean over the H and W dimensions.

        Args:
            feature_map (torch.Tensor): The feature map with dimensions (B, C, H, W).
            mask (torch.Tensor): The mask with dimensions (B, H, W).

        Returns:
            torch.Tensor: Mean values of the masked pixels for each channel with dimensions (B, C).
        """
        B, C, H, W = feature_map.shape
        assert mask.shape == (B, H, W), "Mask dimensions must be (B, H, W)"
        mask_expanded = mask.unsqueeze(1).expand(-1, C, -1, -1)  # Shape: (B, C, H, W)
        masked_feature_pos_map = feature_map * mask_expanded  # Shape: (B, C, H, W)
        masked_feature_neg_map = feature_map * (1 - mask_expanded)  # Shape: (B, C, H, W)
        pooled_mask_pos = masked_feature_pos_map
        pooled_mask_neg = masked_feature_neg_map

        return pooled_mask_pos, pooled_mask_neg


class BoxSegAttention(nn.Module):
    def __init__(self, input_channels=256, segaug=False, input_size=(224, 224)):
        super(BoxSegAttention, self).__init__()
        self.box2text_crossattn = DetrTransformerDecoder(
            embed_dim=input_channels,
            num_heads=8,
            attn_dropout=0.1,
            feedforward_dim=1024,
            ffn_dropout=0.1,
            num_layers=3,
            return_intermediate=False,
            post_norm=True,
        )
        self.seg2text_crossattn = DetrTransformerDecoder(
            embed_dim=input_channels,
            num_heads=8,
            attn_dropout=0.1,
            feedforward_dim=1024,
            ffn_dropout=0.1,
            num_layers=3,
            return_intermediate=False,
            post_norm=True,
        )

        self.box2img_crossattn = DetrTransformerDecoder(
            embed_dim=input_channels,
            num_heads=8,
            attn_dropout=0.1,
            feedforward_dim=1024,
            ffn_dropout=0.1,
            num_layers=3,
            return_intermediate=False,
            post_norm=True,
        )
        self.seg2img_crossattn = DetrTransformerDecoder(
            embed_dim=input_channels,
            num_heads=8,
            attn_dropout=0.1,
            feedforward_dim=1024,
            ffn_dropout=0.1,
            num_layers=3,
            return_intermediate=False,
            post_norm=True,
        )
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=input_channels // 2,
            temperature=10000,
            normalize=True,
        )
        self.position_embedding_1d = PositionEmbeddingSine1D(
            num_pos_feats=input_channels // 2,
            temperature=10000,
            normalize=True,
        )

        self.segboxSA = DetrTransformerEncoder(
            embed_dim=input_channels,
            num_heads=8,
            attn_dropout=0.1,
            feedforward_dim=1024,
            ffn_dropout=0.1,
            num_layers=3,
            post_norm=False,
        )

        self.segboxCA = DetrTransformerDecoder(
            embed_dim=input_channels,
            num_heads=8,
            attn_dropout=0.1,
            feedforward_dim=1024,
            ffn_dropout=0.1,
            num_layers=3,
            return_intermediate=False,
            post_norm=True,
        )

        # self.img_embedding = nn.Conv2d(input_channels, hidden_channels, 1)
        # self.seg_embedding = nn.Conv2d(input_channels, hidden_channels, 1)
        # self.seg_embedding = MLP(input_channels, hidden_channels, hidden_channels, 1)
        # self.lan_embedding = MLP(input_channels, hidden_channels, hidden_channels, 1)
        # self.box_embedding = MLP(input_channels, hidden_channels, hidden_channels, 1)
        self.boxsegpooler = BoxSegPooler()
        self.segaug = segaug
        self.box_query_embed = nn.Embedding(1, input_channels)
        H, W = input_size
        self.seg_query_embed = nn.Embedding(int(H * W / 16 / 16), input_channels)

    def x_mask_pos_enc(self, x, img_shape):
        batch_size = x.size(0)
        input_img_h, input_img_w = img_shape
        x_mask = x.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_shape
            x_mask[img_id, :img_h, :img_w] = 0

        x_mask = F.interpolate(x_mask.unsqueeze(1), size=x.size()[-2:]).to(torch.bool).squeeze(1)
        x_pos_embeds = self.position_embedding(x_mask)
        return x_mask, x_pos_embeds

    def forward(self, box_feat, seg_feat, img_feat, lan_feat=None, lan_mask=None):
        # ! type 6 boxalign box-seg(B,1+256,C) SA(box-seg) CA(box to img) CA(seg to img)
        B, _, H, W = img_feat.shape
        # # img_embedding
        img_embed = img_feat.view(img_feat.shape[0], img_feat.shape[1], -1).permute(2, 0, 1)

        img_masks, seg_pos_embed = self.x_mask_pos_enc(img_feat, img_feat.shape[-2:])
        seg_pos_embed = seg_pos_embed.view(seg_pos_embed.shape[0], seg_pos_embed.shape[1], -1).permute(2, 0, 1)

        # TODO query key padding mask add
        box_seg_query = torch.cat((box_feat.unsqueeze(1), seg_feat), dim=1).transpose(0, 1)
        # attn_mask_pos = seg.argmax(1).reshape(seg.shape[0], -1)
        # attn_mask_pos = attn_mask_pos.float().masked_fill(attn_mask_pos == 1, float('-inf')).masked_fill(attn_mask_pos == 0, float(0.0))
        box_seg_pos_embed = torch.cat((torch.zeros_like(box_feat.unsqueeze(0)), seg_pos_embed), dim=0)
        box_seg_query = self.segboxSA(query=box_seg_query, key=None, value=None, query_pos=box_seg_pos_embed, query_key_padding_mask=None)
        box_embed, seg_embed = box_seg_query.split((1, seg_feat.shape[1]), dim=0)
        box_embed = box_embed + self.box_query_embed.weight.unsqueeze(0).repeat(box_feat.shape[0], 1, 1).transpose(0, 1)
        seg_embed = seg_embed + self.seg_query_embed.weight.unsqueeze(0).repeat(seg_feat.shape[0], 1, 1).transpose(0, 1)
        box_hs = self.box2img_crossattn(
            query=torch.zeros_like(box_embed),
            key=img_embed,
            value=img_embed,
            key_pos=seg_pos_embed,
            query_pos=box_embed,
            # attn_masks=attn_mask_pos,
            # query_key_padding_mask=~seg.argmax(1).reshape(seg.shape[0], -1).bool(),
            # key_padding_mask=~seg.argmax(1).reshape(seg.shape[0], -1).bool(),  # ! add extra segmask
        )[-1]
        seg_hs = self.seg2img_crossattn(
            query=torch.zeros_like(seg_embed),
            key=img_embed,
            value=img_embed,
            key_pos=seg_pos_embed,
            query_pos=seg_embed,
            # query_key_padding_mask=None,
            # key_padding_mask=seg.argmax(1).reshape(seg.shape[0], -1).bool(),  # ! add extra segmask
        )[-1]

        # ! type 6 boxalign box-seg(B,1+256,C) SA(box-seg) CA(box to img) CA(seg to img)
        # B,_, H, W = img_feat.shape
        # # # img_embedding
        # img_feat = self.img_embedding(img_feat)
        # img_embed = img_feat.view(img_feat.shape[0], img_feat.shape[1], -1).permute(2, 0, 1)

        # # # box seg embedding
        # # box_feat, [seg_feat_pos, seg_feat_neg] = self.boxsegpooler(bbox, seg, img_feat)

        # img_masks, seg_pos_embed = self.x_mask_pos_enc(img_feat, img_feat.shape[-2:])
        # seg_pos_embed = seg_pos_embed.view(seg_pos_embed.shape[0], seg_pos_embed.shape[1], -1).permute(2, 0, 1)

        # # TODO query key padding mask add
        # box_seg_query = torch.cat((box_feat.unsqueeze(1), seg_feat), dim=1).transpose(0, 1)
        # # attn_mask_pos = seg.argmax(1).reshape(seg.shape[0], -1)
        # # attn_mask_pos = attn_mask_pos.float().masked_fill(attn_mask_pos == 1, float('-inf')).masked_fill(attn_mask_pos == 0, float(0.0))
        # box_seg_pos_embed = torch.cat((torch.zeros_like(box_feat.unsqueeze(0)), seg_pos_embed), dim=0)
        # box_seg_query = self.segboxSA(query=box_seg_query, key=None, value=None, query_pos=box_seg_pos_embed, query_key_padding_mask=None)
        # box_embed, seg_embed = box_seg_query.split((1, seg_feat.shape[1]), dim=0)
        # box_hs = self.box2img_crossattn(
        #     query=torch.zeros_like(box_embed),
        #     key=img_embed,
        #     value=img_embed,
        #     key_pos=seg_pos_embed,
        #     query_pos=box_embed,
        #     # attn_masks=attn_mask_pos,
        #     # query_key_padding_mask=~seg.argmax(1).reshape(seg.shape[0], -1).bool(),
        #     # key_padding_mask=~seg.argmax(1).reshape(seg.shape[0], -1).bool(),  # ! add extra segmask
        # )[-1]
        # seg_hs = self.seg2img_crossattn(
        #     query=torch.zeros_like(seg_embed),
        #     key=img_embed,
        #     value=img_embed,
        #     key_pos=seg_pos_embed,
        #     query_pos=seg_embed,
        #     # query_key_padding_mask=None,
        #     # key_padding_mask=seg.argmax(1).reshape(seg.shape[0], -1).bool(),  # ! add extra segmask
        # )[-1]

        second_seg_mask = seg_hs.permute(1, 2, 0).reshape(B, -1, H, W)
        second_bbox = box_hs.transpose(0, 1).reshape(B, -1)

        return second_bbox, second_seg_mask

    def forward_debug(self, bbox, seg, img_feat, lan_feat=None, lan_mask=None):
        # bbox_embed = bbox.view(bbox.shape[0], 1, -1)
        # seg_embed = seg.view(seg.shape[0], seg.shape[1], -1).transpose(1, 2)
        # bbox_embed = self.bbox_embedding(bbox_embed).transpose(0, 1)
        # seg_embed = self.seg_embedding(seg_embed).transpose(0, 1)
        # img_embed = self.img_embedding(img_feat.view(img_feat.shape[0], img_feat.shape[1], -1).transpose(1, 2)).transpose(0, 1)

        # ! type 1 SA（bbox-seg concat）
        # img_masks, seg_pos_embed = self.x_mask_pos_enc(seg, seg.shape[-2:])
        # seg_pos_embed = seg_pos_embed.view(seg_pos_embed.shape[0], seg_pos_embed.shape[1], -1).transpose(0, 1)
        # box_pos_embed = torch.zeros_like(bbox_embed)
        # concat_pos_embed = torch.cat((box_pos_embed, seg_pos_embed), dim=0)
        # concat_boxseg_embed = torch.cat((bbox_embed, seg_embed), dim=0)
        # hs = self.segboxSA(query=concat_boxseg_embed, key=None, value=None, query_pos=concat_pos_embed, query_key_padding_mask=None)
        # box_hs, seg_hs = hs[-1].split((bbox_embed.shape[0], seg_embed.shape[0]), dim=0)

        # ! type 2 SA（bbox-seg concat）+CA（bbox-seg to image）
        # img_masks, seg_pos_embed = self.x_mask_pos_enc(seg, seg.shape[-2:])
        # concat_boxseg_embed = torch.cat((bbox_embed, seg_embed), dim=0)
        # seg_pos_embed = seg_pos_embed.view(seg_pos_embed.shape[0], seg_pos_embed.shape[1], -1).transpose(0, 1)
        # box_pos_embed = torch.zeros_like(bbox_embed)
        # concat_pos_embed = torch.cat((box_pos_embed, seg_pos_embed), dim=0)
        # concat_boxseg_embed = torch.cat((bbox_embed, seg_embed), dim=0)
        # concat_boxseg_embed = self.segboxSA(query=concat_boxseg_embed, key=None, value=None, query_pos=concat_pos_embed, query_key_padding_mask=None)
        # hs = self.segboxCA(query=torch.zeros_like(concat_boxseg_embed), key=img_embed, value=img_embed, key_pos=seg_pos_embed,query_pos=concat_boxseg_embed, query_key_padding_mask=None)
        # box_hs, seg_hs = hs[-1].split((bbox_embed.shape[0], seg_embed.shape[0]), dim=0)

        # ! type 3 CA(bbox to seg) CA(seg to bbox-img)
        #  box-to-seg cross attention
        # img_masks, seg_pos_embed = self.x_mask_pos_enc(seg, seg.shape[-2:])
        # seg_pos_embed = seg_pos_embed.view(seg_pos_embed.shape[0], seg_pos_embed.shape[1], -1).transpose(0, 1)
        # box_hs = self.box2seg_crossattn(
        #     query=torch.zeros_like(bbox_embed),
        #     key=seg_embed,
        #     value=seg_embed,
        #     key_pos=seg_pos_embed,
        #     query_pos=bbox_embed,
        #     key_padding_mask=img_masks.view(img_masks.shape[0], -1).bool(),
        # )[-1]
        # # seg-to-box cross attention
        # # box_pos_embed = self.position_embedding_1d(bbox_embed.transpose(0, 1)).unsqueeze(0).repeat(bbox.shape[0], 1, 1).permute(1, 0, 2).cuda()
        # concat_boximg_embed = torch.cat((bbox_embed, img_embed), dim=0)
        # box_pos_embed = torch.zeros_like(bbox_embed)
        # concat_pos_embed = torch.cat((box_pos_embed, seg_pos_embed), dim=0)
        # seg_hs = self.seg2box_crossattn(
        #     query=torch.zeros_like(seg_embed),
        #     key=concat_boximg_embed,
        #     value=concat_boximg_embed,
        #     key_pos=concat_pos_embed,
        #     query_pos=seg_embed,
        #     key_padding_mask=None,
        # )[-1]

        # ! type 4 CA(bbox to seg) CA(img to bbox-seg)
        #  box-to-seg cross attention
        # img_masks, seg_pos_embed = self.x_mask_pos_enc(seg, seg.shape[-2:])
        # seg_pos_embed = seg_pos_embed.view(seg_pos_embed.shape[0], seg_pos_embed.shape[1], -1).transpose(0, 1)
        # box_hs = self.box2seg_crossattn(
        #     query=torch.zeros_like(bbox_embed),
        #     key=seg_embed,
        #     value=seg_embed,
        #     key_pos=seg_pos_embed,
        #     query_pos=bbox_embed,
        #     key_padding_mask=img_masks.view(img_masks.shape[0], -1).bool(),
        # )[-1]
        # box_pos_embed = torch.zeros_like(bbox_embed)
        # concat_pos_embed = torch.cat((box_pos_embed, seg_pos_embed), dim=0)
        # concat_boxseg_embed = torch.cat((bbox_embed, seg_embed), dim=0)
        # seg_hs = self.segboxCA(query=torch.zeros_like(img_embed), key=concat_boxseg_embed, value=concat_boxseg_embed, key_pos=concat_pos_embed,query_pos=img_embed, query_key_padding_mask=None)[-1]

        # ! type 5 box align pool (B,C) seg mask pool (B,C) box-seg concat(B,2,C) as query CA(query to image)
        # img_embed = img_feat.view(img_feat.shape[0], img_feat.shape[1], -1).permute(2, 0, 1)
        # img_masks, seg_pos_embed = self.x_mask_pos_enc(seg, seg.shape[-2:])
        # seg_pos_embed = seg_pos_embed.view(seg_pos_embed.shape[0], seg_pos_embed.shape[1], -1).permute(2, 0, 1)
        # box_feat, [seg_feat_pos, seg_feat_neg] = self.boxsegpooler(bbox, seg, img_feat)
        # box_seg_query = torch.stack((box_feat, seg_feat_pos), dim=1).transpose(0, 1)
        # hs_query = self.segboxCA(
        #     query=torch.zeros_like(box_seg_query),
        #     key=img_embed,
        #     value=img_embed,
        #     key_pos=seg_pos_embed,
        #     query_pos=box_seg_query,
        #     # query_key_padding_mask=None,
        #     key_padding_mask=seg.argmax(1).reshape(seg.shape[0], -1).bool(),
        # )
        # box_hs, seg_hs = hs_query[-1].split((1, 1), dim=0)
        # seg_hs = seg_hs * img_embed

        # ! type 6 boxalign box-seg(B,1+256,C) SA(box-seg) CA(box to img) CA(seg to img)
        # # img_embedding
        img_feat = self.img_embedding(img_feat)
        img_embed = img_feat.view(img_feat.shape[0], img_feat.shape[1], -1).permute(2, 0, 1)

        # box seg embedding
        box_feat, [seg_feat_pos, seg_feat_neg] = self.boxsegpooler(bbox, seg, img_feat)

        img_masks, seg_pos_embed = self.x_mask_pos_enc(img_feat, img_feat.shape[-2:])
        seg_pos_embed = seg_pos_embed.view(seg_pos_embed.shape[0], seg_pos_embed.shape[1], -1).permute(2, 0, 1)

        # TODO query key padding mask add
        box_seg_query = torch.cat((box_feat.unsqueeze(1), seg_feat_pos), dim=1).transpose(0, 1)
        # attn_mask_pos = seg.argmax(1).reshape(seg.shape[0], -1)
        # attn_mask_pos = attn_mask_pos.float().masked_fill(attn_mask_pos == 1, float('-inf')).masked_fill(attn_mask_pos == 0, float(0.0))
        box_seg_pos_embed = torch.cat((torch.zeros_like(box_feat.unsqueeze(0)), seg_pos_embed), dim=0)
        box_seg_query = self.segboxSA(query=box_seg_query, key=None, value=None, query_pos=box_seg_pos_embed, query_key_padding_mask=None)
        box_embed, seg_embed = box_seg_query.split((1, seg_feat_pos.shape[1]), dim=0)
        box_hs = self.box2img_crossattn(
            query=torch.zeros_like(box_embed),
            key=img_embed,
            value=img_embed,
            key_pos=seg_pos_embed,
            query_pos=box_embed,
            # attn_masks=attn_mask_pos,
            # query_key_padding_mask=~seg.argmax(1).reshape(seg.shape[0], -1).bool(),
            # key_padding_mask=~seg.argmax(1).reshape(seg.shape[0], -1).bool(),  # ! add extra segmask
        )[-1]
        seg_hs = self.seg2img_crossattn(
            query=torch.zeros_like(seg_embed),
            key=img_embed,
            value=img_embed,
            key_pos=seg_pos_embed,
            query_pos=seg_embed,
            # query_key_padding_mask=None,
            # key_padding_mask=seg.argmax(1).reshape(seg.shape[0], -1).bool(),  # ! add extra segmask
        )[-1]

        # ! type 7 boxalign box-seg(B,1+256,C) SA(box-seg) CA(box2text, seg2text) CA(box2img, seg2img)
        # # img_embedding
        # img_feat = self.img_embedding(img_feat)
        # lan_feat = self.lan_embedding(lan_feat)
        # img_embed = img_feat.view(img_feat.shape[0], img_feat.shape[1], -1).permute(2, 0, 1)

        # # boxalign segmasked
        # box_feat, [seg_feat_pos, seg_feat_neg] = self.boxsegpooler(bbox, seg, img_feat)
        # # pos embedding and masks
        # img_masks, seg_pos_embed = self.x_mask_pos_enc(img_feat, img_feat.shape[-2:])
        # seg_pos_embed = seg_pos_embed.view(seg_pos_embed.shape[0], seg_pos_embed.shape[1], -1).permute(2, 0, 1)
        # box_seg_pos_embed = torch.cat((torch.zeros_like(box_feat.unsqueeze(0)), seg_pos_embed), dim=0)
        # text_pos_embed = self.position_embedding_1d(lan_feat).unsqueeze(0).repeat(lan_feat.shape[0], 1, 1).permute(1, 0, 2).cuda()

        # # box-seg(B,1+256,C)
        # box_seg_query = torch.cat((box_feat.unsqueeze(1), seg_feat_pos), dim=1).transpose(0, 1)
        # box_seg_query = self.segboxSA(query=box_seg_query, key=None, value=None, query_pos=box_seg_pos_embed, query_key_padding_mask=None)
        # box_embed, seg_embed = box_seg_query.split((1, seg_feat_pos.shape[1]), dim=0)
        # # box2text, seg2text
        # box_embed = self.box2text_crossattn(
        #     query=torch.zeros_like(box_embed),
        #     key=lan_feat.transpose(0, 1),
        #     value=lan_feat.transpose(0, 1),
        #     key_pos=text_pos_embed,
        #     query_pos=box_embed,
        #     key_padding_mask=lan_mask.bool(),
        # )[-1]
        # seg_embed = self.seg2text_crossattn(
        #     query=torch.zeros_like(seg_embed),
        #     key=lan_feat.transpose(0, 1),
        #     value=lan_feat.transpose(0, 1),
        #     key_pos=text_pos_embed,
        #     query_pos=seg_embed,
        #     key_padding_mask=lan_mask.bool(),
        # )[-1]

        # # box2img, seg2img
        # box_hs = self.box2img_crossattn(
        #     query=torch.zeros_like(box_embed),
        #     key=img_embed,
        #     value=img_embed,
        #     key_pos=seg_pos_embed,
        #     query_pos=box_embed,
        # )[-1]
        # seg_hs = self.seg2img_crossattn(
        #     query=torch.zeros_like(seg_embed),
        #     key=img_embed,
        #     value=img_embed,
        #     key_pos=seg_pos_embed,
        #     query_pos=seg_embed,
        # )[-1]

        second_seg_mask = seg_hs.permute(1, 2, 0).reshape(seg.shape[0], -1, seg.shape[-2], seg.shape[-1])
        second_bbox = box_hs.transpose(0, 1).reshape(bbox.shape[0], -1)

        return second_bbox, second_seg_mask


class UnifiedInteractionModule(nn.Module):
    def __init__(self, input_channels=256, box_weights=[0.1, 1.0], weighted_compose="none", enable_box_coorinate_embed=False):
        super(UnifiedInteractionModule, self).__init__()
        self.box2text_crossattn = DetrTransformerDecoder(
            embed_dim=input_channels,
            num_heads=8,
            attn_dropout=0.1,
            feedforward_dim=1024,
            ffn_dropout=0.1,
            num_layers=1,
            return_intermediate=False,
            post_norm=True,
        )
        self.seg2text_crossattn = DetrTransformerDecoder(
            embed_dim=input_channels,
            num_heads=8,
            attn_dropout=0.1,
            feedforward_dim=1024,
            ffn_dropout=0.1,
            num_layers=1,
            return_intermediate=False,
            post_norm=True,
        )

        self.box2img_crossattn = DetrTransformerDecoder(
            embed_dim=input_channels,
            num_heads=8,
            attn_dropout=0.1,
            feedforward_dim=1024,
            ffn_dropout=0.1,
            num_layers=3,
            return_intermediate=False,
            post_norm=True,
        )
        self.seg2img_crossattn = DetrTransformerDecoder(
            embed_dim=input_channels,
            num_heads=8,
            attn_dropout=0.1,
            feedforward_dim=1024,
            ffn_dropout=0.1,
            num_layers=3,
            return_intermediate=False,
            post_norm=True,
        )
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=input_channels // 2,
            temperature=10000,
            normalize=True,
        )
        self.position_embedding_1d = PositionEmbeddingSine1D(
            num_pos_feats=input_channels // 2,
            temperature=10000,
            normalize=True,
        )

        self.sa = DetrTransformerEncoder(
            embed_dim=input_channels,
            num_heads=8,
            attn_dropout=0.1,
            feedforward_dim=1024,
            ffn_dropout=0.1,
            num_layers=3,
            post_norm=False,
        )

        self.segboxCA = DetrTransformerDecoder(
            embed_dim=input_channels,
            num_heads=8,
            attn_dropout=0.1,
            feedforward_dim=1024,
            ffn_dropout=0.1,
            num_layers=3,
            return_intermediate=False,
            post_norm=True,
        )

        # self.img_embedding = nn.Conv2d(input_channels, hidden_channels, 1)
        # self.seg_embedding = nn.Conv2d(input_channels, hidden_channels, 1)
        # self.seg_embedding = MLP(input_channels, hidden_channels, hidden_channels, 1)
        # self.lan_embedding = MLP(input_channels, hidden_channels, hidden_channels, 1)
        # self.box_embedding = MLP(input_channels, hidden_channels, hidden_channels, 1)
        self.box_weights = box_weights
        self.boxsegpooler = BoxSegPooler()
        self.enable_box_coorinate_embed = enable_box_coorinate_embed
        if self.enable_box_coorinate_embed:
            self.box_coorinate_embed = nn.Linear(4, input_channels)
        self.img_embedding = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, bias=False), nn.BatchNorm2d(input_channels), nn.ReLU(inplace=True)
        )
        self.box_linear = MLP(input_channels * 2, input_channels, input_channels, 3)
        if weighted_compose=="boxmask":
            self.seg_linear = MLP(input_channels * 4, input_channels, input_channels, 3)
        else:
            self.seg_linear = MLP(input_channels * 2, input_channels, input_channels, 3)
        self.weighted_compose = weighted_compose

    def x_mask_pos_enc(self, x, img_shape):
        batch_size = x.size(0)
        input_img_h, input_img_w = img_shape
        x_mask = x.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_shape
            x_mask[img_id, :img_h, :img_w] = 0

        x_mask = F.interpolate(x_mask.unsqueeze(1), size=x.size()[-2:]).to(torch.bool).squeeze(1)
        x_pos_embeds = self.position_embedding(x_mask)
        return x_mask, x_pos_embeds

    def generate_box_mask(self, box, image_feat, weights=[0.1, 1.0]):
        bs = box.size(0)
        input_img_h, input_img_w = image_feat.shape[-2:]
        box_mask = image_feat.new_ones((bs, input_img_h, input_img_w)) * weights[0]
        for idx in range(bs):
            bbox = xywh_to_x1y1x2y2(box[idx])
            x1, y1, x2, y2 = bbox * torch.tensor([input_img_w, input_img_h, input_img_w, input_img_h]).cuda()
            x1, y1, x2, y2 = x1.floor().int(), y1.floor().int(), x2.ceil().int(), y2.ceil().int()
            x1.clamp_(0, input_img_w)
            y1.clamp_(0, input_img_h)
            x2.clamp_(0, input_img_w)
            y2.clamp_(0, input_img_h)
            box_mask[idx, y1:y2, x1:x2] = weights[1]
        return box_mask

    def forward(self, pred_box, pred_mask, img_feat, lan_feat=None, lan_mask=None):
        B, _, H, W = img_feat.shape
        img_feat = self.img_embedding(img_feat)

        img_masks, seg_pos_embed = self.x_mask_pos_enc(img_feat, img_feat.shape[-2:])
        seg_pos_embed = seg_pos_embed.view(seg_pos_embed.shape[0], seg_pos_embed.shape[1], -1).permute(2, 0, 1)

        box_feat, _ = self.boxsegpooler(pred_box, pred_mask, img_feat)
        if self.enable_box_coorinate_embed:
            box_embed = self.box_coorinate_embed(pred_box)
            box_feat = torch.cat((box_feat, box_embed), dim=1)
            box_feat = self.box_linear(box_feat)

        lan_embed = lan_feat.transpose(0, 1)

        # # box2text, seg2text
        # box_embed = self.box2text_crossattn(
        #     query=box_embed,
        #     key=lan_feat.transpose(0, 1),
        #     value=lan_feat.transpose(0, 1),
        #     # key_pos=text_pos_embed,
        #     # query_pos=box_embed,
        #     key_padding_mask=lan_mask.bool(),
        # )[-1]
        # seg_embed = self.seg2text_crossattn(
        #     query=seg_embed,
        #     key=lan_feat.transpose(0, 1),
        #     value=lan_feat.transpose(0, 1),
        #     # key_pos=text_pos_embed,
        #     # query_pos=seg_embed,
        #     key_padding_mask=lan_mask.bool(),
        # )[-1]
        # ! type 1
        # img_embed = unified_img_feat.view(unified_img_feat.shape[0], unified_img_feat.shape[1], -1).permute(2, 0, 1)
        # box_embed = box_feat.unsqueeze(1).transpose(0, 1)
        # seg_embed = seg_feat.flatten(2).permute(2, 0, 1)
        # box_hs = self.box2img_crossattn(
        #     query=box_embed,
        #     key=img_embed,
        #     value=img_embed,
        #     key_pos=seg_pos_embed,
        #     # query_pos=box_embed,
        #     # attn_masks=attn_mask_pos,
        #     # query_key_padding_mask=~seg.argmax(1).reshape(seg.shape[0], -1).bool(),
        #     # key_padding_mask=~seg.argmax(1).reshape(seg.shape[0], -1).bool(),  # ! add extra segmask
        # )[-1]

        # seg_hs = self.seg2img_crossattn(
        #     query=seg_embed,
        #     key=img_embed,
        #     value=img_embed,
        #     key_pos=seg_pos_embed,
        #     query_pos=seg_pos_embed,
        #     # query_key_padding_mask=None,
        #     # key_padding_mask=seg.argmax(1).reshape(seg.shape[0], -1).bool(),  # ! add extra segmask
        # )[-1]

        # ! type 2 concat SA
        # box_embed = box_feat.unsqueeze(1).transpose(0, 1)
        # seg_embed = torch.cat((unified_img_feat, img_feat), dim=1) #(B,2C,H, W)
        # seg_embed = seg_embed.flatten(2).transpose(1,2) # (B,Ni,2C)
        # seg_embed = self.seg_linear(seg_embed) # (B,Ni,C)
        # seg_embed = seg_embed.transpose(0,1)# (Ni,B,C)

        # box_hs = self.box2img_crossattn(
        #     query=box_embed,
        #     key=seg_embed,
        #     value=seg_embed,
        #     key_pos=seg_pos_embed,
        #     # query_pos=box_embed,
        #     # attn_masks=attn_mask_pos,
        #     # query_key_padding_mask=~seg.argmax(1).reshape(seg.shape[0], -1).bool(),
        #     # key_padding_mask=~seg.argmax(1).reshape(seg.shape[0], -1).bool(),  # ! add extra segmask
        # )[-1]

        # seg_hs = self.sa(
        #     query=seg_embed,
        #     key=None,
        #     value=None,
        #     query_pos=seg_pos_embed,
        #     # query_key_padding_mask=img_masks,
        # )

        # ! type 3 add text interaction concat SA
        box_embed = box_feat.unsqueeze(1).transpose(0, 1)
        heatmap_mean = torch.mean(img_feat, dim=1, keepdim=True)
        min_vals = heatmap_mean.amin(dim=(2, 3), keepdim=True)  # 在 H 和 W 维度上取最小值
        max_vals = heatmap_mean.amax(dim=(2, 3), keepdim=True)  # 在 H 和 W 维度上取最大值
        img_feat_norm = (heatmap_mean - min_vals) / (max_vals-min_vals+1e-8)
        if self.weighted_compose == "none":
            seg_embed = img_feat.flatten(2).transpose(1, 2)  # (B,Ni,C)
            unified_img_feat = img_feat
            unified_heatmap = img_feat_norm
        else:
            if self.weighted_compose == "box":
                box_mask = self.generate_box_mask(pred_box, img_feat, weights=self.box_weights)  # (N,H,W)
                box_feat = img_feat * box_mask.unsqueeze(1)
                unified_img_feat = img_feat * box_mask.unsqueeze(1)
                box_heatmap = img_feat_norm * box_mask.unsqueeze(1)
                unified_heatmap = img_feat_norm * box_mask.unsqueeze(1)
                seg_embed = torch.cat((unified_img_feat, img_feat), dim=1)  # (B,2C,H, W)
            elif self.weighted_compose == "mask":
                pred_mask = pred_mask.sigmoid()
                seg_feat = img_feat * pred_mask
                unified_img_feat = img_feat * pred_mask
                seg_heatmap = img_feat_norm * pred_mask
                unified_heatmap = img_feat_norm * pred_mask
                seg_embed = torch.cat((unified_img_feat, img_feat), dim=1)  # (B,2C,H, W)
            elif self.weighted_compose == "boxmask":
                box_mask = self.generate_box_mask(pred_box, img_feat, weights=self.box_weights)  # (N,H,W)
                pred_mask = pred_mask.sigmoid()
                seg_feat = img_feat * pred_mask
                box_feat = img_feat * box_mask.unsqueeze(1)
                unified_img_feat = seg_feat * box_mask.unsqueeze(1)
                seg_heatmap = img_feat_norm * pred_mask
                box_heatmap = img_feat_norm * box_mask.unsqueeze(1)
                unified_heatmap = img_feat_norm * pred_mask * box_mask.unsqueeze(1)
                seg_embed = torch.cat((seg_feat, box_feat, unified_img_feat, img_feat), dim=1)  # (B,2C,H, W)
            else:
                raise TypeError()
            seg_embed = seg_embed.flatten(2).transpose(1, 2)  # (B,Ni,2C)
            seg_embed = self.seg_linear(seg_embed)  # (B,Ni,C)
        seg_embed = seg_embed.transpose(0, 1)  # (Ni,B,C)

        # B2T
        box_hs = self.box2text_crossattn(
            query=box_embed,
            key=lan_embed,
            value=lan_embed,
            key_padding_mask=lan_mask.bool(),
        )[-1]
        # B2I
        box_hs = self.box2img_crossattn(query=box_hs, key=seg_embed, value=seg_embed, key_pos=seg_pos_embed)[-1]
        # S2T
        seg_hs = self.seg2text_crossattn(
            query=seg_embed,
            key=lan_embed,
            value=lan_embed,
            # key_pos=seg_pos_embed,
            key_padding_mask=lan_mask.bool(),
        )[-1]
        # SA
        seg_hs = self.sa(
            query=seg_embed,
            key=None,
            value=None,
            query_pos=seg_pos_embed,
        )

        second_seg_mask = seg_hs.permute(1, 2, 0).reshape(B, -1, H, W)
        second_bbox = box_hs.transpose(0, 1).reshape(B, -1)
        
        extra = {"unified_img_feat": unified_heatmap,
                 "img_feat": img_feat_norm}
        if "mask" in self.weighted_compose:
            extra["seg_feat"] = seg_heatmap
        if "box" in self.weighted_compose:
            extra["box_feat"] = box_heatmap
        return second_bbox, second_seg_mask, extra

    def forward_debug(self, bbox, seg, img_feat, lan_feat=None, lan_mask=None):

        # ! type 6 boxalign box-seg(B,1+256,C) SA(box-seg) CA(box to img) CA(seg to img)
        # # img_embedding
        img_feat = self.img_embedding(img_feat)
        img_embed = img_feat.view(img_feat.shape[0], img_feat.shape[1], -1).permute(2, 0, 1)

        # box seg embedding
        box_feat, [seg_feat_pos, seg_feat_neg] = self.boxsegpooler(bbox, seg, img_feat)

        img_masks, seg_pos_embed = self.x_mask_pos_enc(img_feat, img_feat.shape[-2:])
        seg_pos_embed = seg_pos_embed.view(seg_pos_embed.shape[0], seg_pos_embed.shape[1], -1).permute(2, 0, 1)

        # TODO query key padding mask add
        box_seg_query = torch.cat((box_feat.unsqueeze(1), seg_feat_pos), dim=1).transpose(0, 1)
        # attn_mask_pos = seg.argmax(1).reshape(seg.shape[0], -1)
        # attn_mask_pos = attn_mask_pos.float().masked_fill(attn_mask_pos == 1, float('-inf')).masked_fill(attn_mask_pos == 0, float(0.0))
        box_seg_pos_embed = torch.cat((torch.zeros_like(box_feat.unsqueeze(0)), seg_pos_embed), dim=0)
        box_seg_query = self.segboxSA(query=box_seg_query, key=None, value=None, query_pos=box_seg_pos_embed, query_key_padding_mask=None)
        box_embed, seg_embed = box_seg_query.split((1, seg_feat_pos.shape[1]), dim=0)
        box_hs = self.box2img_crossattn(
            query=torch.zeros_like(box_embed),
            key=img_embed,
            value=img_embed,
            key_pos=seg_pos_embed,
            query_pos=box_embed,
            # attn_masks=attn_mask_pos,
            # query_key_padding_mask=~seg.argmax(1).reshape(seg.shape[0], -1).bool(),
            # key_padding_mask=~seg.argmax(1).reshape(seg.shape[0], -1).bool(),  # ! add extra segmask
        )[-1]
        seg_hs = self.seg2img_crossattn(
            query=torch.zeros_like(seg_embed),
            key=img_embed,
            value=img_embed,
            key_pos=seg_pos_embed,
            query_pos=seg_embed,
            # query_key_padding_mask=None,
            # key_padding_mask=seg.argmax(1).reshape(seg.shape[0], -1).bool(),  # ! add extra segmask
        )[-1]

        second_seg_mask = seg_hs.permute(1, 2, 0).reshape(seg.shape[0], -1, seg.shape[-2], seg.shape[-1])
        second_bbox = box_hs.transpose(0, 1).reshape(bbox.shape[0], -1)

        return second_bbox, second_seg_mask
