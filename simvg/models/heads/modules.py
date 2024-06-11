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
                nn.ConvTranspose2d(input_channels, input_channels // 2, 2, 2),
                build_norm_layer(norm_cfg, input_channels // 2)[1],
                nn.GELU(),
                nn.ConvTranspose2d(input_channels // 2, input_channels // 4, 2, 2),
            )
            self.predict = nn.Conv2d(input_channels // 4, 1, 1)
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

    def forward(self, bbox, seg, img_feat):
        B, C, H, W = img_feat.shape
        # * box pooling
        bbox_x1y1x2y2 = xywh_to_x1y1x2y2(bbox)
        # ROI Align
        # bbox_feat = self.box_align(img_feat, [bbox_x1y1x2y2])
        # ROI Pooling
        batch_indices = torch.arange(B).unsqueeze(1).cuda()
        rois = torch.cat([batch_indices, bbox_x1y1x2y2], dim=1)
        bbox_feat = roi_pool(img_feat, rois, output_size=(2, 2))
        # avg pooling
        bbox_feat = self.box_pooler(bbox_feat).reshape(B, C)
        # * seg pooling 
        # @TODO
        seg_mask = (seg.sigmoid()>0.5).int().squeeze(1)
        seg_feat_pos, seg_feat_neg = self.masked_pool(img_feat, seg_mask)
        seg_feat_pos = self.seg_pooler(seg_feat_pos).reshape(B, C)
        seg_feat_neg = self.seg_pooler(seg_feat_neg).reshape(B, C)
        # seg_feat_pos = seg_feat_pos.reshape(B, C, -1).transpose(1, 2)
        # seg_feat_neg = seg_feat_neg.reshape(B, C, -1).transpose(1, 2)
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
    def __init__(self, input_channels=256, segaug=False):
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
        self.boxsegpooler = BoxSegPooler(sample_scale=1 / 16)
        self.segaug = segaug

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
