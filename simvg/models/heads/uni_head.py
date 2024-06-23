import torch
from torch import nn
from torch.nn import functional as F
import torch
from simvg.models import HEADS
import pycocotools.mask as maskUtils
import numpy as np

from .unet_head import *
from simvg.models.heads.uni_head_simple import bbox_loss, box_norm, boxseg_iouloss2, seg_loss, clip_infonNCEloss, consistencyloss
from simvg.models.losses.boxloss import BoxLoss
from ..losses.contristiveloss import HardMiningTripletLoss
from .modules import BoxSegAttention, BoxSegPooler, QueryAugment, SegBranch, BoxBranch, UnifiedInteractionModule
from .unet_head import SimpleFPN
from ..utils import xywh_to_x1y1x2y2
from ..losses.clip_loss import ClipLoss, get_rank, get_world_size
from .projection import Projector


@HEADS.register_module()
class UniHeadCoarseToFine(nn.Module):
    def __init__(
        self,
        input_channels=768,
        hidden_channels=256,
        query_augment=None,
        loss_weight={"mask": 1.0, "bbox": 0.025, "cons": 0.0},
        threshold={"B2S": 0.1},
        start_epoch=0,
        decoder_upsample_type="none",
        uim={"enable": False, "weighted_compose": "none", "enable_box_coorinate_embed": False, "box_weights": [0.1, 1.0]},
    ):
        super(UniHeadCoarseToFine, self).__init__()
        self.seg_branch_first = SegBranch(hidden_channels, upsample_rate=1)
        self.box_branch_first = BoxBranch(hidden_channels)
        self.decoder_upsample_type = decoder_upsample_type
        if self.decoder_upsample_type == "fpn":
            self.neck = SimpleFPN(
                backbone_channel=hidden_channels,
                in_channels=[hidden_channels // 4, hidden_channels // 2, hidden_channels, hidden_channels],
                out_channels=[hidden_channels // 4, hidden_channels // 2, hidden_channels, hidden_channels * 2],
            )
            self.fpn_decoder = SimpleDecoding(hidden_channels * 2)
        elif self.decoder_upsample_type == "tranposeconv":
            self.neck = SegBranch(hidden_channels, upsample_rate=4).neck
        else:
            self.neck = nn.Identity()
        self.seg_branch_second = SegBranch(hidden_channels, upsample_rate=1)
        self.box_branch_second = BoxBranch(hidden_channels)
        self.box_loss = BoxLoss()
        self.seg_loss = nn.functional.cross_entropy
        self.loss_weight = loss_weight
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.boxsegattn = BoxSegAttention(input_channels=hidden_channels, input_size=input_size)
        self.clip_loss = ClipLoss(rank=get_rank(), world_size=get_world_size())
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.lan_embedding = nn.Linear(input_channels, hidden_channels, bias=False)
        self.img_embedding = nn.Conv2d(input_channels, hidden_channels, kernel_size=1, bias=False)
        self.query_embedding = nn.Linear(input_channels, hidden_channels, bias=False)

        self.seg_cons_embedding = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.box_cons_embedding = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.boxsegpooler = BoxSegPooler()
        self.proj_pixel_level_cons = Projector(word_dim=hidden_channels, in_dim=hidden_channels, hidden_dim=hidden_channels // 2, kernel_size=1)
        # query augment module
        self.query_augment_module = None
        if query_augment is not None:
            self.query_augment_module = QueryAugment(hidden_channels=hidden_channels, num_queries=query_augment["num_queries"])
        self.threshold = threshold
        self.start_epoch = start_epoch
        self.unified_interaction_module = uim["enable"]
        if self.unified_interaction_module:
            self.UIM = UnifiedInteractionModule(
                input_channels=hidden_channels,
                box_weights=uim["box_weights"],
                weighted_compose=uim["weighted_compose"],
                enable_box_coorinate_embed=uim["enable_box_coorinate_embed"],
            )

    def text_pooler(self, lan_feat, lan_mask):
        lan_feat_pooler = torch.cat(list(map(lambda feat, mask: torch.max(feat[mask, :], dim=0, keepdim=True)[0], lan_feat, ~lan_mask)))
        return lan_feat_pooler

    def forward_train(self, x, targets, cls_feat=None, lan_feat=None, lan_mask=None, img=None):
        device = x.device
        target_mask = torch.from_numpy(np.concatenate([maskUtils.decode(target)[None] for target in targets["mask"]])).to(device)
        # all feats embedding to hidden_channels
        img_feat = self.img_embedding(x)
        query_feat = self.query_embedding(cls_feat)
        lan_feat = self.lan_embedding(lan_feat)
        lan_pool = self.text_pooler(lan_feat, lan_mask)
        # ! query augment
        if self.query_augment_module is not None:
            query_feat = self.query_augment_module(query_feat, img_feat, lan_feat, lan_mask)
        # ! stage 1
        pred_bbox = self.box_branch_first(query_feat)
        if self.loss_weight["clip"]["pixel"]:
            pred_mask = self.proj_pixel_level_cons(img_feat, lan_pool)
        else:
            pred_mask = self.seg_branch_first(img_feat)
        pred_mask_first = F.interpolate(pred_mask, size=img.shape[-2:], mode="bilinear", align_corners=True)
        pred_bbox_first = pred_bbox
        target_mask_first = target_mask

        # # * first stage -- box seg pooling -> box_feat(B,C) seg_feat(B,C)
        # # ! box seg pooling
        # box_feat_first, [seg_feat_pos_first, seg_feat_neg_first] = self.boxsegpooler(
        #     pred_bbox,
        #     pred_mask,
        #     img_feat,
        #     gt=False,
        #     img_pool=False,
        # )

        # ! stage 2
        # ! unified interaction
        extra_dict = {}
        if self.unified_interaction_module:
            # pred_bbox_2_tmp, pred_mask_2_tmp = self.boxsegattn(box_feat_first, seg_feat_pos_first, img_feat, lan_feat, lan_mask)
            pred_bbox_2_tmp, pred_mask_2_tmp, extra = self.UIM(pred_bbox, pred_mask, img_feat, lan_feat, lan_mask)
            if "unified_img_feat" in extra:
                extra_dict["unified_img_feat"] = F.interpolate(extra["unified_img_feat"], size=img.shape[-2:], mode="bilinear", align_corners=True)
            if "seg_feat" in extra:
                extra_dict["seg_feat"] = F.interpolate(extra["seg_feat"], size=img.shape[-2:], mode="bilinear", align_corners=True)
            if "box_feat" in extra:
                extra_dict["box_feat"] = F.interpolate(extra["box_feat"], size=img.shape[-2:], mode="bilinear", align_corners=True)
            if "img_feat" in extra:
                extra_dict["img_feat"] = F.interpolate(extra["img_feat"], size=img.shape[-2:], mode="bilinear", align_corners=True)
        else:
            pred_bbox_2_tmp, pred_mask_2_tmp = query_feat, img_feat

        # ! decoder upsample
        if self.decoder_upsample_type == "fpn":
            x_c1, x_c2, x_c3, x_c4 = self.neck(pred_mask_2_tmp)
            pred_mask_up4 = self.fpn_decoder(x_c4, x_c3, x_c2, x_c1)
        elif self.decoder_upsample_type == "tranposeconv":
            pred_mask_up4 = self.neck(pred_mask_2_tmp)
        else:
            pred_mask_up4 = pred_mask_2_tmp

        # ! pixel level cons
        if self.loss_weight["clip"]["pixel"]:
            pred_mask_2 = self.proj_pixel_level_cons(pred_mask_up4, lan_pool)
        else:
            pred_mask_2 = self.seg_branch_second(pred_mask_up4)
        pred_bbox_second = self.box_branch_second(pred_bbox_2_tmp)
        pred_mask_second = F.interpolate(pred_mask_2, size=img.shape[-2:], mode="bilinear", align_corners=True)

        # ! loss func
        loss_mask_first = seg_loss(pred_mask_first, target_mask_first, self.loss_weight["mask"]) * self.loss_weight["stage"]["first"]
        loss_bbox_first = bbox_loss(pred_bbox_first, targets["bbox"], img, self.box_loss) * self.loss_weight["bbox"] * self.loss_weight["stage"]["first"]
        loss_mask_second = seg_loss(pred_mask_second, target_mask, self.loss_weight["mask"]) * self.loss_weight["stage"]["second"]
        loss_bbox_second = bbox_loss(pred_bbox_second, targets["bbox"], img, self.box_loss) * self.loss_weight["bbox"] * self.loss_weight["stage"]["second"]

        # ! clip loss
        loss_clip = torch.tensor([0.0], device=device)
        if self.loss_weight["clip"]["box"] + self.loss_weight["clip"]["seg"] > 0:
            pred_mask_down2seg = F.interpolate(pred_mask_second, size=pred_mask_up4.shape[-2:], mode="bilinear", align_corners=True)
            box_feat, [seg_feat_pos, seg_feat_neg] = self.boxsegpooler(pred_bbox_second, pred_mask_down2seg, pred_mask_up4, gt=False, img_pool=True)
            # * use the groundtruth to do the contristive loss
            # target_mask_tmp = F.interpolate(target_mask.unsqueeze(1), size=pred_mask_up4.shape[-2:], mode="nearest")
            # target_box_tmp = box_norm(targets["bbox"], img)
            # box_feat, [seg_feat_pos, seg_feat_neg] = self.boxsegpooler(target_box_tmp, target_mask_tmp, pred_mask_up4)
            box_level_clip_loss = torch.tensor([0.0], device=device)
            seg_level_clip_loss = torch.tensor([0.0], device=device)
            # box
            box_feat = self.box_cons_embedding(box_feat)
            box_level_clip_loss = clip_infonNCEloss(box_feat, lan_pool, self.clip_loss, self.logit_scale)[0] * self.loss_weight["clip"]["box"]
            # seg
            seg_feat = self.seg_cons_embedding(seg_feat_pos)
            seg_level_clip_loss = clip_infonNCEloss(seg_feat, lan_pool, self.clip_loss, self.logit_scale)[0] * self.loss_weight["clip"]["seg"]
            loss_clip = box_level_clip_loss + seg_level_clip_loss

        # ! cons loss
        loss_cons_first = torch.tensor([0.0], device=device)
        loss_cons_second = torch.tensor([0.0], device=device)
        if (self.loss_weight["boxsegcc"]["S2B"] + self.loss_weight["boxsegcc"]["B2S"] > 0) and targets["epoch"] > self.start_epoch:
            loss_S2B_first, loss_B2S_first, _, _ = boxseg_iouloss2(
                pred_bbox_first, pred_mask_first, B2Sthr=self.threshold["B2S"], S2Bthr=self.threshold["S2B"], box_func=self.box_loss
            )
            loss_S2B_second, loss_B2S_second, _, _ = boxseg_iouloss2(
                pred_bbox_second, pred_mask_second, B2Sthr=self.threshold["B2S"], S2Bthr=self.threshold["S2B"], box_func=self.box_loss
            )
            loss_cons_first = (
                sum(loss_S2B_first) / len(loss_S2B_first) * self.loss_weight["boxsegcc"]["S2B"]
                + sum(loss_B2S_first) / len(loss_B2S_first) * self.loss_weight["boxsegcc"]["B2S"]
            ) * self.loss_weight["stage"]["first"]
            loss_cons_second = (
                sum(loss_S2B_second) / len(loss_S2B_second) * self.loss_weight["boxsegcc"]["S2B"]
                + sum(loss_B2S_second) / len(loss_B2S_second) * self.loss_weight["boxsegcc"]["B2S"]
            ) * self.loss_weight["stage"]["second"]

        loss_mask = loss_mask_first + loss_mask_second
        loss_det = loss_bbox_first + loss_bbox_second
        loss_cons = loss_cons_first + loss_cons_second
        loss_dict = {
            "loss_mask": loss_mask,
            "loss_det": loss_det,
            "loss_cons": loss_cons,
            "loss_clip": loss_clip,
            "loss_mask_first": loss_mask_first,
            "loss_bbox_first": loss_bbox_first,
            "loss_mask_second": loss_mask_second,
            "loss_bbox_second": loss_bbox_second,
            "loss_cons_first": loss_cons_first,
            "loss_cons_second": loss_cons_second,
        }
        pred_dict = {"pred_mask": pred_mask_second, "pred_bbox": pred_bbox_second, "pred_mask_first": pred_mask_first, "pred_bbox_first": pred_bbox_first}
        return loss_dict, pred_dict, extra_dict

    def forward_test(self, x, cls_feat=None, lan_feat=None, lan_mask=None, img=None, targets=None):
        img_feat = self.img_embedding(x)
        query_feat = self.query_embedding(cls_feat)
        lan_feat = self.lan_embedding(lan_feat)
        lan_pool = self.text_pooler(lan_feat, lan_mask)
        # ! query augment
        if self.query_augment_module is not None:
            query_feat = self.query_augment_module(query_feat, img_feat, lan_feat, lan_mask)
        # ! stage 1
        pred_bbox = self.box_branch_first(query_feat)
        if self.loss_weight["clip"]["pixel"]:
            pred_mask = self.proj_pixel_level_cons(img_feat, lan_pool)
        else:
            pred_mask = self.seg_branch_first(img_feat)
        pred_mask_first = F.interpolate(pred_mask, size=img.shape[-2:], mode="bilinear", align_corners=True)
        pred_bbox_first = pred_bbox

        # # * first stage -- box seg pooling -> box_feat(B,C) seg_feat(B,C)
        # # ! box seg pooling
        # box_feat_first, [seg_feat_pos_first, seg_feat_neg_first] = self.boxsegpooler(
        #     pred_bbox,
        #     pred_mask,
        #     img_feat,
        #     gt=False,
        #     img_pool=False,
        # )

        # ! stage 2
        # ! unified interaction
        extra_dict = {}
        if self.unified_interaction_module:
            # pred_bbox_2_tmp, pred_mask_2_tmp = self.boxsegattn(box_feat_first, seg_feat_pos_first, img_feat, lan_feat, lan_mask)
            pred_bbox_2_tmp, pred_mask_2_tmp, extra = self.UIM(pred_bbox, pred_mask, img_feat, lan_feat, lan_mask)
            if "unified_img_feat" in extra:
                extra_dict["unified_img_feat"] = F.interpolate(extra["unified_img_feat"], size=img.shape[-2:], mode="bilinear", align_corners=True)
            if "seg_feat" in extra:
                extra_dict["seg_feat"] = F.interpolate(extra["seg_feat"], size=img.shape[-2:], mode="bilinear", align_corners=True)
            if "box_feat" in extra:
                extra_dict["box_feat"] = F.interpolate(extra["box_feat"], size=img.shape[-2:], mode="bilinear", align_corners=True)
            if "img_feat" in extra:
                extra_dict["img_feat"] = F.interpolate(extra["img_feat"], size=img.shape[-2:], mode="bilinear", align_corners=True)
        else:
            pred_bbox_2_tmp, pred_mask_2_tmp = query_feat, img_feat

        # ! decoder upsample
        if self.decoder_upsample_type == "fpn":
            x_c1, x_c2, x_c3, x_c4 = self.neck(pred_mask_2_tmp)
            pred_mask_up4 = self.fpn_decoder(x_c4, x_c3, x_c2, x_c1)
        elif self.decoder_upsample_type == "tranposeconv":
            pred_mask_up4 = self.neck(pred_mask_2_tmp)
        else:
            pred_mask_up4 = pred_mask_2_tmp

        # ! pixel level cons
        if self.loss_weight["clip"]["pixel"]:
            pred_mask_2 = self.proj_pixel_level_cons(pred_mask_up4, lan_pool)
        else:
            pred_mask_2 = self.seg_branch_second(pred_mask_up4)
        pred_bbox_second = self.box_branch_second(pred_bbox_2_tmp)
        pred_mask_second = F.interpolate(pred_mask_2, size=img.shape[-2:], mode="bilinear", align_corners=True)

        pred_dict = {"pred_mask": pred_mask_second, "pred_bbox": pred_bbox_second, "pred_mask_first": pred_mask_first, "pred_bbox_first": pred_bbox_first}
        return pred_dict, extra_dict


@HEADS.register_module()
class UniHead(nn.Module):
    def __init__(self, input_channels, loss_weight={"mask": 1.0, "bbox": 1.0, "cons": 0.1}, loss_stage={"first": 1.0, "second": 1.0}):
        super(UniHead, self).__init__()
        self.seg_branch = SegBranch(input_channels, upsample_rate=1)
        self.box_branch = BoxBranch(input_channels)
        self.boxsegpooler = BoxSegPooler()

        self.loss_weight = loss_weight
        self.loss_stage = loss_stage

        self.box_loss = BoxLoss()
        self.seg_loss = nn.functional.cross_entropy
        self.cons_loss = HardMiningTripletLoss(margin=0.5, normalize_feature=True)

    def forward_train(self, x, targets, cls_feat=None, lan_feature=None, lan_mask=None, img=None):
        # model part
        pred_bbox = self.box_branch(cls_feat)
        pred_mask = self.seg_branch(x)
        # box-seg feat pooler
        box_feat, seg_feat = self.boxsegpooler(pred_bbox, pred_mask, x)

        pred_mask = F.interpolate(pred_mask, size=img.shape[-2:], mode="bilinear", align_corners=True)
        # loss func
        target_mask = torch.from_numpy(np.concatenate([maskUtils.decode(target)[None] for target in targets["mask"]])).cuda()
        loss_mask = seg_loss(pred_mask, target_mask, self.seg_loss) * self.loss_weight["mask"]
        loss_bbox = bbox_loss(pred_bbox, targets["bbox"], img, self.box_loss) * self.loss_weight["bbox"]
        loss_cons = consistencyloss(box_feat, seg_feat, self.cons_loss) * self.loss_weight["cons"]
        loss_mask = loss_mask
        loss_det = loss_bbox
        loss_cons = loss_cons
        loss_dict = {
            "loss_mask": loss_mask,
            "loss_det": loss_det,
            "loss_cons": loss_cons,
        }
        pred_dict = {"pred_mask": pred_mask, "pred_bbox": pred_bbox}
        return loss_dict, pred_dict

    def forward_test(self, x, cls_feat=None, lan_feature=None, lan_mask=None, img=None):
        # model part
        pred_bbox = self.box_branch(cls_feat)
        pred_mask = self.seg_branch(x)
        pred_mask = F.interpolate(pred_mask, size=img.shape[-2:], mode="bilinear", align_corners=True)

        pred_dict = {"pred_mask": pred_mask, "pred_bbox": pred_bbox}
        return pred_dict
