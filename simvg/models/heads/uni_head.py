import torch
from torch import nn
from torch.nn import functional as F
import torch
from simvg.models import HEADS
import pycocotools.mask as maskUtils
import numpy as np
from .unet_head import *
from simvg.models.losses.boxloss import BoxLoss
from ..losses.contristiveloss import HardMiningTripletLoss
from .modules import BoxSegAttention, BoxSegPooler, SegBranch, BoxBranch
from .unet_head import SimpleFPN
from ..utils import xywh_to_x1y1x2y2
from ..losses.clip_loss import ClipLoss, get_rank, get_world_size
from .projection import Projector


def seg_loss(inputs, target, loss_func):
    weight = torch.FloatTensor([0.9, 1.1]).cuda()
    return loss_func(inputs, target.long(), weight=weight)


def bbox_loss(inputs, targets, img, loss_func):
    gt_bbox = torch.stack(targets, dim=0)
    norm_bbox = torch.zeros_like(gt_bbox, device=gt_bbox.device)

    norm_bbox[:, 0] = (gt_bbox[:, 0] + gt_bbox[:, 2]) / 2.0  # x_center
    norm_bbox[:, 1] = (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2.0  # y_center
    norm_bbox[:, 2] = gt_bbox[:, 2] - gt_bbox[:, 0]  # w
    norm_bbox[:, 3] = gt_bbox[:, 3] - gt_bbox[:, 1]  # h

    img_size = torch.tensor(img.shape[-2:], device=img.device)
    norm_bbox = norm_bbox / (img_size.unsqueeze(0).repeat((img.shape[0], 2)))
    loss, loss_box, loss_giou = loss_func(inputs, norm_bbox)
    return loss


def consistencyloss(bbox_feat, seg_feat, loss_func):
    # consistencyloss
    label = torch.arange(bbox_feat.shape[0])
    feats = torch.concat((bbox_feat, seg_feat), dim=0)
    labels = torch.concat((label, label))
    consloss = loss_func(feats, labels)
    return consloss


def clip_infonNCEloss(feat1, feat2, loss_func, logit_scale):
    feat1 = F.normalize(feat1, dim=-1)
    feat2 = F.normalize(feat2, dim=-1)
    return loss_func(feat1, feat2, logit_scale.exp())


def boxseg_iouloss(box, mask):
    # xywh_to_x1y1x2y2
    box = xywh_to_x1y1x2y2(box)
    mask = mask.argmax(1).squeeze(1)
    B, H, W = mask.shape
    iou_loss = []
    for b in range(B):
        x1, y1, x2, y2 = (box[b] * torch.tensor([W, H, W, H]).cuda()).to(torch.int)
        # Ensure coordinates are within bounds
        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)
        # Extract the sub-mask corresponding to the box
        sub_mask = mask[b, y1:y2, x1:x2]
        # Calculate the intersection
        intersection = sub_mask.sum()
        # Calculate the area of the mask
        mask_area = mask[b].sum()
        # Calculate the IoU
        if mask_area > 0:
            iou = intersection / mask_area
        else:
            iou = torch.tensor([0.0], device=mask.device)
        iou_loss.append(1 - iou)
    return sum(iou_loss) / len(iou_loss)


@HEADS.register_module()
class UniHeadCoarseToFine(nn.Module):
    def __init__(
        self, input_channels=768, hidden_channels=256, loss_weight={"mask": 1.0, "bbox": 1.0}, loss_stage={"first": 1.0, "second": 1.0}, clip_loss_weight={}
    ):
        super(UniHeadCoarseToFine, self).__init__()
        self.seg_branch_first = SegBranch(hidden_channels, upsample_rate=1)
        self.box_branch_first = BoxBranch(hidden_channels)
        self.seg_branch_second = SegBranch(hidden_channels, upsample_rate=4)
        self.box_branch_second = BoxBranch(hidden_channels)
        self.box_loss = BoxLoss()
        self.seg_loss = nn.functional.cross_entropy
        self.loss_weight = loss_weight
        self.loss_stage = loss_stage
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.boxsegattn = BoxSegAttention(input_channels=hidden_channels)
        self.cons_loss = HardMiningTripletLoss(margin=0.5, normalize_feature=True)
        self.clip_loss = ClipLoss(rank=get_rank(), world_size=get_world_size())

        self.lan_embedding = nn.Linear(input_channels, hidden_channels, bias=False)
        self.img_embedding = nn.Conv2d(input_channels, hidden_channels, kernel_size=1, bias=False)
        self.query_embedding = nn.Linear(input_channels, hidden_channels, bias=False)

        self.seg_cons_embedding = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.box_cons_embedding = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.clip_loss_weight = clip_loss_weight
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.boxsegpooler = BoxSegPooler(sample_scale=1 / 16)
        self.fpn = SimpleFPN(
            backbone_channel=hidden_channels,
            in_channels=[hidden_channels // 4, hidden_channels // 2, hidden_channels, hidden_channels],
            out_channels=[hidden_channels // 8, hidden_channels // 4, hidden_channels // 2, hidden_channels],
        )
        self.fpn_decoder = SimpleDecoding(hidden_channels)
        self.proj_pixel_level_cons = Projector(word_dim=hidden_channels, in_dim=hidden_channels // 2)

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
        # ! stage 1
        pred_mask = self.seg_branch_first(img_feat)
        pred_bbox = self.box_branch_first(query_feat)

        pred_bbox_first = pred_bbox

        # * bbox branch from the img avg feature
        # pred_bbox = self.box_branch(self.pool(x).squeeze())

        # * downsample target mask to the same size as pred mask
        # pred_mask_first = pred_mask
        # target_mask_first = F.interpolate(target_mask.float().unsqueeze(1), size=pred_mask.shape[-2:], mode="bilinear", align_corners=True).squeeze(1).to(torch.uint8)

        # * upsample pred_mask to the same size as target_mask
        pred_mask_first = F.interpolate(pred_mask, size=img.shape[-2:], mode="bilinear", align_corners=True)
        target_mask_first = target_mask

        # * first stage -- box seg pooling -> box_feat(B,C) seg_feat(B,C)
        box_feat_first, [seg_feat_pos_first, seg_feat_neg_first] = self.boxsegpooler(pred_bbox, pred_mask, img_feat)

        # ! stage 2
        pred_bbox_2_tmp, pred_mask_2_tmp = self.boxsegattn(box_feat_first, seg_feat_pos_first, img_feat, lan_feat, lan_mask)
        pred_mask_2 = self.seg_branch_second(pred_mask_2_tmp)
        pred_bbox_2 = self.box_branch_second(pred_bbox_2_tmp)

        pred_mask_second = F.interpolate(pred_mask_2, size=img.shape[-2:], mode="bilinear", align_corners=True)
        pred_bbox_second = pred_bbox_2

        # ! pixel level cons
        x_c1, x_c2, x_c3, x_c4 = self.fpn(pred_mask_2_tmp)
        pred_mask_2_up4 = self.fpn_decoder(x_c4, x_c3, x_c2, x_c1)
        pixel_level_cons_mask = self.proj_pixel_level_cons(pred_mask_2_up4, self.text_pooler(lan_feat, lan_mask))
        pixel_level_cons_pred_mask = F.interpolate(pixel_level_cons_mask, size=img.shape[-2:], mode="bilinear", align_corners=True)

        pred_mask_second = pred_mask_second + pixel_level_cons_pred_mask + pred_mask_first

        # ! loss func
        loss_mask_first = seg_loss(pred_mask_first, target_mask_first, self.seg_loss) * self.loss_weight["mask"] * self.loss_stage["first"]
        loss_bbox_first = bbox_loss(pred_bbox_first, targets["bbox"], img, self.box_loss) * self.loss_weight["bbox"] * self.loss_stage["first"]
        loss_mask_second = seg_loss(pred_mask_second, target_mask, self.seg_loss) * self.loss_weight["mask"] * self.loss_stage["second"]
        loss_bbox_second = bbox_loss(pred_bbox_second, targets["bbox"], img, self.box_loss) * self.loss_weight["bbox"] * self.loss_stage["second"]
        # loss_cons_second = boxseg_iouloss(pred_bbox_second, pred_mask_second) * self.loss_weight["cons"]
        # loss_cons_first = boxseg_iouloss(pred_bbox_first, pred_mask_first) * self.loss_weight["cons"]
        # loss_cons = loss_cons_first + loss_cons_second

        lan_feat = self.text_pooler(lan_feat, lan_mask)
        clip_loss_first = torch.tensor([0.0], device=device)
        clip_loss_second = torch.tensor([0.0], device=device)
        if "first_stage" in self.clip_loss_weight:
            # ! clip_loss first stage
            box_level_clip_loss_first = torch.tensor([0.0], device=device)
            seg_level_clip_loss_first = torch.tensor([0.0], device=device)
            if "box" in self.clip_loss_weight["first_stage"]:
                box_feat = self.box_cons_embedding(box_feat_first)
                box_level_clip_loss_first, _, _ = clip_infonNCEloss(box_feat, lan_feat, self.clip_loss, self.logit_scale)
                box_level_clip_loss_first = box_level_clip_loss_first * self.clip_loss_weight["first_stage"]["box"]
            if "seg" in self.clip_loss_weight["first_stage"]:
                seg_feat = self.seg_cons_embedding(torch.mean(seg_feat_pos_first, dim=1))
                seg_level_clip_loss_first, _, _ = clip_infonNCEloss(seg_feat, lan_feat, self.clip_loss, self.logit_scale)
                seg_level_clip_loss_first = seg_level_clip_loss_first * self.clip_loss_weight["first_stage"]["seg"]
            clip_loss_first = box_level_clip_loss_first + seg_level_clip_loss_first
        if "second_stage" in self.clip_loss_weight:
            # ! clip_loss second stage
            # * second stage -- box seg pooling -> box_feat(B,C) seg_feat(B,C)
            x_uptostage2size = F.interpolate(img_feat, size=pred_mask_2.shape[-2:], mode="bilinear", align_corners=True)
            box_feat_second, [seg_feat_pos_second, seg_feat_neg_second] = self.boxsegpooler(pred_bbox_2, pred_mask_2, x_uptostage2size)
            box_level_clip_loss_second = torch.tensor([0.0], device=device)
            seg_level_clip_loss_second = torch.tensor([0.0], device=device)
            if "box" in self.clip_loss_weight["second_stage"]:
                box_feat = self.box_cons_embedding(box_feat_second)
                box_level_clip_loss_second, _, _ = clip_infonNCEloss(box_feat, lan_feat, self.clip_loss, self.logit_scale)
                box_level_clip_loss_second = box_level_clip_loss_second * self.clip_loss_weight["second_stage"]["box"]
            if "seg" in self.clip_loss_weight["second_stage"]:
                seg_feat = self.seg_cons_embedding(torch.mean(seg_feat_pos_second, dim=1))
                seg_level_clip_loss_second, _, _ = clip_infonNCEloss(seg_feat, lan_feat, self.clip_loss, self.logit_scale)
                seg_level_clip_loss_second = seg_level_clip_loss_second * self.clip_loss_weight["second_stage"]["seg"]
            clip_loss_second = box_level_clip_loss_second + seg_level_clip_loss_second

        # loss_cons = (
        #     consistencyloss(box_feat, seg_feat_pos, self.cons_loss) * self.loss_weight["cons"]
        #     if "cons" in self.loss_weight
        #     else torch.tensor([0.0], device=device)
        # )

        loss_mask = loss_mask_first + loss_mask_second
        loss_det = loss_bbox_first + loss_bbox_second
        loss_cons = clip_loss_first + clip_loss_second
        loss_dict = {
            "loss_mask": loss_mask,
            "loss_det": loss_det,
            "loss_cons": loss_cons,
            "loss_mask_first": loss_mask_first,
            "loss_bbox_first": loss_bbox_first,
            "loss_mask_second": loss_mask_second,
            "loss_bbox_second": loss_bbox_second,
            "loss_cons_first": clip_loss_first,
            "loss_cons_second": clip_loss_second,
        }
        pred_dict = {"pred_mask": pred_mask_second, "pred_bbox": pred_bbox_second, "pred_mask_first": pred_mask_first, "pred_bbox_first": pred_bbox_first}
        return loss_dict, pred_dict

    def forward_test(self, x, cls_feat=None, lan_feat=None, lan_mask=None, img=None):
        # one stage
        # pred_mask = self.seg_branch(x)
        # pred_bbox = self.box_branch(cls_feat)
        # pred_bbox = self.box_branch(self.pool(x).squeeze())
        # pred_mask = F.interpolate(pred_mask, size=img.shape[-2:], mode="bilinear", align_corners=True)

        # all feats embedding to hidden_channels
        # all feats embedding to hidden_channels
        img_feat = self.img_embedding(x)
        query_feat = self.query_embedding(cls_feat)
        lan_feat = self.lan_embedding(lan_feat)
        # ! stage 1
        pred_mask = self.seg_branch_first(img_feat)
        pred_bbox = self.box_branch_first(query_feat)

        pred_bbox_first = pred_bbox

        # * bbox branch from the img avg feature
        # pred_bbox = self.box_branch(self.pool(x).squeeze())

        # * downsample target mask to the same size as pred mask
        # pred_mask_first = pred_mask
        # target_mask_first = F.interpolate(target_mask.float().unsqueeze(1), size=pred_mask.shape[-2:], mode="bilinear", align_corners=True).squeeze(1).to(torch.uint8)

        # * upsample pred_mask to the same size as target_mask
        pred_mask_first = F.interpolate(pred_mask, size=img.shape[-2:], mode="bilinear", align_corners=True)

        # * first stage -- box seg pooling -> box_feat(B,C) seg_feat(B,C)
        box_feat_first, [seg_feat_pos_first, seg_feat_neg_first] = self.boxsegpooler(pred_bbox, pred_mask, img_feat)

        # ! stage 2
        pred_bbox_2_tmp, pred_mask_2_tmp = self.boxsegattn(box_feat_first, seg_feat_pos_first, img_feat, lan_feat, lan_mask)
        pred_mask_2 = self.seg_branch_second(pred_mask_2_tmp)
        pred_bbox_2 = self.box_branch_second(pred_bbox_2_tmp)

        pred_mask_second = F.interpolate(pred_mask_2, size=img.shape[-2:], mode="bilinear", align_corners=True)
        pred_bbox_second = pred_bbox_2

        # ! pixel level cons
        x_c1, x_c2, x_c3, x_c4 = self.fpn(pred_mask_2_tmp)
        pred_mask_2_up4 = self.fpn_decoder(x_c4, x_c3, x_c2, x_c1)
        pixel_level_cons_mask = self.proj_pixel_level_cons(pred_mask_2_up4, self.text_pooler(lan_feat, lan_mask))
        pixel_level_cons_pred_mask = F.interpolate(pixel_level_cons_mask, size=img.shape[-2:], mode="bilinear", align_corners=True)

        pred_mask_second = pred_mask_second + pixel_level_cons_pred_mask + pred_mask_first

        pred_dict = {"pred_mask": pred_mask_second, "pred_bbox": pred_bbox_second, "pred_mask_first": pred_mask_first, "pred_bbox_first": pred_bbox_first}
        return pred_dict


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
