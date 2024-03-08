import torch
import numpy
from seqtr.models import MODELS
from mmdet.core import BitmapMasks
import pycocotools.mask as maskUtils
from .one_stage import OneStageModel
import torch.nn.functional as F
from torch import nn
from seqtr.models.losses.boxloss import BoxLoss
from mmcv.runner import BaseModule, auto_fp16


@MODELS.register_module()
class VGTR(OneStageModel):

    def __init__(self, word_emb, num_token, vis_enc, lan_enc, head, fusion):
        super(VGTR, self).__init__(
            word_emb, num_token, vis_enc, lan_enc, head, fusion=None
        )

        self.prediction_head = nn.Sequential(
            nn.Linear(head["hidden_dim"] * 4, head["hidden_dim"]),
            nn.BatchNorm1d(head["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(head["hidden_dim"], head["hidden_dim"]),
            nn.BatchNorm1d(head["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(head["hidden_dim"], 4),
        )

        self.loss = BoxLoss()

    def forward_train(
        self,
        img,
        ref_expr_inds,
        img_metas,
        gt_bbox=None,
        gt_mask_vertices=None,
        mass_center=None,
        rescale=False,
    ):

        img_feat, text_feat = self.extract_visual_language(
            img, img_metas, ref_expr_inds
        )
        img_feat = img_feat[-1] if isinstance(img_feat, list) else img_feat
        embed = self.head(img_feat, text_feat, ref_expr_inds)

        embed2 = torch.cat([embed[:, i] for i in range(4)], dim=-1)

        pred = self.prediction_head(embed2).sigmoid()

        gt_bbox = torch.stack(gt_bbox, dim=0)
        norm_bbox = torch.zeros_like(gt_bbox, device=gt_bbox.device)

        norm_bbox[:, 0] = (gt_bbox[:, 0] + gt_bbox[:, 2]) / 2.0  # x_center
        norm_bbox[:, 1] = (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2.0  # y_center
        norm_bbox[:, 2] = gt_bbox[:, 2] - gt_bbox[:, 0]  # w
        norm_bbox[:, 3] = gt_bbox[:, 3] - gt_bbox[:, 1]  # h

        img_size = torch.tensor(img.shape[-2:], device=img.device)
        norm_bbox = norm_bbox / (img_size.unsqueeze(0).repeat((img.shape[0], 2)))
        loss, loss_box, loss_giou = self.loss(pred, norm_bbox)
        losses_dict = {"loss_det": loss, "loss_box": loss_box, "loss_giou": loss_giou}

        with torch.no_grad():
            predictions = self.get_predictions(pred, img_metas, img_size)

        return losses_dict, predictions

    def extract_visual_language(self, img, img_metas, ref_expr_inds):
        y = self.lan_enc(ref_expr_inds)
        x = self.vis_enc(img, y)
        return x, y

    @torch.no_grad()
    def forward_test(
        self,
        img,
        ref_expr_inds,
        img_metas,
        with_bbox=False,
        with_mask=False,
        rescale=False,
    ):
        img_feat, text_feat = self.extract_visual_language(
            img, img_metas, ref_expr_inds
        )

        img_feat = img_feat[-1] if isinstance(img_feat, list) else img_feat
        embed = self.head(img_feat, text_feat, ref_expr_inds)

        embed2 = torch.cat([embed[:, i] for i in range(4)], dim=-1)

        pred = self.prediction_head(embed2).sigmoid()

        img_size = torch.tensor(img.shape[-2:], device=img.device)

        predictions = self.get_predictions(pred, img_metas, img_size=img_size)

        return predictions

    def get_predictions(self, pred, img_metas, img_size):
        pred_masks = None
        pred_bboxes = pred * img_size.unsqueeze(0).repeat((pred.shape[0], 2))
        
        output_bbox = torch.zeros_like(pred_bboxes, device=pred_bboxes.device)
        output_bbox[:, 0] = pred_bboxes[:, 0] - pred_bboxes[:, 2] / 2.0  # x1
        output_bbox[:, 1] = pred_bboxes[:, 1] - pred_bboxes[:, 3] / 2.0  # y1
        output_bbox[:, 2] = pred_bboxes[:, 0] + pred_bboxes[:, 2] / 2.0  # x2
        output_bbox[:, 3] = pred_bboxes[:, 1] + pred_bboxes[:, 3] / 2.0 # y2
        
        return dict(pred_bboxes=output_bbox, pred_masks=pred_masks)
