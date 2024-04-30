# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.builder import build_loss
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, build_assigner, build_sampler, multi_apply, reduce_mean
from . import *


class DistillCriterion(nn.Module):
    def __init__(
        self,
        num_classes=1,
        loss_cls_distill=dict(
            type='DistillCrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0),
        loss_bbox_distill=dict(type='L1Loss', loss_weight=5.0),
        loss_iou_distill=dict(type='GIoULoss', loss_weight=2.0),
        distill_assigner=dict(
            type='DistillHungarianAssigner',
            cls_cost=dict(type='DistillCrossEntropyLossCost', weight=1.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))
    ):
      super().__init__()
      self.distill_assigner = build_assigner(distill_assigner)
      # DETR sampling=False, so use PseudoSampler
      distill_sampler_cfg = dict(type="PseudoSampler")
      self.distill_sampler = build_sampler(distill_sampler_cfg, context=self)
      self.loss_cls_distill = build_loss(loss_cls_distill)
      self.loss_bbox_distill = build_loss(loss_bbox_distill)
      self.loss_iou_distill = build_loss(loss_iou_distill)
      self.num_classes = num_classes
      if self.loss_cls_distill.use_sigmoid:
          self.cls_out_channels = num_classes
      else:
          self.cls_out_channels = num_classes + 1

    def forward_train_distill(
        self,
        all_cls_scores,
        all_bbox_preds,
        img_metas,
        teacher_bboxes,
        teacher_labels,
    ):

        losses, pos_assigned_gt_inds_list_distill = self.loss_distill(
            all_cls_scores, all_bbox_preds, img_metas, teacher_bboxes, teacher_labels, is_layer_by_layer_distill=True
        )
        return losses, pos_assigned_gt_inds_list_distill

    @force_fp32(apply_to=("all_cls_scores_list", "all_bbox_preds_list"))
    def loss_distill(
        self,
        all_cls_scores,
        all_bbox_preds,
        img_metas,
        teacher_bboxes_list,
        teacher_labels_list,
        gt_bboxes_ignore=None,
        is_layer_by_layer_distill=True,
    ):
        """ "Loss function.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.  len:batch_size
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ). len:batch_size
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, f"{self.__class__.__name__} only supports " f"for gt_bboxes_ignore setting to None."

        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        loss_dict = dict()

        # teacher distill
        if is_layer_by_layer_distill:
            all_teacher_bboxes_list = [teacher_bboxes_list[i] for i in range(num_dec_layers)]
            all_teacher_labels_list = [teacher_labels_list[i] for i in range(num_dec_layers)]
        else:
            all_teacher_bboxes_list = [teacher_bboxes_list for _ in range(num_dec_layers)]
            all_teacher_labels_list = [teacher_labels_list for _ in range(num_dec_layers)]
        losses_cls, losses_bbox, losses_iou, losses_cls_distill, losses_bbox_distill, losses_iou_distill, pos_assigned_gt_inds_list_distill = multi_apply(
            self.loss_single_distill,
            all_cls_scores,
            all_bbox_preds,
            all_teacher_bboxes_list,
            all_teacher_labels_list,
            img_metas_list,
            all_gt_bboxes_ignore_list,
        )
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_bbox"] = losses_bbox[-1]
        loss_dict["loss_iou"] = losses_iou[-1]
        loss_dict["loss_cls_distill"] = losses_cls_distill[-1]
        loss_dict["loss_bbox_distill"] = losses_bbox_distill[-1]
        loss_dict["loss_iou_distill"] = losses_iou_distill[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_cls_distill_i, loss_bbox_distill_i, loss_iou_distill_i in zip(
            losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1], losses_cls_distill[:-1], losses_bbox_distill[:-1], losses_iou_distill[:-1]
        ):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_bbox"] = loss_bbox_i
            loss_dict[f"d{num_dec_layer}.loss_iou"] = loss_iou_i
            loss_dict[f"d{num_dec_layer}.loss_cls_distill"] = loss_cls_distill_i
            loss_dict[f"d{num_dec_layer}.loss_bbox_distill"] = loss_bbox_distill_i
            loss_dict[f"d{num_dec_layer}.loss_iou_distill"] = loss_iou_distill_i
            num_dec_layer += 1

        return loss_dict, pos_assigned_gt_inds_list_distill

    def loss_single_distill(self, cls_scores, bbox_preds, teacher_bboxes_list, teacher_labels_list, img_metas, gt_bboxes_ignore_list=None):
        """ "Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        # cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list)
        # (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        # labels = torch.cat(labels_list, 0)
        # label_weights = torch.cat(label_weights_list, 0)
        # bbox_targets = torch.cat(bbox_targets_list, 0)
        # bbox_weights = torch.cat(bbox_weights_list, 0)

        # get teacher distill target
        cls_reg_targets_distill = self.get_distill_targets(
            cls_scores_list, bbox_preds_list, teacher_bboxes_list, teacher_labels_list, img_metas, gt_bboxes_ignore_list
        )
        (
            labels_list_distill,
            label_weights_list_distill,
            bbox_targets_list_distill,
            bbox_weights_list_distill,
            num_total_pos_distill,
            pos_assigned_gt_inds_list_distill,
        ) = cls_reg_targets_distill
        labels_distill = torch.cat(labels_list_distill, 0)
        label_weights_distill = torch.cat(label_weights_list_distill, 0).unsqueeze(-1)
        bbox_targets_distill = torch.cat(bbox_targets_list_distill, 0)
        bbox_weights_distill = torch.cat(bbox_weights_list_distill, 0)

        # classification loss
        # cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # # construct weighted avg_factor to match with the official DETR repo
        # cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        # if self.sync_cls_avg_factor:
        #     cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        # cls_avg_factor = max(cls_avg_factor, 1)

        # loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        loss_cls_distill = self.loss_cls_distill(cls_scores, labels_distill, label_weights_distill, avg_factor=num_total_pos_distill)

        # Compute the average number of gt boxes across all gpus, for

        # num_total_pos = loss_cls.new_tensor([num_total_pos])
        # num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
        # knowledge distill
        num_total_pos_distill = loss_cls_distill.new_tensor([num_total_pos_distill])
        num_total_pos_distill = torch.clamp(reduce_mean(num_total_pos_distill), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta["img_shape"]
            factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0).repeat(bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors

        # bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors
        # # regression IoU loss, defaultly GIoU loss
        # loss_iou = self.loss_iou(bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)
        # # regression L1 loss
        # loss_bbox = self.loss_bbox(bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)

        # distill loss: regression IoU loss, defaultly GIoU loss
        bboxes_gt_distill = bbox_cxcywh_to_xyxy(bbox_targets_distill) * factors
        loss_iou_distill = self.loss_iou_distill(bboxes, bboxes_gt_distill, bbox_weights_distill, avg_factor=num_total_pos_distill)
        # regression L1 loss
        loss_bbox_distill = self.loss_bbox_distill(bbox_preds, bbox_targets_distill, bbox_weights_distill, avg_factor=num_total_pos_distill)

        return loss_cls_distill, loss_bbox_distill, loss_iou_distill, pos_assigned_gt_inds_list_distill

    def get_distill_targets(self, cls_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, "Only supports for gt_bboxes_ignore setting to None."
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, pos_inds_list, pos_assigned_gt_inds_list) = multi_apply(
            self._get_distill_target_single, cls_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        # num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, pos_assigned_gt_inds_list)

    def _get_distill_target_single(self, cls_score, bbox_pred, gt_bboxes, gt_labels, img_meta, gt_bboxes_ignore=None):
        """ "Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            img_meta (dict): Meta information for one image.
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.distill_assigner.assign(bbox_pred, cls_score, gt_bboxes, gt_labels, img_meta, gt_bboxes_ignore)
        sampling_result = self.distill_sampler.sample(assign_result, bbox_pred, gt_bboxes)
        # pos_inds为 i 表示对应teacher的第 i-1 个query相匹配
        pos_inds = sampling_result.pos_inds
        labels = gt_bboxes.new_full(
            (num_bboxes, self.num_classes),
            self.num_classes,
        )
        # dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta["img_shape"]

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, sampling_result.pos_assigned_gt_inds)

    @force_fp32(apply_to=("all_cls_scores_list", "all_bbox_preds_list"))
    def get_bboxes(self, all_cls_scores, all_bbox_preds, enc_cls_scores, enc_bbox_preds, img_metas, rescale=False):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]["img_shape"]
            scale_factor = img_metas[img_id]["scale_factor"]
            proposals = self._get_bboxes_single(cls_score, bbox_pred, img_shape, scale_factor, rescale)
            result_list.append(proposals)
        return result_list

    @force_fp32(apply_to=("all_cls_scores_list", "all_bbox_preds_list"))
    def get_teacher_bboxes_distill(
        self, all_cls_scores, all_bbox_preds, enc_cls_scores, enc_bbox_preds, img_metas, rescale=False, is_layer_by_layer_distill=True
    ):
        if is_layer_by_layer_distill:
            det_bboxes_allstage_list = []
            cls_score_allstage_list = []
            for i in range(len(all_cls_scores)):
                cls_scores = all_cls_scores[i]
                bbox_preds = all_bbox_preds[i]

                det_bboxes_list = []
                cls_score_list = []
                for img_id in range(len(img_metas)):
                    cls_score = cls_scores[img_id]
                    bbox_pred = bbox_preds[img_id]
                    img_shape = img_metas[img_id]["img_shape"]
                    scale_factor = img_metas[img_id]["scale_factor"]
                    det_bboxes, cls_score = self._get_teacher_bboxes_distill_single(cls_score, bbox_pred, img_shape, scale_factor, rescale)
                    det_bboxes_list.append(det_bboxes)
                    cls_score_list.append(cls_score)
                det_bboxes_allstage_list.append(det_bboxes_list)
                cls_score_allstage_list.append(cls_score_list)
            return det_bboxes_allstage_list, cls_score_allstage_list
        else:
            cls_scores = all_cls_scores[-1]
            bbox_preds = all_bbox_preds[-1]

            det_bboxes_list = []
            cls_score_list = []
            for img_id in range(len(img_metas)):
                cls_score = cls_scores[img_id]
                bbox_pred = bbox_preds[img_id]
                img_shape = img_metas[img_id]["img_shape"]
                scale_factor = img_metas[img_id]["scale_factor"]
                det_bboxes, cls_score = self._get_teacher_bboxes_distill_single(cls_score, bbox_pred, img_shape, scale_factor, rescale)
                det_bboxes_list.append(det_bboxes)
                cls_score_list.append(cls_score)
            return det_bboxes_list, cls_score_list

    def _get_teacher_bboxes_distill_single(self, cls_score, bbox_pred, img_shape, scale_factor, rescale=False):

        assert len(cls_score) == len(bbox_pred)
        # max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            # scores, indexes = cls_score.view(-1).topk(max_per_img)
            # det_labels = indexes % self.num_classes
            # bbox_index = indexes // self.num_classes
            # bbox_pred = bbox_pred[bbox_index]
        # else:
        # 如果不用sigmoid，这部分代码还没进行相应的实现

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)

        return det_bboxes, cls_score

    @force_fp32(apply_to=("all_cls_scores_list", "all_bbox_preds_list"))
    def loss(self, all_cls_scores, all_bbox_preds, enc_cls_scores, enc_bbox_preds, gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore=None):
        """ "Loss function.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, f"{self.__class__.__name__} only supports " f"for gt_bboxes_ignore setting to None."

        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_gt_bboxes_list, all_gt_labels_list, img_metas_list, all_gt_bboxes_ignore_list
        )

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [torch.zeros_like(gt_labels_list[i]) for i in range(len(img_metas))]
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = self.loss_single(
                enc_cls_scores, enc_bbox_preds, gt_bboxes_list, binary_labels_list, img_metas, gt_bboxes_ignore
            )
            loss_dict["enc_loss_cls"] = enc_loss_cls
            loss_dict["enc_loss_bbox"] = enc_losses_bbox
            loss_dict["enc_loss_iou"] = enc_losses_iou

        # loss from the last decoder layer
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_bbox"] = losses_bbox[-1]
        loss_dict["loss_iou"] = losses_iou[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1]):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_bbox"] = loss_bbox_i
            loss_dict[f"d{num_dec_layer}.loss_iou"] = loss_iou_i
            num_dec_layer += 1
        return loss_dict
