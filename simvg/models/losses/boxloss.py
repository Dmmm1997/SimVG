# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class BoxLoss(nn.Module):
    def __init__(self, loss_weight = [3,1]):
        super(BoxLoss, self).__init__()
        self.loss_weight = loss_weight
        self.MSELoss = torch.nn.MSELoss(reduction='none')
        
    def forward(self, pred, gt):
        """`
        :param pred:  (bs, 4)
        :param gt: (bs, 4)
        :return:
        """
        bs = pred.shape[0]

        loss_bbox = F.l1_loss(pred, gt, reduction='none')
        loss_bbox = loss_bbox.sum() / bs

        loss_giou = 1 - torch.diag(self.generalized_box_iou(
                                   self.box_cxcywh_to_xyxy(pred),
                                   self.box_cxcywh_to_xyxy(gt)))

        loss_giou = loss_giou.sum() / bs
        loss = 5 * loss_bbox + loss_giou * 2
        return loss, 5 * loss_bbox, loss_giou * 2


    def box_loss(self, pred_box, gt_box, type='L2'):
        """
        :param pred_box: (bs, 4)    (center_x, center_y, h, w) not normalized for L2 loss
        :param gt_box:              (center_x, center_y, h, w) normalized for L1 loss
        :return:
        """
        # loss_box = torch.tensor(0.).cuda()
        if type == 'L1':
            loss_bbox = F.l1_loss(pred_box, gt_box, reduction='none')  # element-wise L1 loss
        elif type == 'L2':
            loss_bbox = self.MSELoss(pred_box, gt_box)
        else:
            raise NotImplementedError('Not Implemented Loss type')
        loss = loss_bbox.sum() / pred_box.shape[0]
        return loss

    def diou_loss(self, preds, bbox, eps=1e-7, reduction='mean'):
        '''
        https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/loss/multibox_loss.py
        :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
        :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
        :param eps: eps to avoid divide 0
        :param reduction: mean or sum
        :return: diou-loss
        '''
        ix1 = torch.max(preds[:, 0], bbox[:, 0])
        iy1 = torch.max(preds[:, 1], bbox[:, 1])
        ix2 = torch.min(preds[:, 2], bbox[:, 2])
        iy2 = torch.min(preds[:, 3], bbox[:, 3])

        iw = (ix2 - ix1 + 1.0).clamp(min=0.)
        ih = (iy2 - iy1 + 1.0).clamp(min=0.)

        # overlaps
        inters = iw * ih

        # union
        uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (
                    bbox[:, 2] - bbox[:, 0] + 1.0) * (
                      bbox[:, 3] - bbox[:, 1] + 1.0) - inters

        # iou
        iou = inters / (uni + eps)

        # inter_diag
        cxpreds = (preds[:, 2] + preds[:, 0]) / 2
        cypreds = (preds[:, 3] + preds[:, 1]) / 2

        cxbbox = (bbox[:, 2] + bbox[:, 0]) / 2
        cybbox = (bbox[:, 3] + bbox[:, 1]) / 2

        inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2

        # outer_diag
        ox1 = torch.min(preds[:, 0], bbox[:, 0])
        oy1 = torch.min(preds[:, 1], bbox[:, 1])
        ox2 = torch.max(preds[:, 2], bbox[:, 2])
        oy2 = torch.max(preds[:, 3], bbox[:, 3])

        outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2

        diou = iou - inter_diag / outer_diag
        diou = torch.clamp(diou, min=-1.0, max=1.0)

        diou_loss = 1 - diou

        if reduction == 'mean':
            loss = torch.mean(diou_loss)
        elif reduction == 'sum':
            loss = torch.sum(diou_loss)
        else:
            raise NotImplementedError
        return loss

    def ciou_loss(self, preds, bbox, eps=1e-7, reduction='mean'):
        '''
        https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/loss/multibox_loss.py
        :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
        :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
        :param eps: eps to avoid divide 0
        :param reduction: mean or sum
        :return: diou-loss
        '''
        ix1 = torch.max(preds[:, 0], bbox[:, 0])
        iy1 = torch.max(preds[:, 1], bbox[:, 1])
        ix2 = torch.min(preds[:, 2], bbox[:, 2])
        iy2 = torch.min(preds[:, 3], bbox[:, 3])

        iw = (ix2 - ix1 + 1.0).clamp(min=0.)
        ih = (iy2 - iy1 + 1.0).clamp(min=0.)

        # overlaps
        inters = iw * ih

        # union
        uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (
                    bbox[:, 2] - bbox[:, 0] + 1.0) * (
                      bbox[:, 3] - bbox[:, 1] + 1.0) - inters

        # iou
        iou = inters / (uni + eps)

        # inter_diag
        cxpreds = (preds[:, 2] + preds[:, 0]) / 2
        cypreds = (preds[:, 3] + preds[:, 1]) / 2

        cxbbox = (bbox[:, 2] + bbox[:, 0]) / 2
        cybbox = (bbox[:, 3] + bbox[:, 1]) / 2

        inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2

        # outer_diag
        ox1 = torch.min(preds[:, 0], bbox[:, 0])
        oy1 = torch.min(preds[:, 1], bbox[:, 1])
        ox2 = torch.max(preds[:, 2], bbox[:, 2])
        oy2 = torch.max(preds[:, 3], bbox[:, 3])

        outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2

        diou = iou - inter_diag / outer_diag

        # calculate v,alpha
        wbbox = bbox[:, 2] - bbox[:, 0] + 1.0
        hbbox = bbox[:, 3] - bbox[:, 1] + 1.0
        wpreds = preds[:, 2] - preds[:, 0] + 1.0
        hpreds = preds[:, 3] - preds[:, 1] + 1.0
        v = torch.pow((torch.atan(wbbox / hbbox) - torch.atan(wpreds / hpreds)), 2) * (4 / (math.pi ** 2))
        alpha = v / (1 - iou + v)
        ciou = diou - alpha * v
        ciou = torch.clamp(ciou, min=-1.0, max=1.0)

        ciou_loss = 1 - ciou
        if reduction == 'mean':
            loss = torch.mean(ciou_loss)
        elif reduction == 'sum':
            loss = torch.sum(ciou_loss)
        else:
            raise NotImplementedError
        return loss

    def generalized_box_iou(self, boxes1, boxes2):
        """
        Generalized IoU from https://giou.stanford.edu/

        The boxes should be in [x0, y0, x1, y1] format

        Returns a [N, M] pairwise matrix, where N = len(boxes1)
        and M = len(boxes2)
        """
        # degenerate boxes gives inf / nan results
        # so do an early check
        assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
        iou, union = self.box_iou(boxes1, boxes2)

        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        area = wh[:, :, 0] * wh[:, :, 1]

        return iou - (area - union) / area

    def box_iou(self, boxes1, boxes2):
        area1 = self.box_area(boxes1)
        area2 = self.box_area(boxes2)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        iou = inter / union
        return iou, union

    def box_area(self, boxes):
        """
        Computes the area of a set of bounding boxes, which are specified by its
        (x1, y1, x2, y2) coordinates.

        Arguments:
            boxes (Tensor[N, 4]): boxes for which the area will be computed. They
                are expected to be in (x1, y1, x2, y2) format

        Returns:
            area (Tensor[N]): area for each box
        """
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    def giou_loss(self, pred_box, gt_box):

        loss_giou = 1 - torch.diag(self.generalized_box_iou(
            self.box_cxcywh_to_xyxy(pred_box), self.box_cxcywh_to_xyxy(gt_box)))

        return loss_giou.sum() / pred_box.shape[0]

    def focal_loss(self, pred, gt, down_sample=32):
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
          Arguments:
            pred (batch x c x h x w)  [batch_size, c, h, w]
            gt: [batch_size, ]
        '''
        pred = pred[:, 0, :, :].unsqueeze(1)
        gt, down_sample_center = self.gaussian_smooth(pred, gt, down_sample=down_sample)
        gt = gt.unsqueeze(1)
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = torch.tensor(0.).cuda()

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss, down_sample_center
