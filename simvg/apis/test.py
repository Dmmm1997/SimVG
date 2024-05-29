import time
import torch
import numpy

import pycocotools.mask as maskUtils
from simvg.datasets import extract_data
from simvg.utils import get_root_logger, reduce_mean, is_main
from torchvision.ops.boxes import box_area
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
from collections import defaultdict
from mmdet.core import BitmapMasks
import numpy as np


def mask_overlaps(gt_mask, pred_masks, is_crowd):
    """Args:
    gt_mask (list[RLE]):
    pred_mask (list[RLE]):
    """

    def computeIoU_RLE(gt_mask, pred_masks, is_crowd):
        mask_iou = maskUtils.iou(pred_masks, gt_mask, is_crowd)
        mask_iou = numpy.diag(mask_iou)
        return mask_iou

    mask_iou = computeIoU_RLE(gt_mask, pred_masks, is_crowd)
    mask_iou = torch.from_numpy(mask_iou)

    return mask_iou


def mask_overlaps_withIU_RLE(gt_masks, pred_masks, is_crowds):
    # decode the mask
    pred_mask = torch.concat([torch.from_numpy(maskUtils.decode(pred_rle)[None]) for pred_rle in pred_masks], dim=0)
    gt_mask = torch.concat([torch.from_numpy(maskUtils.decode(pred_rle)[None]) for pred_rle in gt_masks], dim=0)
    # pred_mask = pred_mask.argmax(1)
    intersection = torch.sum(torch.mul(pred_mask, gt_mask).reshape(pred_mask.shape[0], -1), dim=-1)
    union = torch.stack(
        [
            pred_mask_.sum() if is_crowd else (pred_mask_ + gt_mask_).sum() - inters
            for pred_mask_, gt_mask_, is_crowd, inters in zip(pred_mask, gt_mask, is_crowds, intersection)
        ],
        dim=0,
    )
    intersection = intersection.cuda()
    union = union.cuda()

    # union = torch.sum(torch.add(pred_mask, gt_mask).reshape(pred_mask.shape[0], -1), dim=1).cuda() - intersection
    iou = torch.tensor([i / u if u >= 1 else 0 for i, u in zip(intersection, union)]).cuda()
    return iou, intersection, union


def mask_overlaps_withIU(gt_masks, pred_masks, is_crowd):
    # decode the mask
    pred_mask = torch.concat([torch.from_numpy(maskUtils.decode(pred_rle)[None]) for pred_rle in pred_masks], dim=0)
    gt_mask = torch.concat([torch.from_numpy(maskUtils.decode(gt_rle)[None]) for gt_rle in gt_masks], dim=0)
    # pred_mask = pred_mask.argmax(1)
    intersection = torch.sum(torch.mul(pred_mask, gt_mask).reshape(pred_mask.shape[0], -1), dim=-1).cuda()
    union = torch.sum(torch.add(pred_mask, gt_mask).reshape(pred_mask.shape[0], -1), dim=1).cuda() - intersection
    iou = torch.tensor([i / u if u >= 1 else 0 for i, u in zip(intersection, union)]).cuda()
    return iou, intersection, union


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
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
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def accuracy(pred_bboxes, gt_bbox, pred_masks, gt_mask, is_crowd=None, device="cuda:0"):
    eval_det = pred_bboxes is not None
    eval_mask = pred_masks is not None

    det_acc = torch.tensor([0.0], device=device)
    bbox_iou = torch.tensor([0.0], device=device)
    if eval_det:
        gt_bbox = torch.stack(gt_bbox).to(device)
        bbox_iou = bbox_overlaps(gt_bbox, pred_bboxes, is_aligned=True)
        det_acc = (bbox_iou >= 0.5).float().mean()

    mask_iou = torch.tensor([0.0], device=device)
    mask_acc_at_thrs = torch.full((5,), -1.0, device=device)
    I, U = torch.tensor([0.0], device=device), torch.tensor([0.0], device=device)
    if eval_mask:
        # mask_iou = mask_overlaps(gt_mask, pred_masks, is_crowd).to(device)
        mask_iou, I, U = mask_overlaps_withIU(gt_mask, pred_masks, is_crowd)
        for i, iou_thr in enumerate([0.5, 0.6, 0.7, 0.8, 0.9]):
            mask_acc_at_thrs[i] = (mask_iou >= iou_thr).float().mean()

    return det_acc * 100.0, mask_iou * 100.0, mask_acc_at_thrs * 100.0, I*1.0, U*1.0


def grec_evaluate_f1_nacc(predictions, gt_bboxes, targets, thresh_score=0.7, thresh_iou=0.5, thresh_F1=1.0, device="cuda:0"):
    correct_image = torch.tensor(0, device=device)
    num_image = torch.tensor(0, device=device)
    nt = {
        "TP": torch.tensor(0.0, device=device),
        "TN": torch.tensor(0.0, device=device),
        "FP": torch.tensor(0.0, device=device),
        "FN": torch.tensor(0.0, device=device),
    }
    if predictions is None:
        return torch.tensor(0.0, device=device).float(), torch.tensor(0.0, device=device).float()
    for prediction, gt_bbox, target in zip(predictions, gt_bboxes, targets):
        TP = 0
        assert prediction is not None
        sorted_scores_boxes = sorted(zip(prediction["scores"].tolist(), prediction["boxes"].tolist()), reverse=True)
        sorted_scores, sorted_boxes = zip(*sorted_scores_boxes)
        sorted_boxes = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes])
        converted_bbox_all = []
        no_target_flag = False
        for converted_bbox, one_target in zip(gt_bbox, target):
            if one_target["category_id"] == -1:
                no_target_flag = True
            # target_bbox = one_target["bbox"]
            # converted_bbox = [
            #     target_bbox[0],
            #     target_bbox[1],
            #     target_bbox[2] + target_bbox[0],
            #     target_bbox[3] + target_bbox[1],
            # ]
            converted_bbox_all.append(converted_bbox)
        gt_bbox_all = torch.stack(converted_bbox_all, dim=0)

        sorted_scores_array = numpy.array(sorted_scores)
        idx = sorted_scores_array >= thresh_score
        filtered_boxes = sorted_boxes[idx]
        # filtered_boxes = sorted_boxes[0:1]
        giou = generalized_box_iou(filtered_boxes, gt_bbox_all.view(-1, 4))
        num_prediction = filtered_boxes.shape[0]
        num_gt = gt_bbox_all.shape[0]
        if no_target_flag:
            if num_prediction >= 1:
                nt["FN"] += 1
                F_1 = torch.tensor(0.0, device=device)
            else:
                nt["TP"] += 1
                F_1 = torch.tensor(1.0, device=device)
        else:
            if num_prediction >= 1:
                nt["TN"] += 1
            else:
                nt["FP"] += 1
            for i in range(min(num_prediction, num_gt)):
                top_value, top_index = torch.topk(giou.flatten(0, 1), 1)
                if top_value < thresh_iou:
                    break
                else:
                    top_index_x = top_index // num_gt
                    top_index_y = top_index % num_gt
                    TP += 1
                    giou[top_index_x[0], :] = 0.0
                    giou[:, top_index_y[0]] = 0.0
            FP = num_prediction - TP
            FN = num_gt - TP
            F_1 = 2 * TP / (2 * TP + FP + FN)

        if F_1 >= thresh_F1:
            correct_image += 1
        num_image += 1

    F1_score = correct_image / num_image
    # T_acc = nt["TN"] / (nt["TN"] + nt["FP"])
    N_acc = nt["TP"] / (nt["TP"] + nt["FN"]) if nt["TP"] != 0 else torch.tensor(0.0, device=device)
    return F1_score.float() * 100, N_acc.float() * 100


def evaluate_model(epoch, cfg, model, loader):
    model.eval()

    device = list(model.parameters())[0].device

    batches = len(loader)
    end = time.time()

    with_bbox, with_mask = False, False
    det_acc_list, mask_iou_list, mask_acc_list, mask_I_list, mask_U_list = [], [], [], [], []
    with torch.no_grad():
        for batch, inputs in enumerate(loader):
            gt_bbox, gt_mask, is_crowd = None, None, None

            if "gt_bbox" in inputs:
                with_bbox = True
                if isinstance(inputs["gt_bbox"], torch.Tensor):
                    inputs["gt_bbox"] = [inputs["gt_bbox"][ind] for ind in range(inputs["gt_bbox"].shape[0])]
                    gt_bbox = inputs.pop("gt_bbox")
                else:
                    gt_bbox = inputs.pop("gt_bbox").data[0]
            if "gt_mask_rle" in inputs:
                with_mask = True
                gt_mask = inputs.pop("gt_mask_rle").data[0]
            if "is_crowd" in inputs:
                is_crowd = inputs.pop("is_crowd").data[0]

            img_metas = inputs["img_metas"].data[0]

            if not cfg.distributed:
                inputs = extract_data(inputs)

            predictions = model(
                **inputs,
                return_loss=False,
                rescale=False,
                with_bbox=with_bbox,
                with_mask=with_mask,
            )

            pred_bboxes = predictions.pop("pred_bboxes")
            pred_masks = predictions.pop("pred_masks")

            batch_det_acc, batch_mask_iou, batch_mask_acc_at_thrs, batch_mask_I, batch_mask_U = accuracy(
                pred_bboxes, gt_bbox, pred_masks, gt_mask, is_crowd=is_crowd, device=device
            )
            if cfg.distributed:
                batch_det_acc = reduce_mean(batch_det_acc)
                batch_mask_iou = reduce_mean(batch_mask_iou)
                batch_mask_I = reduce_mean(batch_mask_I)
                batch_mask_U = reduce_mean(batch_mask_U)
                batch_mask_acc_at_thrs = reduce_mean(batch_mask_acc_at_thrs)

            det_acc_list.append(batch_det_acc.item())
            mask_iou_list.append(batch_mask_iou)
            mask_I_list.append(batch_mask_I)
            mask_U_list.append(batch_mask_U)
            mask_acc_list.append(batch_mask_acc_at_thrs)

            det_acc = sum(det_acc_list) / len(det_acc_list)
            mask_miou = torch.cat(mask_iou_list).mean().item()
            mask_I = torch.cat(mask_I_list).mean().item()
            mask_U = torch.cat(mask_U_list).mean().item()
            mask_oiou = 100.0 * mask_I / mask_U
            mask_acc = torch.vstack(mask_acc_list).mean(dim=0).tolist()
            if is_main():
                if (batch + 1) % cfg.log_interval == 0 or batch + 1 == batches:
                    logger = get_root_logger()
                    logger.info(
                        f"validate - epoch [{epoch+1}]-[{batch+1}/{batches}] "
                        + f"time: {(time.time() - end):.2f}, "
                        + f"DetACC@0.5: {det_acc:.2f}, "
                        + f"mIoU: {mask_miou:.2f}, "
                        + f"oIoU: {mask_oiou:.2f},"
                        + f"MaskACC@0.5-0.9: [{mask_acc[0]:.2f}, {mask_acc[1]:.2f}, {mask_acc[2]:.2f},  {mask_acc[3]:.2f},  {mask_acc[4]:.2f}]"
                    )

            end = time.time()

    return det_acc, mask_miou, mask_oiou
