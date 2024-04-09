import time
import torch
import numpy

import pycocotools.mask as maskUtils
from seqtr.datasets import extract_data
from seqtr.utils import get_root_logger, reduce_mean, is_main
from torchvision.ops.boxes import box_area
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
from collections import defaultdict


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
    if eval_mask:
        mask_iou = mask_overlaps(gt_mask, pred_masks, is_crowd).to(device)
        for i, iou_thr in enumerate([0.5, 0.6, 0.7, 0.8, 0.9]):
            mask_acc_at_thrs[i] = (mask_iou >= iou_thr).float().mean()

    return det_acc * 100.0, mask_iou * 100.0, mask_acc_at_thrs * 100.0


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
    det_acc_list, mask_iou_list, mask_acc_list, f1_score_list, n_acc_list = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
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

            if not isinstance(predictions, list):
                predictions_list = [predictions]
            else:
                predictions_list = predictions

            # statistics informations
            map_dict = {0: "decoder", 1: "token"}
            det_acc_dict, f1_score_acc_dict, n_acc_dict = {}, {}, {}
            for ind, predictions in enumerate(predictions_list):
                predict_type = map_dict[ind]
                pred_bboxes = predictions.pop("pred_bboxes")
                pred_masks = predictions.pop("pred_masks")
                if not cfg["dataset"] == "GRefCOCO":
                    with torch.no_grad():
                        batch_det_acc, batch_mask_iou, batch_mask_acc_at_thrs = accuracy(
                            pred_bboxes,
                            gt_bbox,
                            pred_masks,
                            gt_mask,
                            is_crowd=is_crowd,
                            device=device,
                        )
                        if cfg.distributed:
                            batch_det_acc = reduce_mean(batch_det_acc)
                            # batch_mask_iou = reduce_mean(batch_mask_iou)
                            # batch_mask_acc_at_thrs = reduce_mean(batch_mask_acc_at_thrs)
                    det_acc_list[predict_type].append(batch_det_acc.item())
                    det_acc = sum(det_acc_list[predict_type]) / len(det_acc_list[predict_type])
                    det_acc_dict[predict_type] = det_acc
                else:
                    targets = [meta["target"] for meta in img_metas]
                    with torch.no_grad():
                        batch_f1_score, batch_n_acc = grec_evaluate_f1_nacc(pred_bboxes, gt_bbox, targets, device=device)
                        if cfg.distributed:
                            batch_f1_score = reduce_mean(batch_f1_score)
                            batch_n_acc = reduce_mean(batch_n_acc)
                    f1_score_list[predict_type].append(batch_f1_score.item())
                    n_acc_list[predict_type].append(batch_n_acc.item())
                    f1_score_acc = sum(f1_score_list[predict_type]) / len(f1_score_list[predict_type])
                    n_acc = sum(n_acc_list[predict_type]) / len(n_acc_list[predict_type])
                    f1_score_acc_dict[predict_type] = f1_score_acc
                    n_acc_dict[predict_type] = n_acc

            # logging informations
            if is_main() and ((batch + 1) % cfg.log_interval == 0 or batch + 1 == batches):
                logger = get_root_logger()

                if not cfg["dataset"] == "GRefCOCO":
                    ACC_str_list = [
                        "{}Det@.5: {:.2f}, ".format(map_dict[i], det_acc_dict[map_dict[i]]) for i in range(len(predictions_list))
                    ]
                    ACC_str = "".join(ACC_str_list)
                    logger.info(f"val - epoch [{epoch+1}]-[{batch+1}/{batches}] " + f"time: {(time.time()- end):.2f}, " + ACC_str)
                    
                else:
                    F1_Score_str_list = [
                        "{}_f1_score: {:.2f}, ".format(map_dict[i], f1_score_acc_dict[map_dict[i]]) for i in range(len(predictions_list))
                    ]
                    n_acc_str_list = [
                        "{}_n_acc: {:.2f}, ".format(map_dict[i], n_acc_dict[map_dict[i]]) for i in range(len(predictions_list))
                    ]
                    F1_Score_str = "".join(F1_Score_str_list)
                    n_acc_str = "".join(n_acc_str_list)
                    logger.info(
                        f"Validate - epoch [{epoch+1}]-[{batch+1}/{batches}] "
                        + f"time: {(time.time()- end):.2f}, "
                        + F1_Score_str
                        + n_acc_str
                    )
            

            end = time.time()
    
    if not cfg["dataset"] == "GRefCOCO":
        det_acc = sum(list(det_acc_dict.values())) / len(det_acc_dict)
        mask_iou = 0
    else:
        det_acc = sum(list(f1_score_acc_dict.values())) / len(f1_score_acc_dict)
        mask_iou = sum(list(n_acc_dict.values())) / len(n_acc_dict)
        

    return det_acc, mask_iou
