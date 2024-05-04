import time
import copy
import numpy
import torch
import random

from simvg.apis.test import grec_evaluate_f1_nacc

from .test import accuracy
from simvg.datasets import extract_data
from simvg.utils import get_root_logger, reduce_mean, is_main
from collections import defaultdict

try:
    import apex
except:
    pass


def set_random_seed(seed, deterministic=False):
    """Args:
    seed (int): Seed to be used.
    deterministic (bool): Whether to set the deterministic option for
        CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
        to True and `torch.backends.cudnn.benchmark` to False.
        Default: False.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(epoch, cfg, model, model_ema, optimizer, loader):
    model.train()

    if cfg.distributed:
        loader.sampler.set_epoch(epoch)

    device = list(model.parameters())[0].device

    batches = len(loader)
    end = time.time()

    det_acc_list, mask_iou_list, mask_acc_list, mask_I_list, mask_U_list = [], [], [], [], []
    loss_det_list, loss_mask_list = [], []
    for batch, inputs in enumerate(loader):
        data_time = time.time() - end
        gt_bbox, gt_mask, is_crowd = None, None, None

        if "gt_bbox" in inputs:
            if isinstance(inputs["gt_bbox"], torch.Tensor):
                inputs["gt_bbox"] = [inputs["gt_bbox"][ind] for ind in range(inputs["gt_bbox"].shape[0])]
                gt_bbox = copy.deepcopy(inputs["gt_bbox"])
            else:
                gt_bbox = copy.deepcopy(inputs["gt_bbox"].data[0])

        img_metas = inputs["img_metas"].data[0]

        if "gt_mask_rle" in inputs:
            gt_mask = inputs.pop("gt_mask_rle").data[0]
        if "is_crowd" in inputs:
            is_crowd = inputs.pop("is_crowd").data[0]

        if not cfg.distributed:
            inputs = extract_data(inputs)

        losses, predictions = model(**inputs, rescale=False)

        loss_det = losses.get("loss_det", torch.tensor([0.0], device=device))
        loss_mask = losses.pop("loss_mask", torch.tensor([0.0], device=device))
        loss = loss_det + loss_mask
        optimizer.zero_grad()
        if cfg.use_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if cfg.grad_norm_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip)
        optimizer.step()

        if cfg.ema:
            model_ema.update_params()

        if cfg.distributed:
            loss_det = reduce_mean(loss_det)
            loss_mask = reduce_mean(loss_mask)

        pred_bboxes = predictions.pop("pred_bboxes")
        pred_masks = predictions.pop("pred_masks")

        with torch.no_grad():
            batch_det_acc, batch_mask_iou, batch_mask_acc_at_thrs, batch_mask_I, batch_mask_U = accuracy(pred_bboxes, gt_bbox, pred_masks, gt_mask, is_crowd=is_crowd, device=device)
            if cfg.distributed:
                batch_det_acc = reduce_mean(batch_det_acc)
                batch_mask_iou = reduce_mean(batch_mask_iou)
                batch_mask_I = reduce_mean(batch_mask_I)
                batch_mask_U = reduce_mean(batch_mask_U)
                batch_mask_acc_at_thrs = reduce_mean(batch_mask_acc_at_thrs)

        det_acc_list.append(batch_det_acc.item())
        mask_iou_list.append(batch_mask_iou)
        mask_acc_list.append(batch_mask_acc_at_thrs)
        loss_det_list.append(loss_det.item())
        loss_mask_list.append(loss_mask.item())
        mask_I_list.append(batch_mask_I)
        mask_U_list.append(batch_mask_U)

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
                    f"train - epoch [{epoch+1}]-[{batch+1}/{batches}] "
                    + f"time: {(time.time()- end):.2f}, data_time: {data_time:.2f}, "
                    + f"loss_det: {sum(loss_det_list) / len(loss_det_list) :.4f}, "
                    + f"loss_mask: {sum(loss_mask_list) / len(loss_mask_list):.4f}, "
                    + f"lr: {optimizer.param_groups[0]['lr']:.6f}, "
                    + f"DetACC@0.5: {det_acc:.2f}, "
                    + f"mIoU: {mask_miou:.2f}, "
                    + f"oIoU: {mask_oiou:.2f}, "
                    + f"MaskACC@0.5-0.9: [{mask_acc[0]:.2f}, {mask_acc[1]:.2f}, {mask_acc[2]:.2f},  {mask_acc[3]:.2f},  {mask_acc[4]:.2f}]"
                )

        end = time.time()
