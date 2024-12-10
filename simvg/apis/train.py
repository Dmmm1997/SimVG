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
import wandb

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

    det_acc_list, det_accs_list, mask_iou_list, mask_acc_list, mask_I_list, mask_U_list = [], [], [], [], [], []
    det_acc_list_fs, mask_iou_list_fs, mask_acc_list_fs, mask_I_list_fs, mask_U_list_fs = [], [], [], [], []
    loss_det_list, loss_mask_list, loss_cons_list, loss_clip_list = [], [], [], []
    # loss_cons_fs_list, loss_cons_ss_list  = [], []
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

        losses, predictions = model(**inputs, gt_mask=gt_mask, epoch=epoch, rescale=False)

        # loss_multitask = losses.pop("loss_multi_task", torch.tensor([0.0], device=device))
        loss_det = losses.pop("loss_det", torch.tensor([0.0], device=device)) + losses.pop("loss_multi_task", torch.tensor([0.0], device=device))
        loss_mask = losses.pop("loss_mask", torch.tensor([0.0], device=device))
        loss_cons = losses.pop("loss_cons", torch.tensor([0.0], device=device))
        loss_clip = losses.pop("loss_clip", torch.tensor([0.0], device=device))
        loss = loss_det + loss_mask + loss_cons + loss_clip
        # loss = loss_det
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

        loss_cons_first = losses.pop("loss_cons_first", torch.tensor([0.0], device=device))
        loss_cons_second = losses.pop("loss_cons_second", torch.tensor([0.0], device=device))
        if cfg.distributed:
            loss_det = reduce_mean(loss_det)
            loss_mask = reduce_mean(loss_mask)
            loss_cons = reduce_mean(loss_cons)
            loss_clip = reduce_mean(loss_clip)
            loss_cons_first = reduce_mean(loss_cons_first)
            loss_cons_second = reduce_mean(loss_cons_second)

        pred_bboxes = predictions.pop("pred_bboxes")
        pred_masks = predictions.pop("pred_masks")
        pred_bboxes_firststage = predictions.pop("pred_bboxes_first", None)
        pred_masks_firststage = predictions.pop("pred_masks_first", None)

        with torch.no_grad():
            batch_det_acc, batch_mask_iou, batch_mask_acc_at_thrs, batch_mask_I, batch_mask_U, batch_det_acc_at_thrs = accuracy(
                pred_bboxes, gt_bbox, pred_masks, gt_mask, is_crowd=is_crowd, device=device
            )
            if pred_bboxes_firststage is not None and len(pred_bboxes_firststage) > 0:
                batch_det_acc_fs, batch_mask_iou_fs, batch_mask_acc_at_thrs_fs, batch_mask_I_fs, batch_mask_U_fs, batch_det_acc_at_thrs_fs = accuracy(
                    pred_bboxes_firststage, gt_bbox, pred_masks_firststage, gt_mask, is_crowd=is_crowd, device=device
                )
            if cfg.distributed:
                batch_det_acc = reduce_mean(batch_det_acc)
                batch_mask_iou = reduce_mean(batch_mask_iou)
                batch_mask_I = reduce_mean(batch_mask_I)
                batch_mask_U = reduce_mean(batch_mask_U)
                batch_mask_acc_at_thrs = reduce_mean(batch_mask_acc_at_thrs)
                batch_det_acc_at_thrs = reduce_mean(batch_det_acc_at_thrs)
                if pred_bboxes_firststage is not None and len(pred_bboxes_firststage) > 0:
                    batch_det_acc_fs = reduce_mean(batch_det_acc_fs)
                    batch_mask_iou_fs = reduce_mean(batch_mask_iou_fs)
                    batch_mask_I_fs = reduce_mean(batch_mask_I_fs)
                    batch_mask_U_fs = reduce_mean(batch_mask_U_fs)
                    batch_mask_acc_at_thrs_fs = reduce_mean(batch_mask_acc_at_thrs_fs)

        loss_det_list.append(loss_det.item())
        loss_mask_list.append(loss_mask.item())
        loss_cons_list.append(loss_cons.item())
        loss_clip_list.append(loss_clip.item())
        # loss_cons_fs_list.append(loss_cons_first.item())
        # loss_cons_ss_list.append(loss_cons_second.item())

        det_acc_list.append(batch_det_acc.item())
        mask_iou_list.append(batch_mask_iou)
        mask_acc_list.append(batch_mask_acc_at_thrs)
        det_accs_list.append(batch_det_acc_at_thrs)
        mask_I_list.append(batch_mask_I)
        mask_U_list.append(batch_mask_U)
        det_acc = sum(det_acc_list) / len(det_acc_list)
        mask_miou = torch.cat(mask_iou_list).mean().item()
        mask_I = torch.cat(mask_I_list).mean().item()
        mask_U = torch.cat(mask_U_list).mean().item()
        mask_oiou = 100.0 * mask_I / mask_U
        mask_acc = torch.vstack(mask_acc_list).mean(dim=0).tolist()
        det_accs = torch.vstack(det_accs_list).mean(dim=0).tolist()

        det_acc_fs, mask_miou_fs, mask_oiou_fs, mask_acc_fs = 0, 0, 0, [0, 0, 0, 0, 0]
        if pred_bboxes_firststage is not None and len(pred_bboxes_firststage) > 0:
            det_acc_list_fs.append(batch_det_acc_fs.item())
            mask_iou_list_fs.append(batch_mask_iou_fs)
            mask_acc_list_fs.append(batch_mask_acc_at_thrs_fs)
            mask_I_list_fs.append(batch_mask_I_fs)
            mask_U_list_fs.append(batch_mask_U_fs)
            det_acc_fs = sum(det_acc_list_fs) / len(det_acc_list_fs)
            mask_miou_fs = torch.cat(mask_iou_list_fs).mean().item()
            mask_I_fs = torch.cat(mask_I_list_fs).mean().item()
            mask_U_fs = torch.cat(mask_U_list_fs).mean().item()
            mask_oiou_fs = 100.0 * mask_I_fs / mask_U_fs
            mask_acc_fs = torch.vstack(mask_acc_list_fs).mean(dim=0).tolist()

        if is_main():
            if (batch + 1) % cfg.log_interval == 0 or batch + 1 == batches:
                logger = get_root_logger()
                logger.info(
                    f"train - epoch [{epoch+1}]-[{batch+1}/{batches}] "
                    + f"time: {(time.time()- end):.2f}, data_time: {data_time:.2f}, "
                    + f"loss_det: {sum(loss_det_list) / len(loss_det_list) :.4f}, "
                    + f"loss_mask: {sum(loss_mask_list) / len(loss_mask_list):.4f}, "
                    + f"loss_cons: {sum(loss_cons_list) / len(loss_cons_list):.4f}, "
                    + f"loss_clip: {sum(loss_clip_list) / len(loss_clip_list):.4f}, "
                    # + f"loss_cons_fs: {sum(loss_cons_fs_list) / len(loss_cons_fs_list):.4f}, "
                    # + f"loss_cons_ss: {sum(loss_cons_ss_list) / len(loss_cons_ss_list):.4f}, "
                    + f"lr: {optimizer.param_groups[0]['lr']:.6f}, "
                    + f"DetACC: {det_acc:.2f}, "
                    + f"mIoU: {mask_miou:.2f}, "
                    + f"oIoU: {mask_oiou:.2f}, "
                    + f"fs_DetACC: {det_acc_fs:.2f}, "
                    + f"fs_mIoU: {mask_miou_fs:.2f}, "
                    + f"fs_oIoU: {mask_oiou_fs:.2f}, "
                    # + f"MaskACC@0.5-0.9: [{mask_acc[0]:.2f}, {mask_acc[1]:.2f}, {mask_acc[2]:.2f},  {mask_acc[3]:.2f},  {mask_acc[4]:.2f}], "
                    # + f"fs_MaskACC@0.5-0.9: [{mask_acc_fs[0]:.2f}, {mask_acc_fs[1]:.2f}, {mask_acc_fs[2]:.2f},  {mask_acc_fs[3]:.2f},  {mask_acc_fs[4]:.2f}]"
                )

                wandb.log(
                    {
                        "loss_det": sum(loss_det_list) / len(loss_det_list),
                        "loss_mask": sum(loss_mask_list) / len(loss_mask_list),
                        "loss_cons": sum(loss_cons_list) / len(loss_cons_list),
                        "loss_clip": sum(loss_clip_list) / len(loss_clip_list),
                        # "loss_cons_fs": sum(loss_cons_fs_list) / len(loss_cons_fs_list),
                        # "loss_cons_ss": sum(loss_cons_ss_list) / len(loss_cons_ss_list),
                        "lr": optimizer.param_groups[0]["lr"],
                        "DetACC@0.5": det_acc,
                        "fs_DetACC@0.5": det_acc_fs,
                        "mIoU": mask_miou,
                        "fs_mIoU": mask_miou_fs,
                        "oIoU": mask_oiou,
                        "fs_oIoU": mask_oiou_fs,
                        "MaskACC@0.5": mask_acc[0],
                        "MaskACC@0.6": mask_acc[1],
                        "MaskACC@0.7": mask_acc[2],
                        "MaskACC@0.8": mask_acc[3],
                        "MaskACC@0.9": mask_acc[4],
                        "fs_MaskACC@0.5": mask_acc_fs[0],
                        "fs_MaskACC@0.6": mask_acc_fs[1],
                        "fs_MaskACC@0.7": mask_acc_fs[2],
                        "fs_MaskACC@0.8": mask_acc_fs[3],
                        "fs_MaskACC@0.9": mask_acc_fs[4],
                    }
                )

        end = time.time()
