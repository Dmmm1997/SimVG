import torch
import shutil
import os.path as osp
from simvg.utils import is_main
from .logger import get_root_logger
import copy

try:
    import apex
except:
    pass


def is_paral_model(model):
    from mmcv.parallel import MMDistributedDataParallel
    from torch.nn.parallel import DistributedDataParallel

    return isinstance(model, MMDistributedDataParallel) or isinstance(model, DistributedDataParallel)


def is_paral_state(state_dict):
    return list(state_dict.keys())[0].startswith("module.")


def de_parallel(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[key[7:]] = value
    return new_state_dict


def log_loaded_info(ckpt, load_file):
    logger = get_root_logger()
    log_str = f"loaded checkpoint from {load_file}\n"
    best_d_acc, best_miou = 0.0, 0.0
    if "epoch" and "lr" in ckpt:
        log_str += f"epoch: {ckpt['epoch']+1} lr: {ckpt['lr']:.6f}\n"
    if "best_d_acc" in ckpt:
        log_str += f"best det acc: {ckpt['best_d_acc']:.2f}\n"
        best_d_acc = ckpt["best_d_acc"]
    if "best_miou" in ckpt:
        log_str += f"best mIoU: {ckpt['best_miou']:.2f}\n"
        best_miou = ckpt["best_miou"]
    if "d_acc" in ckpt:
        log_str += f"loaded det acc: {ckpt['d_acc']:.2f}\n"
    if "miou" in ckpt:
        log_str += f"loaded mIoU: {ckpt['miou']:.2f}\n"
    logger.info(log_str)
    return best_d_acc, best_miou


# only for finetuning, if resume from pretraining, use load_checkpoint
def load_pretrained_checkpoint(model, model_ema=None, finetune_from=None, amp=False):
    assert model_ema is None, "We do not use EMA during finetuning."
    start_epoch, best_d_acc, best_miou = -1, 0.0, 0.0
    ckpt = torch.load(finetune_from, map_location=lambda storage, loc: storage.cuda())
    state = ckpt["state_dict"]
    if is_paral_state(state) and not is_paral_model(model):
        state = de_parallel(state)
    state_copy = copy.deepcopy(state)
    # state_copy.pop("lan_enc.embedding.weight")

    # model_seq_embed_dim = model.head.transformer.seq_positional_encoding.embedding.weight.size(
    #     0)
    # state_seq_embed_dim = state_copy["head.transformer.seq_positional_encoding.embedding.weight"].size(
    #     0)
    # # finetuning on RES since pretraining is only performed on REC
    # if model_seq_embed_dim != state_seq_embed_dim:
    #     state_copy.pop("head.transformer.seq_positional_encoding.embedding.weight")
    missing_keys, unexpected_keys = model.load_state_dict(state_copy, strict=False)
    if is_main():
        logger = get_root_logger()
        logger.info("missing keys:{}".format(missing_keys))
        logger.info("unexpected keys:{}".format(unexpected_keys))
    if "amp" in ckpt and amp:
        apex.amp.load_state_dict(ckpt["amp"])
    if is_main():
        best_d_acc, best_miou = log_loaded_info(ckpt, finetune_from)
    return start_epoch, best_d_acc, best_miou


def load_checkpoint(model, model_ema=None, resume_from=None, load_from=None, amp=False, optimizer=None, scheduler=None):
    start_epoch, best_d_acc, best_miou = -1, 0.0, 0.0
    flag = True
    assert not (resume_from is not None and load_from is not None)
    load_file = resume_from or load_from
    ckpt = torch.load(load_file, map_location=lambda storage, loc: storage.cuda())
    state = ckpt["state_dict"]
    if "ema_state_dict" in ckpt:
        ema_state = ckpt["ema_state_dict"]
        if is_paral_state(ema_state) and not is_paral_model(model):
            ema_state = de_parallel(ema_state)
    if is_paral_state(state) and not is_paral_model(model):
        state = de_parallel(state)
    try:
        model.load_state_dict(state, strict=True)
    except:
        model.load_state_dict(state, strict=False)
        flag = False
    if model_ema is not None:
        model_ema.shadow = ema_state
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if "amp" in ckpt and amp:
        apex.amp.load_state_dict(ckpt["amp"])

    if "epoch" in ckpt:
        if load_from is None and resume_from is not None:
            start_epoch = ckpt["epoch"]
    if is_main():
        best_d_acc, best_miou = log_loaded_info(ckpt, load_file)
    return start_epoch, best_d_acc, best_miou, flag


def save_checkpoint(work_dir, interval, model, model_ema, optimizer, scheduler, checkpoint):
    epoch = checkpoint["epoch"] + 1
    logger = get_root_logger()
    use_fp16 = checkpoint.pop("use_fp16", False)
    if use_fp16:
        checkpoint.update({"amp": apex.amp.state_dict()})
    checkpoint.update(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "lr": optimizer.param_groups[0]["lr"],
        }
    )
    if model_ema is not None:
        checkpoint.update({"ema_state_dict": model_ema.shadow})
    latest_path = osp.join(work_dir, "latest.pth")
    det_best_path = osp.join(work_dir, "det_best.pth")
    segm_best_path = osp.join(work_dir, "segm_best.pth")
    torch.save(checkpoint, latest_path)
    if is_main():
        logger.info(f"saved epoch {epoch} checkpoint at {latest_path}")
    if interval > 0 and epoch % interval == 0:
        torch.save(checkpoint, osp.join(work_dir, f"epoch_{epoch}.pth"))
    if checkpoint["d_acc"] > checkpoint["best_d_acc"]:
        shutil.copyfile(latest_path, det_best_path)
        if is_main():
            logger.info(f"saved epoch {epoch} checkpoint at {det_best_path}")
    if checkpoint["miou"] > checkpoint["best_miou"]:
        shutil.copyfile(latest_path, segm_best_path)
        if is_main():
            logger.info(f"saved epoch {epoch} checkpoint at {segm_best_path}")
