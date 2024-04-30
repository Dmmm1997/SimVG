import time
import argparse
import os.path as osp
import torch.distributed as dist

import mmcv
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info
from mmcv.parallel import MMDistributedDataParallel

from simvg.core import build_optimizer, build_scheduler
from simvg.datasets import build_dataset, build_dataloader
from simvg.models import build_model, ExponentialMovingAverage
from simvg.apis import set_random_seed, train_model, evaluate_model
from simvg.utils import get_root_logger, load_checkpoint, save_checkpoint, load_pretrained_checkpoint, is_main, init_dist

import warnings

warnings.filterwarnings("ignore")

try:
    import apex
except:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="SeqTR-train")
    parser.add_argument("config", help="training configuration file path.")
    parser.add_argument("--work-dir", help="directory of config file, training logs, and checkpoints.")
    parser.add_argument("--resume-from", help="resume training from the saved .pth checkpoint, only used in training.")
    parser.add_argument("--load-from", help="resume training from the saved .pth checkpoint, only used in training.")
    parser.add_argument(
        "--finetune-from",
        help="finetune from the saved .pth checkpoint, only used after SeqTR has been pre-trained on the merged dadtaset.",
    )
    parser.add_argument("--launcher", choices=["none", "pytorch"], default="none")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    args = parser.parse_args()
    return args


def main_worker(cfg):
    cfg.distributed = False
    if cfg.launcher == "pytorch":
        cfg.distributed = True
        init_dist()
    cfg.rank, cfg.world_size = get_dist_info()
    if is_main():
        logger = get_root_logger(log_file=osp.join(cfg.work_dir, str(cfg.timestamp) + "_train_log.txt"))
        logger.info(cfg.pretty_text)
        cfg.dump(osp.join(cfg.work_dir, f"{cfg.timestamp}_" + osp.basename(cfg.config)))

    datasets_cfgs = [cfg.data.train]
    if cfg.dataset == "Mixed":
        items = ["val_refcoco_unc", "val_refcocoplus_unc", "val_refcocog_umd", "val_referitgame_berkeley", "val_flickr30k"]
        for item in items:
            if getattr(cfg.data, item, None):
                datasets_cfgs += [getattr(cfg.data, item)]
    else:
        datasets_cfgs += [cfg.data.val]

    datasets = list(map(build_dataset, datasets_cfgs))
    dataloaders = list(map(lambda dataset: build_dataloader(cfg, dataset), datasets))

    model = build_model(cfg.model, word_emb=datasets[0].word_emb, num_token=datasets[0].num_token)
    model = model.cuda()
    train_params = [
        {
            "params": [p for n, p in model.named_parameters() if "vis_enc" in n and p.requires_grad],
            "lr": cfg.optimizer_config.pop("lr_vis_enc"),
        },
        {
            "params": [p for n, p in model.named_parameters() if "lan_enc" in n and p.requires_grad],
            "lr": cfg.optimizer_config.pop("lr_lan_enc"),
        },
        {
            "params": [p for n, p in model.named_parameters() if "lan_enc" not in n and "vis_enc" not in n and p.requires_grad],
            "lr": cfg.optimizer_config.lr,
        },
    ]

    optimizer = build_optimizer(cfg.optimizer_config, train_params)
    scheduler = build_scheduler(cfg.scheduler_config, optimizer)

    if cfg.use_fp16:
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
        for m in model.modules():
            if hasattr(m, "fp16_enabled"):
                m.fp16_enabled = True

    if cfg.distributed:
        model = MMDistributedDataParallel(model, device_ids=[cfg.rank], find_unused_parameters=True)
    model_ema = ExponentialMovingAverage(model, cfg.ema_factor) if cfg.ema else None
    start_epoch, best_d_acc, best_miou = -1, 0.0, 0.0
    if cfg.resume_from:
        start_epoch, _, _, flag = load_checkpoint(model, model_ema, cfg.resume_from, amp=cfg.use_fp16, optimizer=optimizer, scheduler=scheduler)
        if not flag:
            model_ema = ExponentialMovingAverage(model, cfg.ema_factor) if cfg.ema else None
    elif cfg.finetune_from:
        load_pretrained_checkpoint(model, model_ema, cfg.finetune_from, amp=cfg.use_fp16)
    elif cfg.load_from:
        start_epoch, best_d_acc, best_miou, flag = load_checkpoint(model, model_ema, load_from=cfg.load_from)
        if not flag:
            model_ema = ExponentialMovingAverage(model, cfg.ema_factor) if cfg.ema else None

    import time

    begin_time = time.time()
    for epoch in range(start_epoch + 1, cfg.scheduler_config.max_epoch):
        start_time = time.time()
        train_model(epoch, cfg, model, model_ema, optimizer, dataloaders[0])
        this_epoch_train_time = int(time.time() - start_time)
        if is_main():
            logger.info("this_epoch_train_time={}m-{}s".format(this_epoch_train_time // 60, this_epoch_train_time % 60))
            
        if epoch%cfg.evaluate_interval==0 and epoch>=cfg.start_evaluate_epoch:
            d_acc, miou = 0, 0
            for _loader in dataloaders[1:]:
                if is_main():
                    logger.info("Evaluating dataset: {}".format(_loader.dataset.which_set))
                set_d_acc, set_miou = evaluate_model(epoch, cfg, model, _loader)

                if cfg.ema:
                    if is_main():
                        logger.info("Evaluating dataset using ema: {}".format(_loader.dataset.which_set))
                    model_ema.apply_shadow()
                    ema_set_d_acc, ema_set_miou = evaluate_model(epoch, cfg, model, _loader)
                    model_ema.restore()

                if cfg.ema:
                    d_acc += ema_set_d_acc
                    miou += ema_set_miou
                else:
                    d_acc += set_d_acc
                    miou += set_miou

            d_acc /= len(dataloaders[1:])
            miou /= len(dataloaders[1:])

            if is_main():
                this_epoch_total_time = int(time.time() - start_time)
                logger.info("this_epoch_total_time={}m-{}s".format(this_epoch_total_time // 60, this_epoch_total_time % 60))
                total_time = int(time.time() - begin_time)
                logger.info("total_time={}m-{}s".format(total_time // 60, total_time % 60))

            if is_main():
                # if cfg["dataset"]=="GRefCOCO":
                #     saved_info = {"epoch": epoch, "f1_score": d_acc, "n_acc": miou, "best_f1_score": best_d_acc, "best_n_acc": best_miou, "amp": cfg.use_fp16}
                # else:
                #     saved_info = {"epoch": epoch, "d_acc": d_acc, "miou": miou, "best_d_acc": best_d_acc, "best_miou": best_miou, "amp": cfg.use_fp16}
                saved_info = {
                    "epoch": epoch,
                    "d_acc": d_acc,
                    "miou": miou,
                    "best_d_acc": best_d_acc,
                    "best_miou": best_miou,
                    "amp": cfg.use_fp16,
                }
                save_checkpoint(
                    cfg.work_dir,
                    cfg.save_interval,
                    model,
                    model_ema,
                    optimizer,
                    scheduler,
                    saved_info,
                )
            best_d_acc = max(d_acc, best_d_acc)
            best_miou = max(miou, best_miou)

        scheduler.step()
        
        if cfg.distributed:
            dist.barrier()

    if cfg.distributed:
        dist.destroy_process_group()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # cfg.work_dir = f"./work_dir/{cfg.timestamp}_" + osp.splitext(osp.basename(args.config))[0]
        cfg.work_dir = f"./work_dir/"+args.config.split("configs/")[-1].split(".py")[0]
    cfg.work_dir = osp.join(cfg.work_dir, f"{cfg.timestamp}")
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.finetune_from is not None:
        cfg.finetune_from = args.finetune_from
    if args.load_from is not None:
        cfg.load_from = args.load_from
    cfg.launcher = args.launcher
    cfg.config = args.config

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    if cfg.seed is not None:
        set_random_seed(cfg.seed, deterministic=cfg.deterministic)

    main_worker(cfg)


if __name__ == "__main__":
    main()
