import argparse
import torch.distributed as dist

from simvg.apis import evaluate_model, set_random_seed
from simvg.datasets import build_dataset, build_dataloader
from simvg.models import build_model, ExponentialMovingAverage
from simvg.utils import get_root_logger, load_checkpoint, init_dist, is_main, load_pretrained_checkpoint

from mmcv.runner import get_dist_info
from mmcv.utils import Config, DictAction
from mmcv.parallel import MMDistributedDataParallel
import os
import pandas as pd

try:
    import apex
except:
    pass


def main_worker(cfg):
    cfg.distributed = False
    if cfg.launcher == "pytorch":
        cfg.distributed = True
        init_dist()
    cfg.rank, cfg.world_size = get_dist_info()
    work_dir = os.path.dirname(cfg.load_from)
    if is_main():
        logger = get_root_logger(log_file=os.path.join(work_dir, "test_log.txt"))
        # logger = get_root_logger()
        logger.info(cfg.pretty_text)

    if cfg.dataset == "Mixed":
        prefix = [
            "val_refcoco_unc",
            "val_refcocoplus_unc",
            "val_refcocog_umd",
            #   'val_referitgame_berkeley',
            #   'val_flickr30k'
        ]
        datasets_cfgs = [
            cfg.data.train,
            cfg.data.val_refcoco_unc,
            cfg.data.val_refcocoplus_unc,
            cfg.data.val_refcocog_umd,
            # cfg.data.val_referitgame_berkeley,
            # cfg.data.val_flickr30k
        ]
    elif cfg.dataset == "MixedSeg":
        prefix = [
            "val_refcoco_unc",
            "testA_refcoco_unc",
            "testB_refcoco_unc",
            "val_refcocoplus_unc",
            "testA_refcocoplus_unc",
            "testB_refcocoplus_unc",
            "val_refcocog_umd",
            "test_refcocog_umd",
            # "val_refcocog_google",
        ]
        datasets_cfgs = [
            cfg.data.train,
            cfg.data.val_refcoco_unc,
            cfg.data.testA_refcoco_unc,
            cfg.data.testB_refcoco_unc,
            cfg.data.val_refcocoplus_unc,
            cfg.data.testA_refcocoplus_unc,
            cfg.data.testB_refcocoplus_unc,
            cfg.data.val_refcocog_umd,
            cfg.data.test_refcocog_umd,
            # cfg.data.val_refcocog_google,
        ]
    else:
        prefix = ["val"]
        datasets_cfgs = [cfg.data.train, cfg.data.val]
        if hasattr(cfg.data, "testA") and hasattr(cfg.data, "testB"):
            datasets_cfgs.append(cfg.data.testA)
            datasets_cfgs.append(cfg.data.testB)
            prefix.extend(["testA", "testB"])
        elif hasattr(cfg.data, "test"):
            datasets_cfgs.append(cfg.data.test)
            prefix.extend(["test"])
    datasets = list(map(build_dataset, datasets_cfgs))
    dataloaders = list(map(lambda dataset: build_dataloader(cfg, dataset), datasets[1:]))
    cfg.model.mask_save_target_dir = work_dir
    cfg.model.threshold = cfg.threshold
    model = build_model(cfg.model, word_emb=datasets[0].word_emb, num_token=datasets[0].num_token)
    model = model.cuda()
    if cfg.use_fp16:
        model = apex.amp.initialize(model, opt_level="O1")
        for m in model.modules():
            if hasattr(m, "fp16_enabled"):
                m.fp16_enabled = True
    if cfg.distributed:
        model = MMDistributedDataParallel(model, device_ids=[cfg.rank])
    model_ema = ExponentialMovingAverage(model, cfg.ema_factor) if cfg.ema else None
    if cfg.load_from:
        load_checkpoint(model, model_ema, load_from=cfg.load_from)
    elif cfg.finetune_from:
        # hacky way
        load_pretrained_checkpoint(model, model_ema, cfg.finetune_from, amp=cfg.use_fp16)

    excel_results = {
        'DetAcc': [],
        'MaskAcc': [],
        'miou': [],
        'oiou': []
    }
    index_names = []
    for eval_loader, _prefix in zip(dataloaders, prefix):
        if is_main():
            logger = get_root_logger()
            logger.info(f"SimVG - evaluating set {_prefix}")
        set_d_acc, set_m_acc, set_miou, set_oiou = evaluate_model(-1, cfg, model, eval_loader)
        if cfg.ema:
            if is_main():
                logger = get_root_logger()
                logger.info(f"SimVG - evaluating set {_prefix} using ema")
            model_ema.apply_shadow()
            set_d_acc, set_m_acc, set_miou, set_oiou = evaluate_model(-1, cfg, model, eval_loader)
            model_ema.restore()
        if is_main():
            excel_results["DetAcc"].append("{:.2f}".format(set_d_acc))
            excel_results["MaskAcc"].append("{:.2f}".format(set_m_acc))
            excel_results["miou"].append("{:.2f}".format(set_miou))
            excel_results["oiou"].append("{:.2f}".format(set_oiou))
            index_names.append(_prefix)
    if is_main():
        df = pd.DataFrame(excel_results, index=index_names)
        target_excel_path = os.path.join(work_dir, 'output.xlsx')
        df.to_excel(target_excel_path, engine='openpyxl')
        logger.info("sucessfully save the results to {} !!!".format(target_excel_path))

    if cfg.distributed:
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="SeqTR-test")
    parser.add_argument("config", help="test configuration file path.")
    parser.add_argument("--load-from", help="load from the saved .pth checkpoint, only used in validation.")
    parser.add_argument("--finetune-from", help="load from the pretrained checkpoint, only used in validation.")
    parser.add_argument("--launcher", choices=["none", "pytorch"], default="none")
    parser.add_argument("--threshold", default=0.5, type=float)
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


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.load_from = args.load_from
    cfg.finetune_from = args.finetune_from
    cfg.launcher = args.launcher
    cfg.threshold = args.threshold

    if cfg.seed is not None:
        set_random_seed(cfg.seed, deterministic=cfg.deterministic)

    main_worker(cfg)


if __name__ == "__main__":
    main()
