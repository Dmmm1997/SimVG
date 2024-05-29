# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

from mmdet.apis import inference_detector, show_result_pyplot
from mmcv.utils import Config
from simvg.models import build_model
from simvg.utils import load_checkpoint
from simvg.datasets.pipelines import Compose
from simvg.datasets import extract_data
from mmcv.parallel import collate
import os
from simvg.core import imshow_expr_bbox, imshow_expr_mask
import cv2
import numpy as np
from detectron2.utils.visualizer import GenericMask, VisImage
import pycocotools.mask as maskUtils
import matplotlib.colors as mplc


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--img", default="data/demo.jpg", help="Image file")
    parser.add_argument("--expression", default="front chair", help="text")
    parser.add_argument(
        "--config",
        default="work_dir/segmentation/pretrain/beit3-mask-seg_512_mixcoco/20240505_141258/20240505_141258_beit3-mask-seg_512_mixcoco.py",
        help="Config file",
    )
    parser.add_argument(
        "--checkpoint",
        default="work_dir/segmentation/pretrain/beit3-mask-seg_512_mixcoco/20240505_141258/segm_best.pth",
        help="Checkpoint file",
    )
    parser.add_argument("--output_dir", default="visualize/seg", help="Path to output file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument("--palette", default="coco", choices=["coco", "voc", "citys", "random"], help="Color palette used for visualization")
    parser.add_argument("--score-thr", type=float, default=0.7, help="bbox score threshold")
    parser.add_argument("--async-test", action="store_true", help="whether to set async options for async inference.")
    args = parser.parse_args()
    return args


def init_detector(args):
    cfg = Config.fromfile(args.config)
    cfg.img = args.img
    cfg.expression = args.expression
    cfg.output_dir = args.output_dir
    cfg.score_thr = args.score_thr
    cfg.device = args.device

    model = build_model(cfg.model)
    load_checkpoint(model, load_from=args.checkpoint)
    return model, cfg


def inference_detector(cfg, model):
    img, text = cfg.img, cfg.expression
    if "val" in cfg.data:
        pipline = cfg.data.val.pipeline
    else:
        pipline = cfg.data.val_refcoco_unc.pipeline
    pipline[0].type = "LoadFromRawSource"
    pipline[0].with_mask = False
    del pipline[5]["keys"][4]
    del pipline[5]["keys"][3]
    test_pipeline = Compose(pipline)
    result = {}
    ann = {}
    if cfg["dataset"] == "GRefCOCO":
        ann["bbox"] = [[[0, 0, 0, 0]]]
        ann["annotations"] = ["no target"]
    else:
        ann["bbox"] = [0, 0, 0, 0]
    ann["category_id"] = 0
    ann["expressions"] = [text]
    result["ann"] = ann
    result["which_set"] = "val"
    result["filepath"] = img

    data = test_pipeline(result)
    data = collate([data], samples_per_gpu=1)
    inputs = extract_data(data)

    img_metas = inputs["img_metas"]

    if "gt_bbox" in inputs:
        inputs.pop("gt_bbox")

    predictions = model(**inputs, return_loss=False, rescale=True, with_bbox=True)
    return predictions, img_metas


def draw_results(cfg, predictions, img_metas):
    pred_bboxes = predictions.pop("pred_bboxes")
    if cfg["dataset"] == "GRefCOCO":
        tmp_pred_bboxes = []
        for pred_bbox in pred_bboxes:
            img_level_bboxes = pred_bbox["boxes"]
            scores = pred_bbox["scores"]
            keep_ind = scores > cfg.score_thr
            img_level_bboxes = img_level_bboxes[keep_ind]
            tmp_pred_bboxes.append(img_level_bboxes)
        pred_bboxes = tmp_pred_bboxes

    for j, (img_meta, pred_bbox) in enumerate(zip(img_metas, pred_bboxes)):
        filename, expression = img_meta["filename"], img_meta["expression"]
        # scale_factors = img_meta["scale_factor"]
        # pred_bbox /= pred_bbox.new_tensor(scale_factors)
        os.makedirs(cfg.output_dir, exist_ok=True)
        outfile = os.path.join(cfg.output_dir, expression.replace(" ", "_") + "_" + os.path.basename(filename))
        imshow_expr_bbox(filename, pred_bbox, outfile)
        
def draw_results_seg(cfg, predictions, img_metas):    
    pred_masks = predictions.pop("pred_masks")
    for j, (img_meta, pred_mask) in enumerate(zip(img_metas, pred_masks)):
        filename, expression = img_meta["filename"], img_meta["expression"]
        os.makedirs(cfg.output_dir, exist_ok=True)
        outfile = os.path.join(cfg.output_dir, expression.replace(" ", "_") + "_" + os.path.basename(filename))
        imshow_expr_mask(filename, pred_mask, outfile)


def main(args):
    # build the model from a config file and a checkpoint file
    model, cfg = init_detector(args)
    model.to(args.device)
    # test a single image
    predictions, img_metas = inference_detector(cfg, model)
    # show the results
    draw_results_seg(cfg, predictions, img_metas)


if __name__ == "__main__":
    args = parse_args()
    main(args)
