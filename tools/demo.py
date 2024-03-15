# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

from mmdet.apis import inference_detector, show_result_pyplot
from mmcv.utils import Config
from seqtr.models import build_model
from seqtr.utils import load_checkpoint
from seqtr.datasets.pipelines import Compose
from seqtr.datasets import extract_data
from mmcv.parallel import collate
import os
from seqtr.core import imshow_expr_bbox


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--img", default="data/demo.jpg", help="Image file")
    parser.add_argument("--text", default="the smaller chair", help="text")
    parser.add_argument(
        "--config",
        default="work_dir/beit3_distill_314/(3-14)-noema-1.0decoder#1.0token#texaug#loadpretrain#share_predicthead#1.0aux_distill#schedulerdif/20240314_165033/20240314_165033_(3-14)-noema-1.0decoder#1.0token#texaug#loadpretrain#share_predicthead#1.0aux_distill#schedulerdif.py",
        help="Config file",
    )
    parser.add_argument(
        "--checkpoint",
        default="work_dir/beit3_distill_314/(3-14)-noema-1.0decoder#1.0token#texaug#loadpretrain#share_predicthead#1.0aux_distill#schedulerdif/20240314_165033/det_best.pth",
        help="Checkpoint file",
    )
    parser.add_argument("--output_dir", default="visualize/results", help="Path to output file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument("--palette", default="coco", choices=["coco", "voc", "citys", "random"], help="Color palette used for visualization")
    parser.add_argument("--score-thr", type=float, default=0.3, help="bbox score threshold")
    parser.add_argument("--async-test", action="store_true", help="whether to set async options for async inference.")
    args = parser.parse_args()
    return args


def init_detector(args):
    cfg = Config.fromfile(args.config)
    cfg.output_dir = args.output_dir
    cfg.score_thr = args.score_thr
    cfg.device = args.device

    model = build_model(cfg.model)
    load_checkpoint(model, load_from=args.checkpoint)
    return model, cfg


def inference_detector(cfg, model, img, text):
    cfg.data.val.pipeline[0].type = "LoadFromRawSource"
    test_pipeline = Compose(cfg.data.val.pipeline)
    result = {}
    ann = {}
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

    predictions = model(**inputs, return_loss=False, rescale=True, with_bbox=True)[1]

    return predictions, img_metas


def draw_results(cfg, predictions, img_metas):
    pred_bboxes = predictions.pop("pred_bboxes")
    if cfg["dataset"] == "GRefCOCO":
        tmp_pred_bboxes = []
        for pred_bbox in pred_bboxes:
            img_level_bboxes = pred_bbox["boxes"]
            scores = pred_bbox["scores"]
            keep_ind = scores > cfg.score_threahold
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


def main(args):
    # build the model from a config file and a checkpoint file
    model, cfg = init_detector(args)
    model.to(args.device)
    # test a single image
    predictions, img_metas = inference_detector(cfg, model, args.img, args.text)
    # show the results
    draw_results(cfg, predictions, img_metas)


if __name__ == "__main__":
    args = parse_args()
    main(args)
