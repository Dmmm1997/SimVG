# from visualizer import get_local
# get_local.activate() # 激活装饰器
import mmcv
import torch
import os.path as osp
# from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image
from simvg.utils import load_checkpoint, get_root_logger
from simvg.core import imshow_expr_bbox, imshow_expr_mask
from simvg.models import build_model, ExponentialMovingAverage
from simvg.datasets import extract_data, build_dataset, build_dataloader
# from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad


try:
    import apex
except:
    pass


def inference_model(cfg):
    datasets_cfg = [cfg.data.train]
    for which_set in cfg.which_set:
        datasets_cfg.append(eval(f"cfg.data.{which_set}"))

    datasets = list(map(build_dataset, datasets_cfg))
    dataloaders = list(map(lambda dataset: build_dataloader(cfg, dataset), datasets))

    model = build_model(cfg.model, word_emb=datasets[0].word_emb, num_token=datasets[0].num_token)
    model = model.cuda()
    if cfg.use_fp16:
        model = apex.amp.initialize(model, opt_level="O1")
        for m in model.modules():
            if hasattr(m, "fp16_enabled"):
                m.fp16_enabled = True
    if cfg.ema:
        model_ema = ExponentialMovingAverage(model, cfg.ema_factor)
    else:
        model_ema = None
    load_checkpoint(model, model_ema, None, cfg.checkpoint)
    if cfg.ema:
        model_ema.apply_shadow()

    model.eval()
    logger = get_root_logger()
    with_bbox, with_mask = False, False
    for i, which_set in enumerate(cfg.which_set):
        logger.info(f"inferencing on split {which_set}")
        prog_bar = mmcv.ProgressBar(len(datasets[i + 1]))
        with torch.no_grad():
            for batch, inputs in enumerate(dataloaders[i + 1]):
                gt_bbox, gt_mask, is_crowd = None, None, None
                if "gt_bbox" in inputs:
                    with_bbox = True
                    gt_bbox = inputs.pop("gt_bbox").data[0]
                if "gt_mask_rle" in inputs:
                    with_mask = True
                    gt_mask = inputs.pop("gt_mask_rle").data[0]
                if "is_crowd" in inputs:
                    inputs.pop("is_crowd").data[0]

                if not cfg.distributed:
                    inputs = extract_data(inputs)

                img_metas = inputs["img_metas"]
                batch_size = len(img_metas)

                predictions = model(**inputs, return_loss=False, rescale=True, with_bbox=with_bbox, with_mask=with_mask)[1]

                pred_bboxes = [None for _ in range(batch_size)]
                if with_bbox:
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

                pred_masks = [None for _ in range(batch_size)]
                if with_mask:
                    pred_masks = predictions.pop("pred_masks")

                for j, (img_meta, pred_bbox, pred_mask) in enumerate(zip(img_metas, pred_bboxes, pred_masks)):
                    filename, expression = img_meta["filename"], img_meta["expression"]
                    bbox_gt, mask_gt = None, None
                    if cfg.with_gt and with_bbox:
                        bbox_gt = gt_bbox[j]
                    if cfg.with_gt and with_mask:
                        mask_gt = gt_mask[j]

                    scale_factors = img_meta["scale_factor"]
                    # pred_bbox /= pred_bbox.new_tensor(scale_factors)
                    bbox_gt /= bbox_gt.new_tensor(scale_factors)

                    outfile = osp.join(cfg.output_dir, cfg.dataset + "_" + which_set, expression.replace(" ", "_") + "_" + osp.basename(filename))

                    if with_bbox:
                        imshow_expr_bbox(filename, pred_bbox, outfile, gt_bbox=bbox_gt)
                    if with_mask:
                        imshow_expr_mask(filename, pred_mask, outfile, gt_mask=mask_gt, overlay=cfg.overlay)

                    prog_bar.update()
                    
                    if cfg.enable_attnmap:
                        # 从缓存中获取注意力权重
                        decoder_atten_weight_list = []
                        for tmp_weight in get_local.cache['MultiheadAttention.forward']:
                            if tmp_weight.shape[-1] == 400:
                                decoder_atten_weight_list.append(tmp_weight)
                        decoder_atten_weight_list = decoder_atten_weight_list[-1]
                        
                        decoder_attn_map = decoder_atten_weight_list[j]
                        # for idx, decoder_attn_map in enumerate(decoder_atten_weight_list):
                        decoder_attn_map = decoder_attn_map.reshape(1, 1, 20, 20)
                        decoder_attn_map_min = decoder_attn_map.min()
                        decoder_attn_map_max = decoder_attn_map.max()
                        decoder_attn_map = (decoder_attn_map - decoder_attn_map_min) / (decoder_attn_map_max - decoder_attn_map_min)
                        decoder_attn_map_ = torch.nn.functional.interpolate(torch.from_numpy(decoder_attn_map), img_meta["ori_shape"][:-1], mode="bilinear")[
                            0, 0]
                        image = mmcv.imread(filename)
                        # image_with_bounding_boxes = det_cam_visualizer.show_cam(image, bboxes, labels, decoder_attn_map_.numpy(), False)
                        cam_image_renormalized = show_cam_on_image(image / 255, decoder_attn_map_.numpy(), use_rgb=False)

                        outfile = osp.join(cfg.output_dir, cfg.dataset + "_" + which_set, expression.replace(" ", "_") + "_attnmap_" + osp.basename(filename))
                        mmcv.imwrite(cam_image_renormalized, outfile)
                        
                        attn_map = get_local.cache['EncoderLayer.forward'][12][0, j, -1, 1 : -20].reshape(1, 1, 20, 20)
                        attn_map_min = attn_map.min()
                        attn_map_max = attn_map.max()
                        attn_map = (attn_map - attn_map_min) / (attn_map_max - attn_map_min)
                        encoder_attn_map_ = torch.nn.functional.interpolate(torch.from_numpy(attn_map), img_meta["ori_shape"][:-1], mode="bilinear")[0,0]
                        cam_image_renormalized = show_cam_on_image(image / 255, encoder_attn_map_.numpy(), use_rgb=False)
                        outfile = osp.join(cfg.output_dir, cfg.dataset + "_" + which_set, expression.replace(" ", "_") + "_attnmap_encoder_" + osp.basename(filename))
                        mmcv.imwrite(cam_image_renormalized, outfile)
                        
    if cfg.ema:
        model_ema.restore()
