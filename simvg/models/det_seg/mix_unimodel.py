import torch
import numpy
from simvg.models import MODELS
from mmdet.core import BitmapMasks
import pycocotools.mask as maskUtils
from .one_stage import OneStageModel
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from simvg.utils import is_main
import os
from ..utils import xywh_to_x1y1x2y2
from ..heads.uni_head_simple import get_maskouterbox
import cv2

font = ImageFont.load_default()


def box_seg_visualization(pred_box, pred_seg, pred_box_first, pred_seg_first, save_filename, img_metas, text, gt_mask, gt_box, threshold=0.5):
    H, W = pred_seg.shape[-2:]

    gt_mask = maskUtils.decode(gt_mask)

    pred_box = (xywh_to_x1y1x2y2(pred_box).cpu().detach().numpy() * [W, H, W, H]).astype(np.int32)
    pred_seg = pred_seg.sigmoid().squeeze(0)
    pred_seg[pred_seg < threshold] = 0.0
    pred_seg[pred_seg >= threshold] = 1.0
    pred_segouterbox = get_maskouterbox(pred_seg.unsqueeze(0), threshold=threshold).squeeze(0)
    pred_seg = pred_seg.cpu().detach().numpy().astype(np.int32)

    pred_box_first = (xywh_to_x1y1x2y2(pred_box_first).cpu().detach().numpy() * [W, H, W, H]).astype(np.int32)
    pred_seg_first = pred_seg_first.sigmoid().squeeze(0)
    pred_seg_first[pred_seg_first < threshold] = 0.0
    pred_seg_first[pred_seg_first >= threshold] = 1.0
    pred_segouterbox_first = get_maskouterbox(pred_seg_first.unsqueeze(0), threshold=threshold).squeeze(0)
    pred_seg_first = pred_seg_first.cpu().detach().numpy().astype(np.int32)

    # draw pred
    mask_image = Image.fromarray(pred_seg * 255)
    image_pred = Image.new("RGB", (W, H))
    image_pred.paste(mask_image)
    draw_pred = ImageDraw.Draw(image_pred)
    draw_pred.rectangle(list(pred_box), outline="blue", width=2)
    # draw_pred.rectangle(list(pred_segouterbox), outline="blue", width=2)
    if isinstance(text, str):
        text_position = (10, 10)
        draw_pred.text(text_position, text, fill="red", font=font)

    # draw pred first
    mask_image_first = Image.fromarray(pred_seg_first * 255)
    image_pred_first = Image.new("RGB", (W, H))
    image_pred_first.paste(mask_image_first)
    draw_pred_first = ImageDraw.Draw(image_pred_first)
    draw_pred_first.rectangle(list(pred_box_first), outline="blue", width=2)
    # draw_pred_first.rectangle(list(pred_segouterbox_first), outline="blue", width=2)
    if isinstance(text, str):
        text_position = (10, 10)
        draw_pred_first.text(text_position, text, fill="red", font=font)

    # draw gt
    box_gt = (gt_box.cpu().detach().numpy()).astype(np.int32)
    mask_gt = gt_mask.astype(np.int32)
    mask_gt = Image.fromarray(mask_gt * 255)
    image_gt = Image.new("RGB", (W, H))
    image_gt.paste(mask_gt)
    draw_gt = ImageDraw.Draw(image_gt)
    draw_gt.rectangle(list(box_gt), outline="red", width=2)

    # draw source image
    img_metas["filename"]
    file_name = img_metas["filename"]
    expression = img_metas["expression"]
    img_source = Image.open(file_name)
    img_source = img_source.resize((W, H))

    concat_image = Image.new("RGB", (W * 4 + 30, H), "white")
    concat_image.paste(img_source, (0, 0))
    concat_image.paste(image_gt, (W + 10, 0))
    concat_image.paste(image_pred_first, (W * 2 + 20, 0))
    concat_image.paste(image_pred, (W * 3 + 30, 0))

    save_filename = save_filename + "-{}-{}".format(expression, file_name.split("/")[-1])

    concat_image.save(save_filename)


def heatmap_visulization(featmap, saved_path):
    heatmap = featmap.cpu().detach().numpy()[0]
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    heatmap = (heatmap * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(saved_path, colored_heatmap)


@MODELS.register_module()
class MIXUniModel(OneStageModel):
    def __init__(self, word_emb, num_token, vis_enc, lan_enc, head, fusion, loss_bbox=None, mask_save_target_dir="", threshold=0.5):
        super(MIXUniModel, self).__init__(word_emb, num_token, vis_enc, lan_enc, head, fusion)
        self.patch_size = vis_enc["patch_size"]
        self.visualize = False
        if len(mask_save_target_dir) > 0:
            self.visualize = True
        if self.visualize:
            self.train_mask_save_target_dir = os.path.join(mask_save_target_dir, "train_vis")
            self.val_mask_save_target_dir = os.path.join(mask_save_target_dir, "val_vis")
            os.makedirs(self.train_mask_save_target_dir, exist_ok=True)
            os.makedirs(self.val_mask_save_target_dir, exist_ok=True)
        self.iter = 0
        self.threshold = threshold

    def forward_train(
        self,
        img,
        ref_expr_inds,
        img_metas,
        text_attention_mask=None,
        gt_bbox=None,
        gt_mask_vertices=None,
        mass_center=None,
        gt_mask=None,
        rescale=False,
        epoch=None,
    ):
        """Args:
        img (tensor): [batch_size, c, h_batch, w_batch].

        ref_expr_inds (tensor): [batch_size, max_token].

        img_metas (list[dict]): list of image info dict where each dict
            has: 'img_shape', 'scale_factor', and may also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys see
            `seqtr/datasets/pipelines/formatting.py:CollectData`.

        gt_bbox (list[tensor]): [4, ], in [tl_x, tl_y, br_x, br_y] format,
            the coordinates are in 'img_shape' scale.

        gt_mask_vertices (list[tensor]): [batch_size, 2, num_ray], padded values are -1,
            the coordinates are in 'pad_shape' scale.

        rescale (bool): whether to rescale predictions from `img_shape`/`pad_shape`
            back to `ori_shape`.

        """
        B, _, H, W = img.shape
        img_feat, text_feat, cls_feat = self.extract_visual_language(img, ref_expr_inds, text_attention_mask)
        img_feat = img_feat.transpose(-1, -2).reshape(B, -1, H // self.patch_size, W // self.patch_size)  # (B, C, H, W)

        targets = {"mask": gt_mask, "bbox": gt_bbox, "img_metas": img_metas, "epoch": epoch}

        losses_dict, pred_dict, extra_dict = self.head.forward_train(img_feat, targets, cls_feat, text_feat, text_attention_mask, img)

        with torch.no_grad():
            predictions = self.get_predictions(pred_dict, img_metas, rescale=rescale, threshold=self.threshold)

        self.iter += 1
        if is_main() and self.iter % 50 == 0 and self.visualize:
            self.visualiation(pred_dict, img_metas, targets, self.train_mask_save_target_dir, extra_dict)

        return losses_dict, predictions

    def visualiation(self, pred_dict, img_metas, targets, save_target_dir, extra_dict=None):
        # save the box and seg
        save_filename = os.path.join(save_target_dir, str(self.iter))
        box_seg_visualization(
            pred_dict["pred_bbox"][0],
            # maskouterbbox.int().cpu().detach().numpy(),
            pred_dict["pred_mask"][0],
            pred_dict["pred_bbox_first"][0],
            pred_dict["pred_mask_first"][0],
            save_filename=save_filename,
            img_metas=img_metas[0],
            text=None,
            gt_box=targets["bbox"][0],
            gt_mask=targets["mask"][0],
            threshold=self.threshold,
        )
        if extra_dict is not None:
            if "unified_img_feat" in extra_dict:
                unified_img_feat = extra_dict["unified_img_feat"]
                heatmap_visulization(unified_img_feat[0], save_filename + "unified_heatmap.jpg")
            if "img_feat" in extra_dict:
                img_feat = extra_dict["img_feat"]
                heatmap_visulization(img_feat[0], save_filename + "heatmap.jpg")
            if "seg_feat" in extra_dict:
                img_feat = extra_dict["seg_feat"]
                heatmap_visulization(img_feat[0], save_filename + "seg_heatmap.jpg")
            if "box_feat" in extra_dict:
                img_feat = extra_dict["box_feat"]
                heatmap_visulization(img_feat[0], save_filename + "box_heatmap.jpg")

    def extract_visual_language(self, img, ref_expr_inds, text_attention_mask=None):
        x, y, c = self.vis_enc(img, ref_expr_inds, text_attention_mask)
        return x, y, c

    @torch.no_grad()
    def forward_test(
        self,
        img,
        ref_expr_inds,
        img_metas,
        text_attention_mask=None,
        with_bbox=False,
        with_mask=False,
        gt_bbox=None,
        gt_mask=None,
        rescale=False,
    ):
        """Args:
        img (tensor): [batch_size, c, h_batch, w_batch].

        ref_expr_inds (tensor): [batch_size, max_token], padded value is 0.

        img_metas (list[dict]): list of image info dict where each dict
            has: 'img_shape', 'scale_factor', and may also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys see
            `rec/datasets/pipelines/formatting.py:CollectData`.

        with_bbox/with_mask: whether to generate bbox coordinates or mask contour vertices,
            which has slight differences.

        rescale (bool): whether to rescale predictions from `img_shape`/`pad_shape`
            back to `ori_shape`.
        """

        B, _, H, W = img.shape
        img_feat, text_feat, cls_feat = self.extract_visual_language(img, ref_expr_inds, text_attention_mask)
        img_feat = img_feat.transpose(-1, -2).reshape(B, -1, H // self.patch_size, W // self.patch_size)

        targets = {"mask": gt_mask, "bbox": gt_bbox, "img_metas": img_metas}

        pred_dict, extra_dict = self.head.forward_test(img_feat, cls_feat, text_feat, text_attention_mask, img)

        predictions = self.get_predictions(pred_dict, img_metas, rescale=rescale, threshold=self.threshold)

        self.iter += 1
        if is_main() and self.iter % 20 == 0 and self.visualize:
            self.visualiation(pred_dict, img_metas, targets, self.val_mask_save_target_dir, extra_dict)

        return predictions

    def get_predictions(self, pred, img_metas, rescale=False, threshold=0.5):
        """Args:
        seq_out_dict (dict[tensor]): [batch_size, 4/2*num_ray+1].

        rescale (bool): whether to rescale predictions from `img_shape`/`pad_shape`
            back to `ori_shape`.
        """

        pred_bboxes, pred_masks = [], []
        pred_bboxes_first, pred_masks_first = [], []
        bboxes, mask_seg = pred.get("pred_bbox", None), pred.get("pred_mask", None)
        bboxes_first_stage, mask_seg_first_stage = pred.get("pred_bbox_first", None), pred.get("pred_mask_first", None)

        if bboxes is not None:
            for pred_box, img_meta in zip(bboxes, img_metas):
                img_size = img_meta["img_shape"][:2]
                pred_bbox = pred_box * img_size[0]
                output_bbox = torch.zeros_like(pred_bbox, device=pred_bbox.device)
                output_bbox[0] = pred_bbox[0] - pred_bbox[2] / 2.0  # x1
                output_bbox[1] = pred_bbox[1] - pred_bbox[3] / 2.0  # y1
                output_bbox[2] = pred_bbox[0] + pred_bbox[2] / 2.0  # x2
                output_bbox[3] = pred_bbox[1] + pred_bbox[3] / 2.0  # y2
                if rescale:
                    scale_factors = img_meta["scale_factor"]
                    output_bbox /= output_bbox.new_tensor(scale_factors)
                pred_bboxes.append(output_bbox)

        if bboxes_first_stage is not None:
            for pred_box, img_meta in zip(bboxes_first_stage, img_metas):
                img_size = img_meta["img_shape"][:2]
                pred_bbox = pred_box * img_size[0]
                output_bbox = torch.zeros_like(pred_bbox, device=pred_bbox.device)
                output_bbox[0] = pred_bbox[0] - pred_bbox[2] / 2.0  # x1
                output_bbox[1] = pred_bbox[1] - pred_bbox[3] / 2.0  # y1
                output_bbox[2] = pred_bbox[0] + pred_bbox[2] / 2.0  # x2
                output_bbox[3] = pred_bbox[1] + pred_bbox[3] / 2.0  # y2
                if rescale:
                    scale_factors = img_meta["scale_factor"]
                    output_bbox /= output_bbox.new_tensor(scale_factors)
                pred_bboxes_first.append(output_bbox)

        if mask_seg is not None:
            mask_binary = mask_seg.sigmoid().squeeze(1)
            mask_binary[mask_binary < threshold] = 0.0
            mask_binary[mask_binary >= threshold] = 1.0
            for mask, img_meta in zip(mask_binary, img_metas):
                h_pad, w_pad = img_meta["pad_shape"][:2]
                # h, w = img_meta['img_shape'][:2]
                pred_rle = maskUtils.encode(numpy.asfortranarray(mask.cpu().numpy().astype(np.uint8)))
                if rescale:
                    h_img, w_img = img_meta["ori_shape"][:2]
                    pred_mask = BitmapMasks(maskUtils.decode(pred_rle)[None], h_pad, w_pad)
                    pred_mask = pred_mask.resize((h_img, w_img))
                    pred_mask = pred_mask.masks[0]
                    pred_mask = numpy.asfortranarray(pred_mask)
                    pred_rle = maskUtils.encode(pred_mask)  # dict
                pred_masks.append(pred_rle)

        if mask_seg_first_stage is not None:
            mask_binary = mask_seg_first_stage.sigmoid().squeeze(1)
            mask_binary[mask_binary < threshold] = 0.0
            mask_binary[mask_binary >= threshold] = 1.0
            for mask, img_meta in zip(mask_binary, img_metas):
                h_pad, w_pad = img_meta["pad_shape"][:2]
                # h, w = img_meta['img_shape'][:2]
                pred_rle = maskUtils.encode(numpy.asfortranarray(mask.cpu().numpy().astype(np.uint8)))
                if rescale:
                    h_img, w_img = img_meta["ori_shape"][:2]
                    pred_mask = BitmapMasks(maskUtils.decode(pred_rle)[None], h_pad, w_pad)
                    pred_mask = pred_mask.resize((h_img, w_img))
                    pred_mask = pred_mask.masks[0]
                    pred_mask = numpy.asfortranarray(pred_mask)
                    pred_rle = maskUtils.encode(pred_mask)  # dict
                pred_masks_first.append(pred_rle)

        return dict(pred_bboxes=pred_bboxes, pred_masks=pred_masks, pred_bboxes_first=pred_bboxes_first, pred_masks_first=pred_masks_first)
