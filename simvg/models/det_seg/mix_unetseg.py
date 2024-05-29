import torch
import numpy
from simvg.models import MODELS
from mmdet.core import BitmapMasks
import pycocotools.mask as maskUtils
from .one_stage import OneStageModel
import numpy as np


@MODELS.register_module()
class MIXUnetSeg(OneStageModel):
    def __init__(self, word_emb, num_token, vis_enc, lan_enc, head, fusion, loss_bbox=None):
        super(MIXUnetSeg, self).__init__(word_emb, num_token, vis_enc, lan_enc, head, fusion)
        self.patch_size = vis_enc["patch_size"]

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
        img_feat = img_feat.transpose(-1, -2).reshape(B, -1, H // self.patch_size, W // self.patch_size)

        losses_dict, mask_seg = self.head.forward_train(img_feat, gt_mask, cls_feat, text_feat, text_attention_mask)

        with torch.no_grad():
            predictions = self.get_predictions(mask_seg, img_metas, rescale=rescale)

        return losses_dict, predictions

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
        
        mask_seg = self.head.forward_test(img_feat, cls_feat, text_feat, text_attention_mask)

        predictions = self.get_predictions(mask_seg, img_metas, rescale=rescale)

        return predictions

    def get_predictions(self, mask_seg, img_metas, rescale=False):
        """Args:
        seq_out_dict (dict[tensor]): [batch_size, 4/2*num_ray+1].

        rescale (bool): whether to rescale predictions from `img_shape`/`pad_shape`
            back to `ori_shape`.
        """
        pred_bboxes, pred_masks = None, []
        mask_binary = mask_seg.argmax(1)
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

        return dict(pred_bboxes=pred_bboxes, pred_masks=pred_masks)
