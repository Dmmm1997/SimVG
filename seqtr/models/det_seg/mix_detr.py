import torch
from seqtr.models import MODELS
from .one_stage import OneStageModel
from detectron2.modeling import detector_postprocess


@MODELS.register_module()
class MIXDETR(OneStageModel):
    def __init__(self, word_emb, num_token, vis_enc, lan_enc, head, fusion):
        super(MIXDETR, self).__init__(word_emb, num_token, vis_enc, lan_enc, head, fusion)
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

        losses_dict, output = self.head.forward_train(
            img_feat, img_metas, gt_bbox=gt_bbox, text_feat=text_feat, gt_mask_vertices=gt_mask_vertices
        )

        with torch.no_grad():
            if img_metas[0].get("target", None) is None:
                predictions = self.get_predictions(output, img_metas, rescale=rescale)
            else: # grec output
                predictions = self.get_predictions_grec(output, img_metas, rescale=rescale)

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

        output = self.head.forward_test(img_feat, img_metas, text_feat=text_feat, with_bbox=with_bbox, with_mask=with_mask)

        if img_metas[0].get("target", None) is None:
            predictions = self.get_predictions(output, img_metas, rescale=rescale)
        else: # grec output
            predictions = self.get_predictions_grec(output, img_metas, rescale=rescale)

        return predictions

    def get_predictions(self, output, img_metas, rescale=False):
        box_cls = output["pred_logits"]
        box_pred = output["pred_boxes"]
        image_sizes = [img_meta["img_shape"] for img_meta in img_metas]
        results = self.head.inference(box_cls, box_pred, image_sizes)
        # processed_results = []
        pred_bboxes = []
        for results_per_image, image_size in zip(results, image_sizes):
            height = image_size[0]
            width = image_size[1]
            r = detector_postprocess(results_per_image, height, width)
            # infomation extract
            pred_boxes = r.pred_boxes
            scores = r.scores
            pred_classes = r.pred_classes
            # best index
            best_ind = torch.argmax(scores)
            pred_box = pred_boxes[int(best_ind)].tensor
            pred_bboxes.append(pred_box)
            # processed_results.append({"instances": r})
        pred_bboxes = torch.cat(pred_bboxes, dim=0)
        pred_masks = None
        return dict(pred_bboxes=pred_bboxes, pred_masks=pred_masks)
    
    def get_predictions_grec(self, output, img_metas, rescale=False):
        box_cls = output["pred_logits"]
        box_pred = output["pred_boxes"]
        image_sizes = [img_meta["img_shape"] for img_meta in img_metas]
        results = self.head.inference(box_cls, box_pred, image_sizes)
        # processed_results = []
        pred_bboxes = []
        for results_per_image, image_size in zip(results, image_sizes):
            height = image_size[0]
            width = image_size[1]
            r = detector_postprocess(results_per_image, height, width)
            # infomation extract
            pred_box = r.pred_boxes.tensor
            score = r.scores
            pred_class = r.pred_classes
            cur_predict_dict = {
                "boxes":pred_box,
                "scores":score,
                "labels":pred_class
            }
            pred_bboxes.append(cur_predict_dict)
            # processed_results.append({"instances": r})
        pred_masks = None
        return dict(pred_bboxes=pred_bboxes, pred_masks=pred_masks)
