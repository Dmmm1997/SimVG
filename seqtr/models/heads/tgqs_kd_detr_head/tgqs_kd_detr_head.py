from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.layers.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.layers.mlp import MLP
from detrex.modeling.criterion.criterion import SetCriterion
from detrex.layers.position_embedding import PositionEmbeddingLearned, PositionEmbeddingSine
from detrex.modeling.matcher.matcher import HungarianMatcher
from detectron2.structures import Boxes, ImageList, Instances

from .transformer import DetrTransformer, DetrTransformerEncoder, DetrTransformerDecoder
from seqtr.models import HEADS


@HEADS.register_module()
class TextGuidedQuerySelectKDDETRHead(nn.Module):
    def __init__(
        self,
        num_queries=100,
        in_channels=768,
        text_max_token=20,
        embed_dim=256,
        num_classes=1,
        aux_loss=True,
        num_encoder_layers=6,
        num_decoder_layers=6,
        only_decoder=False,
        text_embed_aug=False,
        target_type=["decoder_gt"],
        branch_loss_weight={"decoder_gt": 1.0, "decoder_predict": 0.5, "token_gt": 0.5},
    ):
        super(TextGuidedQuerySelectKDDETRHead, self).__init__()
        self.transformer = DetrTransformer(
            encoder=DetrTransformerEncoder(
                embed_dim=embed_dim,
                num_heads=8,
                attn_dropout=0.1,
                feedforward_dim=2048,
                ffn_dropout=0.1,
                num_layers=num_encoder_layers,
                post_norm=False,
            ),
            decoder=DetrTransformerDecoder(
                embed_dim=embed_dim,
                num_heads=8,
                attn_dropout=0.1,
                feedforward_dim=2048,
                ffn_dropout=0.1,
                num_layers=num_decoder_layers,
                return_intermediate=True,
                post_norm=True,
            ),
            only_decoder=only_decoder,
        )
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.input_text_proj = nn.Linear(in_channels, embed_dim)
        self.input_cls_proj = nn.Linear(in_channels, embed_dim)
        self.text_guided_query_generation_proj = nn.Linear(text_max_token, num_queries)
        self.num_queries = num_queries
        self.text_embed_aug = text_embed_aug
        self.target_type = target_type
        assert all(x in ["decoder_gt", "token_gt", "decoder_predict"] for x in target_type)
        self.branch_loss_weight = branch_loss_weight
        # self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=embed_dim // 2,
            temperature=10000,
            normalize=True,
        )

        # define classification head and box head
        self.class_embed_decoder = nn.Linear(embed_dim, num_classes + 1)
        self.bbox_embed_decoder = MLP(input_dim=embed_dim, hidden_dim=embed_dim, output_dim=4, num_layers=3)

        self.class_embed_token = nn.Linear(embed_dim, num_classes + 1)
        self.bbox_embed_token = MLP(input_dim=embed_dim, hidden_dim=embed_dim, output_dim=4, num_layers=3)

        matcher = HungarianMatcher(
            cost_class=1,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="ce_cost",
        )
        self.criterion = SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict={
                "loss_class": 1,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
            },
            loss_class_type="ce_loss",
            eos_coef=0.1,
        )

        self.criterion_predict = SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict={
                "loss_class": 1,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
            },
            loss_class_type="weighted_ce_loss",
            eos_coef=0.1,
        )

        if self.aux_loss:
            weight_dict = self.criterion.weight_dict
            aux_weight_dict = {}
            for i in range(self.transformer.decoder.num_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
            self.criterion.weight_dict = weight_dict

    def prepare_targets(self, targets, img_metas):
        new_targets = []
        for target_bbox, img_meta in zip(targets, img_metas):
            h, w = img_meta["img_shape"][:2]
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=target_bbox.device)
            if len(target_bbox.shape) == 1:
                target_bbox = target_bbox.unsqueeze(0)
                gt_classes = torch.zeros(1, device=target_bbox.device).long()
            else:  # for grec # TODO None object can be set as label 1 ? or just set no GT
                assert int(target_bbox.shape[0]) == len(img_meta["target"])
                gt_classes = torch.tensor([1 if t["category_id"] == -1 else 0 for t in img_meta["target"]], device=target_bbox.device).long()
            gt_boxes = target_bbox.float() / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        return new_targets

    def prepare_soft_targets(self, targets, decoder_branch_output, img_metas, predict_threahold=0.0):
        new_targets_pred = []
        new_targets_gt = []
        decoder_scores = F.softmax(decoder_branch_output["pred_logits"], dim=-1)[:, :, 0:1]
        decoder_pred_boxes = decoder_branch_output["pred_boxes"]

        for target_bbox, img_meta in zip(targets, img_metas):
            h, w = img_meta["img_shape"][:2]
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=target_bbox.device)
            if len(target_bbox.shape) == 1:
                target_bbox = target_bbox.unsqueeze(0)
                gt_classes = torch.zeros(1, device=target_bbox.device).long()
            else:  # for grec
                assert int(target_bbox.shape[0]) == len(img_meta["target"])
                gt_classes = torch.tensor([1 if t["category_id"] == -1 else 0 for t in img_meta["target"]], device=target_bbox.device).long()
            gt_boxes = target_bbox.float() / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets_gt.append({"labels": gt_classes, "boxes": gt_boxes})

        for predict_bbox, predict_score, img_meta in zip(decoder_pred_boxes, decoder_scores, img_metas):
            mask = predict_score.squeeze(-1) > predict_threahold
            predict_weight = torch.zeros_like(predict_score)
            predict_weight[mask] = predict_score[mask]
            predict_bbox = predict_bbox[mask, :]
            if sum(mask) == 0:
                gt_classes = torch.tensor([], device=target_bbox.device).long()
            else:
                gt_classes = torch.zeros((predict_bbox.shape[0]), device=target_bbox.device).long()
            new_targets_pred.append({"labels": gt_classes, "boxes": predict_bbox, "weight": predict_weight})

        return new_targets_gt, new_targets_pred

    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def x_mask_pos_enc(self, x, img_metas):
        batch_size = x.size(0)
        input_img_h, input_img_w = img_metas[0]["batch_input_shape"]
        x_mask = x.new_ones((batch_size, input_img_h, input_img_w))
        # CAUTION: do not support random flipping
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]["img_shape"]
            x_mask[img_id, :img_h, :img_w] = 0

        x_mask = F.interpolate(x_mask.unsqueeze(1), size=x.size()[-2:]).to(torch.bool).squeeze(1)

        x_pos_embeds = self.position_embedding(x_mask)

        # x_mask = x_mask.view(batch_size, -1)
        # x_pos_embeds = x_pos_embeds.view(
        #     batch_size, self.d_model, -1).transpose(1, 2)

        return x_mask, x_pos_embeds

    def calc_loss(self, output_class, output_coord, targets, type_loss="gt"):
        output = {"pred_logits": output_class[-1], "pred_boxes": output_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(output_class, output_coord)

        if type_loss == "predict":
            loss_dict = self.criterion_predict(output, targets)
        else:
            loss_dict = self.criterion(output, targets)
        weight_dict = self.criterion.weight_dict
        for k in loss_dict.keys():
            if k in weight_dict:
                loss_dict[k] *= weight_dict[k]
        return loss_dict

    def forward_general(self, x_mm, img_metas, cls_feat=None, text_feat=None):
        # feature proj to embed channels
        x_mm = self.input_proj(x_mm)
        text_feat = self.input_text_proj(text_feat)
        cls_feat = self.input_cls_proj(cls_feat).unsqueeze(1)
        img_masks, pos_embed = self.x_mask_pos_enc(x_mm, img_metas)  # TODO: fix the img mask

        # text guided query generation
        query_embed = self.text_guided_query_generation_proj(text_feat.transpose(1, 2)).transpose(1, 2)

        # extend the token_branch query numbers
        if self.text_embed_aug:
            cls_feat = cls_feat.repeat((1, self.num_queries, 1)) + query_embed
        else:
            cls_feat = cls_feat.repeat((1, self.num_queries, 1))

        # decoder
        hidden_states, _ = self.transformer(x_mm, img_masks, query_embed, pos_embed)

        # cls_feat branch
        outputs_class_token_branch = self.class_embed_token(cls_feat).unsqueeze(0)
        outputs_coord_token_branch = self.bbox_embed_token(cls_feat).sigmoid().unsqueeze(0)

        outputs_class_decoder_branch = self.class_embed_decoder(hidden_states)
        outputs_coord_decoder_branch = self.bbox_embed_decoder(hidden_states).sigmoid()

        token_branch_output = {
            "pred_logits": outputs_class_token_branch[-1],
            "pred_boxes": outputs_coord_token_branch[-1],
        }
        decoder_branch_output = {
            "pred_logits": outputs_class_decoder_branch[-1],
            "pred_boxes": outputs_coord_decoder_branch[-1],
        }
        output = {
            "token_branch_output": token_branch_output,
            "decoder_branch_output": decoder_branch_output,
            "outputs_class_decoder_branch": outputs_class_decoder_branch,
            "outputs_coord_decoder_branch": outputs_coord_decoder_branch,
            "outputs_class_token_branch": outputs_class_token_branch,
            "outputs_coord_token_branch": outputs_coord_token_branch,
        }

        return output

    def forward_train(self, x_mm, img_metas, cls_feat=None, text_feat=None, gt_bbox=None, gt_mask_vertices=None):
        device = x_mm.device
        output = self.forward_general(x_mm, img_metas, cls_feat=cls_feat, text_feat=text_feat)

        outputs_class_decoder_branch = output["outputs_class_decoder_branch"]
        outputs_coord_decoder_branch = output["outputs_coord_decoder_branch"]
        outputs_class_token_branch = output["outputs_class_token_branch"]
        outputs_coord_token_branch = output["outputs_coord_token_branch"]
        token_branch_output = output["token_branch_output"]
        decoder_branch_output = output["decoder_branch_output"]

        # prepare the targets
        targets_decoder_branch = self.prepare_targets(gt_bbox, img_metas)
        targets_gt, targets_predict = self.prepare_soft_targets(gt_bbox, decoder_branch_output, img_metas)

        loss_decoder_branch, loss_token_branch_gt, loss_token_branch_predict = (
            torch.tensor(0, device=device),
            torch.tensor(0, device=device),
            torch.tensor(0, device=device),
        )
        if "decoder_gt" in self.target_type:
            loss_dict_decoder_branch = self.calc_loss(outputs_class_decoder_branch, outputs_coord_decoder_branch, targets_decoder_branch)
            loss_decoder_branch = sum(loss_dict_decoder_branch.values())
        if "token_gt" in self.target_type:
            loss_dict_token_branch_gt = self.calc_loss(outputs_class_token_branch, outputs_coord_token_branch, targets_gt)
            loss_token_branch_gt = sum(loss_dict_token_branch_gt.values())
        if "decoder_predict" in self.target_type:
            loss_dict_token_branch_predict = self.calc_loss(outputs_class_token_branch, outputs_coord_token_branch, targets_predict, type_loss="predict")
            loss_token_branch_predict = sum(loss_dict_token_branch_predict.values())
        loss_dict = {}
        loss_dict["loss_det"] = (
            self.branch_loss_weight["decoder_gt"] * loss_decoder_branch
            + self.branch_loss_weight["token_gt"] * loss_token_branch_gt
            + self.branch_loss_weight["decoder_predict"] * loss_token_branch_predict
        )
        return loss_dict, output

        # proj_queries = F.normalize(self.contrastive_align_projection_image(logits), p=2, dim=-1)
        # proj_tokens = F.normalize(
        #     self.contrastive_align_projection_text(memory_cache["text_memory"]).transpose(0, 1),
        #     p=2,
        #     dim=-1,
        # )
        # out.update(
        #     {
        #         "proj_queries": proj_queries[-1],
        #         "proj_tokens": proj_tokens,
        #         "tokenized": memory_cache["tokenized"],
        #     }
        # )
        # assert proj_tokens is not None and proj_queries is not None
        # out["aux_outputs"] = [
        #     {
        #         "pred_logits": a,
        #         "pred_boxes": b,
        #         "proj_queries": c,
        #         "proj_tokens": proj_tokens,
        #         "tokenized": memory_cache["tokenized"],
        #     }
        #     for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], proj_queries[:-1])
        # ]

        # loss_dict = {}
        # if self.criterion is not None:
        #     loss_dict.update(self.criterion(out, targets, positive_map))

        # loss_ce = self.loss(logits, targets, with_bbox=with_bbox, with_mask=with_mask)

    def forward_test(self, x_mm, img_metas, text_feat=None, cls_feat=None, with_bbox=False, with_mask=False):
        return self.forward_general(x_mm, img_metas, text_feat=text_feat, cls_feat=cls_feat)

    def inference(self, box_cls, box_pred, image_sizes):
        """Inference function for DETR

        Args:
            box_cls (torch.Tensor): tensor of shape ``(batch_size, num_queries, K)``.
                The tensor predicts the classification probability for each query.
            box_pred (torch.Tensor): tensors of shape ``(batch_size, num_queries, 4)``.
                The tensor predicts 4-vector ``(x, y, w, h)`` box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # For each box we assign the best class or the second best if the best on is `no_object`.
        scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(scores, labels, box_pred, image_sizes)):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results