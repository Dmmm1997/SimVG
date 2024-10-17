from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from detrex.layers.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, box_iou
# from detrex.layers.mlp import MLP
from detrex.layers.position_embedding import PositionEmbeddingLearned, PositionEmbeddingSine
from detrex.modeling.matcher.matcher import HungarianMatcher
from detectron2.structures import Boxes, ImageList, Instances
from simvg.models.utils import freeze_params
from .transformer import DetrTransformer, DetrTransformerEncoder, DetrTransformerDecoder
from simvg.models import HEADS
from simvg.core.criterion.distill_criterion import DistillCriterion
from simvg.models.heads.utils import PositionEmbeddingSine1D,MLP
from simvg.core.criterion.criterion import SetCriterion




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
        num_tgqg_layers=1,
        only_decoder=False,
        text_embed_aug=False,
        branch_loss_weight={},
        as_target_query_thr=0.0,
        distill_type="",  # "hard", "hard_weighted", "soft"
        decoder_freeze=False,
        prepare_target_mode="score_weighted",  # "score_weighted", "score_iou_weighted"
        share_predicthead=False,
        num_token_mlp_layers=3,
        mlp_aux_loss=False,
        tgqs_mid_dim=512,
        aux_distill_mode = "klloss", # "klloss" "smoothl1loss"
        text_guided_query_generation = False
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
        assert prepare_target_mode in ["score_weighted", "score_iou_weighted"]
        assert distill_type in ["hard", "hard_weighted", "soft"]
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.input_text_proj = nn.Linear(in_channels, embed_dim)
        self.input_cls_proj = nn.Linear(in_channels, embed_dim)
        self.num_queries = num_queries
        self.text_embed_aug = text_embed_aug
        self.as_target_query_thr = as_target_query_thr
        self.distill_type = distill_type
        self.prepare_target_mode = prepare_target_mode
        self.mlp_aux_loss = mlp_aux_loss
        self.text_guided_query_generation = text_guided_query_generation
        self.num_token_mlp_layers = num_token_mlp_layers
        assert all(x in ["decoder", "token", "distill", "merge", "aux_distill", "balanced_distill"] for x in branch_loss_weight.keys())
        self.branch_loss_weight = branch_loss_weight
        # self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=embed_dim // 2,
            temperature=10000,
            normalize=True,
        )
        self.position_embedding_1d = PositionEmbeddingSine1D(
            num_pos_feats=embed_dim // 2,
            temperature=10000,
            normalize=True,
        )
        self.query_embed = nn.Embedding(num_queries, embed_dim)
            
        # mlp
        if num_token_mlp_layers>0:
            self.mlp = MLP(input_dim=embed_dim, hidden_dim=embed_dim, output_dim=embed_dim, num_layers=num_token_mlp_layers, return_intermediate=True)
        else:
            self.mlp = nn.Identity()

        # define classification head and box head
        if share_predicthead:
            self.class_embed_decoder = nn.Linear(embed_dim, num_classes + 1)
            self.bbox_embed_decoder = MLP(input_dim=embed_dim, hidden_dim=embed_dim, output_dim=4, num_layers=3)
            self.class_embed_token = self.class_embed_decoder
            self.bbox_embed_token = self.bbox_embed_decoder
        else:
            self.class_embed_decoder = nn.Linear(embed_dim, num_classes + 1)
            self.bbox_embed_decoder = MLP(input_dim=embed_dim, hidden_dim=embed_dim, output_dim=4, num_layers=3)
            self.class_embed_token = nn.Linear(embed_dim, num_classes + 1)
            self.bbox_embed_token = MLP(input_dim=embed_dim, hidden_dim=embed_dim, output_dim=4, num_layers=3)
            
        if text_guided_query_generation:
            self.text_guided_query_generation_transformer = DetrTransformerDecoder(
                    embed_dim=embed_dim,
                    num_heads=8,
                    attn_dropout=0.1,
                    feedforward_dim=tgqs_mid_dim,
                    ffn_dropout=0.1,
                    num_layers=num_tgqg_layers,
                    return_intermediate=False,
                    post_norm=True,
                )
        
        self.matcher = HungarianMatcher(
            cost_class=1,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="ce_cost",
        )
        self.criterion = SetCriterion(
            num_classes=num_classes,
            matcher=self.matcher,
            weight_dict={
                "loss_class": 1,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
            },
            loss_class_type="ce_loss",
            eos_coef=0.1,
        )

        if "hard_weighted" in self.distill_type:
            self.criterion_harddistill = SetCriterion(
                num_classes=num_classes,
                matcher=self.matcher,
                weight_dict={
                    "loss_class": 1.0,
                    "loss_bbox": 5.0,
                    "loss_giou": 2.0,
                },
                loss_class_type="weighted_ce_loss",
                eos_coef=0.1,
            )
        elif "soft" in self.distill_type:
            self.criterion_softdistill = DistillCriterion(num_classes=num_classes + 1)
            
        if "aux_distill" in self.branch_loss_weight:
            self.aux_distill_mode = aux_distill_mode
            if self.aux_distill_mode=="klloss":
                self.aux_distill_loss = nn.KLDivLoss(reduction="batchmean")
            elif self.aux_distill_mode=="smoothl1loss":
                self.aux_distill_loss = nn.SmoothL1Loss(reduction="mean")
            else:
                raise TypeError("{} is not suppoert now!!!".format(self.aux_distill_mode))

        if self.aux_loss:
            weight_dict = self.criterion.weight_dict
            aux_weight_dict = {}
            for i in range(self.transformer.decoder.num_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
            self.criterion.weight_dict = weight_dict

        if decoder_freeze:
            self.transformer.eval()
            freeze_params(self.transformer)
            freeze_params(self.input_proj)
            freeze_params(self.text_guided_query_generation_proj)
            freeze_params(self.input_text_proj)
            freeze_params(self.class_embed_decoder)
            freeze_params(self.box)

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

    def prepare_soft_targets(self, targets, decoder_branch_output, img_metas, predict_threahold=0.0, prepare_target_mode="iou_weighted"):
        new_targets_pred = []
        new_targets_gt = []
        # decoder_branch_output_new = deepcopy(decoder_branch_output)
        decoder_pred_logits = decoder_branch_output["pred_logits"].detach().data
        decoder_pred_boxes = decoder_branch_output["pred_boxes"].detach().data
        decoder_scores = F.softmax(decoder_pred_logits, dim=-1)[:, :, 0:1].detach().data
        # decoder_scores = F.softmax(decoder_branch_output["pred_logits"], dim=-1)
        for target_bbox, img_meta in zip(targets, img_metas):
            h, w = img_meta["img_shape"][:2]
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=target_bbox.device)
            if len(target_bbox.shape) == 1:
                target_bbox_ = target_bbox.unsqueeze(0)
                gt_classes = torch.zeros(1, device=target_bbox_.device).long()
            else:  # for grec
                assert int(target_bbox.shape[0]) == len(img_meta["target"])
                gt_classes = []
                target_bbox_ = torch.zeros((0,4),device=target_bbox.device)
                for ind, t in enumerate(img_meta["target"]):
                    if t["category_id"] != -1:
                        gt_classes.append(0)
                        target_bbox_ = torch.concat((target_bbox_, target_bbox[ind:ind+1]))
                gt_classes = torch.tensor(gt_classes, device=target_bbox.device).long()
                # gt_classes = torch.tensor([1 if t["category_id"] == -1 else 0 for t in img_meta["target"]], device=target_bbox.device).long()
                # target_bbox_ = target_bbox
            gt_boxes = target_bbox_.float() / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes).float()
            new_targets_gt.append({"labels": gt_classes, "boxes": gt_boxes})

        # use all predict
        if prepare_target_mode == "score_weighted":
            for predict_bbox, predict_score, img_meta in zip(decoder_pred_boxes, decoder_scores, img_metas):
                mask = predict_score.squeeze(-1) > predict_threahold
                predict_weight = torch.zeros_like(predict_score)
                predict_weight[mask] = predict_score[mask]
                predict_bbox_ = predict_bbox[mask, :]
                if sum(mask) == 0:
                    gt_classes = torch.tensor([], device=target_bbox.device).long()
                else:
                    gt_classes = torch.zeros((predict_bbox_.shape[0]), device=target_bbox.device).long()
                new_targets_pred.append({"labels": gt_classes, "boxes": predict_bbox_, "weight": predict_weight})
        elif prepare_target_mode == "score_iou_weighted":  # iou weighted distill
            new_targets_gt_out = deepcopy(new_targets_gt)
            decoder_pred_boxes = deepcopy(decoder_pred_boxes)
            decoder_scores = deepcopy(decoder_scores)
            indices = self.matcher(decoder_branch_output, new_targets_gt_out)
            for indice, predict_bbox, predict_score, target_gt, img_meta in zip(indices, decoder_pred_boxes, decoder_scores, new_targets_gt_out, img_metas):
                predict_bbox_ = predict_bbox[indice[0]]
                target_gt_ = target_gt["boxes"][indice[1]]
                target_gt_ = torch.cat([target_gt_], dim=0)
                ious = torch.diag(box_iou(box_cxcywh_to_xyxy(predict_bbox_), box_cxcywh_to_xyxy(target_gt_))[0])
                predict_score_ = predict_score[indice[0]].reshape(-1)
                predict_weight = predict_score_ * ious
                if predict_weight.shape[0] == 0:
                    gt_classes = torch.tensor([], device=target_bbox.device).long()
                else:
                    gt_classes = torch.zeros((predict_bbox_.shape[0]), device=target_bbox.device).long()
                new_targets_pred.append({"labels": gt_classes, "boxes": predict_bbox_, "weight": predict_weight})
        else:
            raise TypeError("{} type is not support yet!! you can choose [score_weighted, iou_weighted] types!!!".format(prepare_target_mode))

        return new_targets_gt, new_targets_pred

    def prepare_merge_target(self, targets, decoder_branch_output, img_metas):
        new_targets_merge = []
        new_targets_gt = []
        decoder_scores = F.softmax(decoder_branch_output["pred_logits"], dim=-1)[:, :, 0]
        # decoder_scores = F.softmax(decoder_branch_output["pred_logits"], dim=-1)
        decoder_pred_boxes = decoder_branch_output["pred_boxes"]

        decoder_scores = decoder_scores.detach()
        decoder_pred_boxes = decoder_pred_boxes.detach()

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

        new_targets_gt_out = deepcopy(new_targets_gt)
        decoder_pred_boxes = deepcopy(decoder_pred_boxes)
        decoder_scores = deepcopy(decoder_scores)
        indices = self.matcher(decoder_branch_output, new_targets_gt_out)
        for indice, predict_bbox, predict_score, target_gt, img_meta in zip(indices, decoder_pred_boxes, decoder_scores, new_targets_gt_out, img_metas):
            predict_bbox_ = predict_bbox[indice[0]]
            target_gt_boxes = target_gt["boxes"][indice[1]]
            target_gt_labels = target_gt["labels"][indice[1]]
            target_gt_boxes = torch.cat([target_gt_boxes], dim=0)
            ious = torch.diag(box_iou(box_cxcywh_to_xyxy(predict_bbox_), box_cxcywh_to_xyxy(target_gt_boxes))[0])
            predict_score_ = predict_score[indice[0]]
            predict_weight = predict_score_ * ious
            merged_labels = torch.cat((target_gt_labels, gt_classes), dim=0)
            merged_bboxes = torch.cat((target_gt_boxes, predict_bbox_), dim=0)
            merged_weights = torch.cat((torch.ones(target_gt_labels.shape[0], device=predict_weight.device), predict_weight), dim=0)
            if predict_weight.shape[0] == 0:
                gt_classes = torch.tensor([], device=target_bbox.device).long()
            else:
                gt_classes = torch.zeros((predict_bbox_.shape[0]), device=target_bbox.device).long()
            new_targets_merge.append({"labels": merged_labels, "boxes": merged_bboxes, "weight": merged_weights})

        return new_targets_gt, new_targets_merge

    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def x_mask_pos_enc(self, x, img_metas):
        batch_size = x.size(0)
        try:
            input_img_h, input_img_w = img_metas[0]["batch_input_shape"]
        except:
            input_img_h, input_img_w, _ = img_metas[0]["img_shape"]
        x_mask = x.new_ones((batch_size, input_img_h, input_img_w))
        # CAUTION: do not support random flipping
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]["img_shape"]
            x_mask[img_id, :img_h, :img_w] = 0

        x_mask = F.interpolate(x_mask.unsqueeze(1), size=x.size()[-2:]).to(torch.bool).squeeze(1)

        x_pos_embeds = self.position_embedding(x_mask)

        return x_mask, x_pos_embeds

    def calc_loss(self, output_class, output_coord, targets):
        output = {"pred_logits": output_class[-1], "pred_boxes": output_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(output_class, output_coord)

        loss_dict = self.criterion(output, targets)
        weight_dict = self.criterion.weight_dict
        for k in loss_dict.keys():
            if k in weight_dict:
                loss_dict[k] *= weight_dict[k]
        return loss_dict

    def calc_distill_loss(self, all_cls_scores, all_bbox_preds, img_metas, teacher_bboxes, teacher_labels, targets=None, distill_type="soft"):
        if distill_type == "soft":
            teacher_bboxes = teacher_bboxes[-1:]
            teacher_labels = teacher_labels[-1:]
            loss_dict = self.criterion_softdistill.forward_train_distill(all_cls_scores, all_bbox_preds, img_metas, teacher_bboxes, teacher_labels)
        elif distill_type == "hard_weighted":
            output = {"pred_logits": all_cls_scores[-1], "pred_boxes": all_bbox_preds[-1]}
            if self.aux_loss:
                output["aux_outputs"] = self._set_aux_loss(all_cls_scores, all_bbox_preds)
            loss_dict = self.criterion_harddistill(output, targets)
        elif distill_type == "hard":
            output = {"pred_logits": all_cls_scores[-1], "pred_boxes": all_bbox_preds[-1]}
            if self.aux_loss:
                output["aux_outputs"] = self._set_aux_loss(all_cls_scores, all_bbox_preds)
            loss_dict = self.criterion(output, targets)
        else:
            raise TypeError("The distill type is not correct!!!")
        weight_dict = self.criterion.weight_dict
        for k in loss_dict.keys():
            if k in weight_dict:
                loss_dict[k] *= weight_dict[k]
        return loss_dict

    def forward_general(self, x_mm, img_metas, cls_feat=None, text_feat=None, text_mask=None):
        # feature proj to embed channels
        x_mm = self.input_proj(x_mm)
        text_feat = self.input_text_proj(text_feat)
        cls_feat = self.input_cls_proj(cls_feat).unsqueeze(1)
        img_masks, pos_embed = self.x_mask_pos_enc(x_mm, img_metas)  # TODO: fix the img mask
        
        # extend the the query number of cls_token
        cls_feat = cls_feat.repeat((1, self.num_queries, 1))
        # text guided query generation
        if self.text_guided_query_generation:
            text_feat_filter = torch.cat(list(map(lambda feat, mask: torch.max(feat[mask, :], dim=0, keepdim=True)[0], text_feat, ~text_mask))).unsqueeze(1).repeat(1,self.num_queries,1)
            query_embed_input = self.query_embed.weight.unsqueeze(0).repeat(x_mm.shape[0],1,1).transpose(0,1)
            target = torch.zeros_like(query_embed_input)
            text_pos_embed = self.position_embedding_1d(text_feat).unsqueeze(0).repeat(text_feat.shape[0],1,1).permute(1,0,2).cuda()
            text_feat_input = text_feat.transpose(0,1)
            query_embed = self.text_guided_query_generation_transformer(
                query=target,
                key=text_feat_input,
                value=text_feat_input,
                key_pos=text_pos_embed,
                query_pos=query_embed_input,
                key_padding_mask=text_mask.bool())
            query_embed = query_embed[0].transpose(0,1) + text_feat_filter + query_embed_input.transpose(0,1)
            cls_feat = query_embed + cls_feat
        else:
            query_embed = self.query_embed.weight.unsqueeze(0).repeat(x_mm.shape[0],1,1)
            
        if "decoder" in self.branch_loss_weight and len(self.branch_loss_weight)==1:
            token_branch_output = {
                "pred_logits": None,
                "pred_boxes": None,
            }
            outputs_class_token_branch=None
            outputs_coord_token_branch=None
        else:
            cls_feat = self.mlp(cls_feat)
            if self.num_token_mlp_layers==0:
                cls_feat = cls_feat.unsqueeze(0)
            # cls_feat branch
            outputs_class_token_branch = self.class_embed_token(cls_feat)
            outputs_coord_token_branch = self.bbox_embed_token(cls_feat).sigmoid()
            token_branch_output = {
                "pred_logits": outputs_class_token_branch[-1],
                "pred_boxes": outputs_coord_token_branch[-1],
            }

        only_token=False
        if not only_token:
            # decoder
            hidden_states, _ = self.transformer(x_mm, img_masks, query_embed, pos_embed)

            outputs_class_decoder_branch = self.class_embed_decoder(hidden_states)
            outputs_coord_decoder_branch = self.bbox_embed_decoder(hidden_states).sigmoid()
            
            decoder_branch_output = {
                "pred_logits": outputs_class_decoder_branch[-1],
                "pred_boxes": outputs_coord_decoder_branch[-1],
            }
        else:
            hidden_states = None
            outputs_class_decoder_branch = None
            outputs_coord_decoder_branch = None
            decoder_branch_output = {
                "pred_logits": None,
                "pred_boxes": None,
            }
        
        output = {
            "token_branch_output": token_branch_output,
            "decoder_branch_output": decoder_branch_output,
            "outputs_class_decoder_branch": outputs_class_decoder_branch,
            "outputs_coord_decoder_branch": outputs_coord_decoder_branch,
            "outputs_class_token_branch": outputs_class_token_branch,
            "outputs_coord_token_branch": outputs_coord_token_branch,
            "token_features": cls_feat,
            "decoder_features": hidden_states,
        }

        return output

    def forward_train(self, x_mm, img_metas, cls_feat=None, text_feat=None, gt_bbox=None, text_mask=None):
        device = x_mm.device
        output = self.forward_general(x_mm, img_metas, cls_feat=cls_feat, text_feat=text_feat, text_mask=text_mask)

        outputs_class_decoder_branch = output["outputs_class_decoder_branch"]
        outputs_coord_decoder_branch = output["outputs_coord_decoder_branch"]
        outputs_class_token_branch = output["outputs_class_token_branch"]
        outputs_coord_token_branch = output["outputs_coord_token_branch"]
        token_branch_output = output["token_branch_output"]
        decoder_branch_output = output["decoder_branch_output"]
        token_features = output["token_features"]
        decoder_features = output["decoder_features"]

        # prepare the targets
        # targets_decoder_branch = self.prepare_targets(gt_bbox, img_metas)
        targets_gt, targets_predict = self.prepare_soft_targets(
            gt_bbox, decoder_branch_output, img_metas, predict_threahold=self.as_target_query_thr, prepare_target_mode=self.prepare_target_mode
        )

        loss_decoder_branch, loss_token_branch, loss_kd_branch, loss_merge_branch, loss_aux_distill_branch = (
            torch.tensor(0, device=device).float(),
            torch.tensor(0, device=device).float(),
            torch.tensor(0, device=device).float(),
            torch.tensor(0, device=device).float(),
            torch.tensor(0, device=device).float(),
        )
        loss_dict = {}
        if "decoder" in self.branch_loss_weight:
            loss_dict_decoder_branch = self.calc_loss(outputs_class_decoder_branch, outputs_coord_decoder_branch, targets_gt)
            loss_decoder_branch = sum(loss_dict_decoder_branch.values())
            loss_decoder_branch = self.branch_loss_weight["decoder"] * loss_decoder_branch
            loss_dict["loss_dgt"] = loss_decoder_branch
            
        if "balanced_distill" in self.branch_loss_weight:
                
            weights_distill = torch.mean(torch.cat([t["weight"] for t in targets_predict]))
            
            if not self.mlp_aux_loss and len(outputs_class_token_branch.shape)==4:
                outputs_class_token_branch_ = outputs_class_token_branch[-1:]
                outputs_coord_token_branch_ = outputs_coord_token_branch[-1:]
            else:
                outputs_class_token_branch_ = outputs_class_token_branch
                outputs_coord_token_branch_ = outputs_coord_token_branch
            loss_dict_token_branch_gt = self.calc_loss(outputs_class_token_branch_, outputs_coord_token_branch_, targets_gt)
            loss_token_branch = sum(loss_dict_token_branch_gt.values())
            loss_token_branch = self.branch_loss_weight["balanced_distill"]["token"] * loss_token_branch * (1-weights_distill)
            loss_dict["loss_tgt"] = loss_token_branch
            
            loss_dict_token_branch_predict = self.calc_loss(outputs_class_token_branch_,outputs_coord_token_branch_,targets_predict)
            loss_kd_branch = sum(loss_dict_token_branch_predict.values())
            loss_kd_branch = self.branch_loss_weight["balanced_distill"]["distill"] * loss_kd_branch * weights_distill
            loss_dict["loss_kd"] = loss_kd_branch
            
            loss_dict["loss_distill_w"] = weights_distill
        else:
            if "token" in self.branch_loss_weight:
                # if mlp_aux_loss is False, select the last mlp output as the input of the loss
                if not self.mlp_aux_loss and len(outputs_class_token_branch.shape)==4:
                    outputs_class_token_branch_ = outputs_class_token_branch[-1:]
                    outputs_coord_token_branch_ = outputs_coord_token_branch[-1:]
                else:
                    outputs_class_token_branch_ = outputs_class_token_branch
                    outputs_coord_token_branch_ = outputs_coord_token_branch
                loss_dict_token_branch_gt = self.calc_loss(outputs_class_token_branch_, outputs_coord_token_branch_, targets_gt)
                loss_token_branch = sum(loss_dict_token_branch_gt.values())
                loss_token_branch = self.branch_loss_weight["token"] * loss_token_branch
                loss_dict["loss_tgt"] = loss_token_branch
                
            if "distill" in self.branch_loss_weight:
                # if mlp_aux_loss is False, select the last mlp output as the input of the loss
                if not self.mlp_aux_loss:
                    outputs_class_token_branch_ = outputs_class_token_branch[-1:]
                    outputs_coord_token_branch_ = outputs_coord_token_branch[-1:]
                else:
                    outputs_class_token_branch_ = outputs_class_token_branch
                    outputs_coord_token_branch_ = outputs_coord_token_branch
                loss_dict_token_branch_predict = self.calc_distill_loss(
                    outputs_class_token_branch_,
                    outputs_coord_token_branch_,
                    img_metas,
                    outputs_coord_decoder_branch,
                    outputs_class_decoder_branch,
                    targets=targets_predict,
                    distill_type=self.distill_type,
                )
                loss_kd_branch = sum(loss_dict_token_branch_predict.values())
                loss_kd_branch = self.branch_loss_weight["distill"] * loss_kd_branch
                loss_dict["loss_kd"] = loss_kd_branch
        if "merge" in self.branch_loss_weight:
            targets_gt, targets_merge = self.prepare_merge_target(gt_bbox, decoder_branch_output, img_metas)
            loss_dict_merge_branch_gt = self.calc_loss(outputs_class_token_branch, outputs_coord_token_branch, targets_merge)
            loss_merge_branch = sum(loss_dict_merge_branch_gt.values())
            loss_merge_branch = self.branch_loss_weight["merge"] * loss_merge_branch
            loss_dict["loss_merge"] = loss_merge_branch
        if "aux_distill" in self.branch_loss_weight:
            assert token_features.shape == decoder_features.shape, "the number of the mlp layers and ecoder layers must be the same!!"
            decoder_output = []
            token_output = []
            for decoder_class, decoder_coord, token_class, token_coord in zip(outputs_class_decoder_branch[:-1], outputs_coord_decoder_branch[:-1],outputs_class_token_branch[:-1], outputs_coord_token_branch[:-1]):
                decoder_output.append({"pred_logits":decoder_class,"pred_boxes":decoder_coord})
                token_output.append({"pred_logits":token_class,"pred_boxes":token_coord})
            for decoder_o, token_o in zip(decoder_output, token_output):
                targets_gt_aux, targets_predict_aux = self.prepare_soft_targets(gt_bbox, decoder_o, img_metas, predict_threahold=self.as_target_query_thr, prepare_target_mode=self.prepare_target_mode)
                loss_dict_aux_distill_branch_gt = self.calc_distill_loss(
                    token_o["pred_logits"].unsqueeze(0),
                    token_o["pred_boxes"].unsqueeze(0),
                    img_metas,
                    decoder_o["pred_logits"].unsqueeze(0),
                    decoder_o["pred_boxes"].unsqueeze(0),
                    targets=targets_predict_aux,
                    distill_type=self.distill_type,
                )
                loss_aux_distill_branch += sum(loss_dict_aux_distill_branch_gt.values())
            loss_dict["aux_distill"] = self.branch_loss_weight["aux_distill"] * loss_aux_distill_branch

        loss_dict["loss_total"] = loss_decoder_branch + loss_token_branch + loss_kd_branch + loss_merge_branch + loss_aux_distill_branch
        return loss_dict, output

    def forward_test(self, x_mm, img_metas, text_feat=None, cls_feat=None, with_bbox=False, with_mask=False, text_mask=None):
        return self.forward_general(x_mm, img_metas, text_feat=text_feat, cls_feat=cls_feat, text_mask=text_mask)

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
