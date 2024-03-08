import torch
import random
import torch.nn as nn
import torch.nn.functional as f

from mmdet.models.losses import CrossEntropyLoss
from mmdet.models.utils import build_transformer

from seqtr.models import HEADS
from seqtr.core.layers import LinearModule
from seqtr.core.losses import LabelSmoothCrossEntropyLoss
from seqtr.core.layers.mdetr_transformer import MLP
from torch.nn import functional as F
from seqtr.core.matcher import HungarianMatcher


@HEADS.register_module()
class MDETRHead(nn.Module):
    def __init__(
        self,
        in_ch=1024,
        num_bin=1000,
        multi_task=False,
        num_query=100,
        shuffle_fraction=-1,
        mapping="relative",
        top_p=-1,
        num_ray=18,
        det_coord=[0],
        det_coord_weight=1.5,
        predictor=dict(
            num_fcs=3,
            in_chs=[256, 256, 256],
            out_chs=[256, 256, 1001],
            fc=[
                dict(
                    linear=dict(type="Linear", bias=True),
                    act=dict(type="ReLU", inplace=True),
                    drop=None,
                ),
                dict(
                    linear=dict(type="Linear", bias=True),
                    act=dict(type="ReLU", inplace=True),
                    drop=None,
                ),
                dict(linear=dict(type="Linear", bias=True), act=None, drop=None),
            ],
        ),
        transformer=dict(
            type="AutoRegressiveTransformer",
            encoder=dict(
                num_layers=6,
                layer=dict(
                    d_model=256,
                    nhead=8,
                    dim_feedforward=1024,
                    dropout=0.1,
                    activation="relu",
                    batch_first=True,
                ),
            ),
            decoder=dict(
                num_layers=3,
                layer=dict(
                    d_model=256,
                    nhead=8,
                    dim_feedforward=1024,
                    dropout=0.1,
                    activation="relu",
                    batch_first=True,
                ),
            ),
        ),
        x_positional_encoding=dict(
            type="SinePositionalEncoding2D", num_feature=128, normalize=True
        ),
        query_positional_encoding=dict(
            type="LearnedPositionalEncoding1D", num_embedding=100, num_feature=256
        ),
        matcher=dict(set_cost_class=1, set_cost_bbox=5, set_cost_giou=2),
        #  losses=["labels", "boxes", "cardinality","contrastive_align"],
        losses=["labels", "boxes"],
        loss_coef=dict(loss_ce=1, loss_bbox=5, loss_giou=2, loss_contrastive_align=1),
        temperature_NCE=0.07,
    ):
        super(MDETRHead, self).__init__()
        self.num_bin = num_bin
        self.multi_task = multi_task
        self.shuffle_fraction = shuffle_fraction
        assert mapping in ["relative", "absolute"]
        self.mapping = mapping
        self.top_p = top_p
        self.num_ray = num_ray
        self.det_coord = det_coord
        self.det_coord_weight = det_coord_weight

        self.transformer = build_transformer(transformer)
        self.d_model = self.transformer.d_model
        self.query_embed = nn.Embedding(num_query, self.d_model)
        self.bbox_embed = MLP(self.d_model, self.d_model, 4, 3)
        self.class_embed = nn.Linear(self.d_model, 1000 + 1)
        self.matcher = HungarianMatcher(
            cost_class=matcher["set_cost_class"],
            cost_bbox=matcher["set_cost_bbox"],
            cost_giou=matcher["set_cost_giou"],
        )
        self.losses = losses

        self._init_layers(
            in_ch,
            predictor,
            multi_task,
            x_positional_encoding,
            query_positional_encoding,
        )
        self.loss_coef = loss_coef
        num_classes = 255
        self.criterion = SetCriterion(
            num_classes,
            matcher=self.matcher,
            eos_coef=self.loss_coef,
            losses=self.losses,
            temperature=temperature_NCE,
        )

        # loss_type = loss.pop('type')
        # if loss_type == "CrossEntropyLoss":
        #     self.loss_ce = CrossEntropyLoss()
        # elif loss_type == "LabelSmoothCrossEntropyLoss":
        #     self.loss_ce = LabelSmoothCrossEntropyLoss(
        #         neg_factor=loss.pop('neg_factor', 0.1))

    def _init_layers(
        self,
        in_ch,
        predictor_cfg,
        multi_task,
        x_positional_encoding,
        seq_positional_encoding,
    ):
        num_fcs = predictor_cfg.pop("num_fcs")
        in_chs, out_chs = predictor_cfg.pop("in_chs"), predictor_cfg.pop("out_chs")
        fc_cfg = predictor_cfg.pop("fc")
        assert num_fcs == len(fc_cfg) == len(in_chs) == len(out_chs)
        predictor = []
        for i in range(num_fcs):
            _cfg = fc_cfg[i]
            _cfg["linear"]["in_features"] = in_chs[i]
            _cfg["linear"]["out_features"] = out_chs[i]
            predictor.append(LinearModule(**_cfg))
            if i == num_fcs - 1:
                self.vocab_size = out_chs[i]
        assert self.vocab_size == self.num_bin + 1
        self.end = self.vocab_size - 1
        self.predictor = nn.Sequential(*predictor)

        if multi_task:
            # bbox_token, x1, y1, x2, y2, mask_token, x1, y1, ..., xN, yN
            self.task_embedding = nn.Embedding(2, self.d_model)

        self.transformer._init_layers(
            in_ch, self.vocab_size, x_positional_encoding, seq_positional_encoding
        )

    def shuffle_sequence(self, seq):
        batch_size, num_pts = seq.size(0), seq.size(1) // 2
        seq = seq.reshape(batch_size * num_pts, 2)
        shuffle_idx = random.sample(
            range(batch_size), int(batch_size * self.shuffle_fraction)
        )
        shuffle_idx = [idx * num_pts for idx in shuffle_idx]
        perm = torch.randperm(num_pts, device=seq.device)
        for idx in shuffle_idx:
            s = idx
            e = s + num_pts
            seq[s:e, :] = seq[s:e, :][perm]
        seq = seq.reshape(batch_size, num_pts * 2)
        return seq

    def forward_train(
        self,
        x_mm,
        img_metas,
        gt_bbox=None,
        gt_mask_vertices=None,
    ):
        """Args:
        x_mm (tensor): [batch_size, c, h, w].

        img_metas (list[dict]): list of image info dict where each dict
            has: 'img_shape', 'scale_factor', and may also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys see
            `seqtr/datasets/pipelines/formatting.py:CollectData`.

        gt_bbox (list[tensor]): [4, ], [tl_x, tl_y, br_x, br_y] format,
            and the coordinates are in 'img_shape' scale.

        gt_mask_vertices (list[tensor]): [batch_size, 2, num_ray], padded values are -1,
            the coordinates are in 'pad_shape' scale.
        """
        with_bbox = gt_bbox is not None
        with_mask = gt_mask_vertices is not None

        x_mask, x_pos_embeds = self.transformer.x_mask_pos_enc(x_mm, img_metas)

        memory_cache = self.transformer.forward_encoder(x_mm, x_mask, x_pos_embeds)

        query_embed = self.query_embed.weight

        out = dict()
        logits = self.transformer.forward_decoder(query_embed, memory_cache)
        # logits = self.predictor(logits)
        outputs_class = self.class_embed(logits)
        outputs_coord = self.bbox_embed(logits).sigmoid()
        out.update(
            {
                "pred_logits": outputs_class[-1],
                "pred_boxes": outputs_coord[-1],
            }
        )

        proj_queries = F.normalize(
            self.contrastive_align_projection_image(logits), p=2, dim=-1
        )
        proj_tokens = F.normalize(
            self.contrastive_align_projection_text(
                memory_cache["text_memory"]
            ).transpose(0, 1),
            p=2,
            dim=-1,
        )
        out.update(
            {
                "proj_queries": proj_queries[-1],
                "proj_tokens": proj_tokens,
                "tokenized": memory_cache["tokenized"],
            }
        )
        assert proj_tokens is not None and proj_queries is not None
        out["aux_outputs"] = [
            {
                "pred_logits": a,
                "pred_boxes": b,
                "proj_queries": c,
                "proj_tokens": proj_tokens,
                "tokenized": memory_cache["tokenized"],
            }
            for a, b, c in zip(
                outputs_class[:-1], outputs_coord[:-1], proj_queries[:-1]
            )
        ]

        loss_dict = {}
        if self.criterion is not None:
            loss_dict.update(self.criterion(out, targets, positive_map))

        loss_ce = self.loss(logits, targets, with_bbox=with_bbox, with_mask=with_mask)

        # training statistics
        with torch.no_grad():
            if with_mask and with_bbox:
                logits_bbox = logits[:, :4, :-1]
                scores_bbox = f.softmax(logits_bbox, dim=-1)
                _, seq_out_bbox = scores_bbox.max(dim=-1, keepdim=False)
                logits_mask = logits[:, 5:, :]
                scores_mask = f.softmax(logits_mask, dim=-1)
                _, seq_out_mask = scores_mask.max(dim=-1, keepdim=False)
                return dict(loss_multi_task=loss_ce), dict(
                    seq_out_bbox=seq_out_bbox.detach(),
                    seq_out_mask=seq_out_mask.detach(),
                )
            else:
                if with_bbox:
                    logits = logits[:, :-1, :-1]
                scores = f.softmax(logits, dim=-1)
                _, seq_out = scores.max(dim=-1, keepdim=False)

                if with_bbox:
                    return dict(loss_det=loss_ce), dict(seq_out_bbox=seq_out.detach())
                elif with_mask:
                    return dict(loss_mask=loss_ce), dict(seq_out_mask=seq_out.detach())

    def loss(self, logits, targets, with_bbox=False, with_mask=False):
        """Args:
        logits (tensor): [batch_size, 1+4 or 1+2*num_ray, vocab_size].

        target (tensor): [batch_size, 1+4 or 1+2*num_ray].
        """
        batch_size, num_token = logits.size()[:2]

        if with_bbox and with_mask:
            weight = logits.new_ones((batch_size, num_token))
            overlay = [
                self.det_coord_weight if i % 5 in self.det_coord else 1.0
                for i in range(5)
            ]
            overlay = torch.tensor(overlay, device=weight.device, dtype=weight.dtype)
            for elem in weight:
                elem[:5] = overlay
            weight = weight.reshape(-1)
        elif with_bbox:
            weight = logits.new_tensor(
                [
                    self.det_coord_weight if i % 5 in self.det_coord else 1.0
                    for i in range(batch_size * num_token)
                ]
            )
        elif with_mask:
            weight = logits.new_tensor([1.0 for _ in range(batch_size * num_token)])
            weight[targets.view(-1) == self.end] /= 10.0

        loss_ce = self.loss_ce(logits, targets, weight=weight)
        return loss_ce

    def forward_test(self, x_mm, img_metas, with_bbox=False, with_mask=False):
        x_mask, x_pos_embeds = self.transformer.x_mask_pos_enc(x_mm, img_metas)
        memory = self.transformer.forward_encoder(x_mm, x_mask, x_pos_embeds)
        return self.generate_sequence(
            memory, x_mask, x_pos_embeds, with_bbox=with_bbox, with_mask=with_mask
        )

    def generate(
        self, seq_in_embeds, memory, x_pos_embeds, x_mask, decode_steps, with_mask
    ):
        seq_out = []
        for step in range(decode_steps):
            out = self.transformer.forward_decoder(
                seq_in_embeds, memory, x_pos_embeds, x_mask
            )
            logits = out[:, -1, :]
            logits = self.predictor(logits)
            if self.multi_task:
                if step < 4:
                    logits = logits[:, :-1]
            else:
                if not with_mask:
                    logits = logits[:, :-1]
            probs = f.softmax(logits, dim=-1)
            if self.top_p > 0.0:
                sorted_score, sorted_idx = torch.sort(probs, descending=True)
                cum_score = sorted_score.cumsum(dim=-1)
                sorted_idx_to_remove = cum_score > self.top_p
                sorted_idx_to_remove[..., 1:] = sorted_idx_to_remove[..., :-1].clone()
                sorted_idx_to_remove[..., 0] = 0
                idx_to_remove = sorted_idx_to_remove.scatter(
                    1, sorted_idx, sorted_idx_to_remove
                )
                probs = probs.masked_fill(idx_to_remove, 0.0)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                _, next_token = probs.max(dim=-1, keepdim=True)

            seq_in_embeds = torch.cat(
                [seq_in_embeds, self.transformer.query_embedding(next_token)], dim=1
            )

            seq_out.append(next_token)

        seq_out = torch.cat(seq_out, dim=-1)

        return seq_out

    def generate_sequence(
        self, memory, x_mask, x_pos_embeds, with_bbox=False, with_mask=False
    ):
        """Args:
        memory (tensor): encoder's output, [batch_size, h*w, d_model].

        x_mask (tensor): [batch_size, h*w], dtype is torch.bool, True means
            ignored position.

        x_pos_embeds (tensor): [batch_size, h*w, d_model].
        """
        batch_size = memory.size(0)
        if with_bbox and with_mask:
            task_bbox = (
                self.task_embedding.weight[0]
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )
            seq_out_bbox = self.generate(
                task_bbox, memory, x_pos_embeds, x_mask, 4, False
            )
            task_mask = (
                self.task_embedding.weight[1]
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )
            seq_in_embeds_box = self.transformer.query_embedding(seq_out_bbox)
            seq_in_embeds_mask = torch.cat(
                [task_bbox, seq_in_embeds_box, task_mask], dim=1
            )
            seq_out_mask = self.generate(
                seq_in_embeds_mask,
                memory,
                x_pos_embeds,
                x_mask,
                2 * self.num_ray + 1,
                True,
            )
            return dict(seq_out_bbox=seq_out_bbox, seq_out_mask=seq_out_mask)
        else:
            seq_in_embeds = memory.new_zeros((batch_size, 1, self.d_model))
            if with_mask:
                decode_steps = self.num_ray * 2 + 1
            elif with_bbox:
                decode_steps = 4
            seq_out = self.generate(
                seq_in_embeds, memory, x_pos_embeds, x_mask, decode_steps, with_mask
            )
            if with_bbox:
                return dict(seq_out_bbox=seq_out)
            elif with_mask:
                return dict(seq_out_mask=seq_out)


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, eos_coef, losses, temperature):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        self.temperature = temperature
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_isfinal(self, outputs, targets, positive_map, indices, num_boxes):
        """This loss is used in some referring expression dataset (specifically Clevr-REF+)
        It trains the model to predict which boxes are being referred to (ie are "final")
        Eg if the caption is "the cube next to the cylinder", MDETR will detect both the cube and the cylinder.
        However, the cylinder is an intermediate reasoning step, only the cube is being referred here.
        """
        idx = self._get_src_permutation_idx(indices)
        src_isfinal = outputs["pred_isfinal"][idx].squeeze(-1)
        target_isfinal = torch.cat(
            [t["isfinal"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_isfinal = F.binary_cross_entropy_with_logits(
            src_isfinal, target_isfinal, reduction="none"
        )

        losses = {}
        losses["loss_isfinal"] = loss_isfinal.sum() / num_boxes
        acc = (src_isfinal.sigmoid() > 0.5) == (target_isfinal > 0.5)
        if acc.numel() == 0:
            acc = acc.sum()
        else:
            acc = acc.float().mean()
        losses["accuracy_isfinal"] = acc

        return losses

    def loss_labels(self, outputs, targets, positive_map, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """

        logits = outputs["pred_logits"].log_softmax(
            -1
        )  # BS x (num_queries) x (num_tokens)

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = []
        offset = 0
        for i, (_, tgt) in enumerate(indices):
            tgt_idx.append(tgt + offset)
            offset += len(targets[i]["boxes"])
        tgt_idx = torch.cat(tgt_idx)

        tgt_pos = positive_map[tgt_idx]
        target_sim = torch.zeros_like(logits)
        target_sim[:, :, -1] = 1
        target_sim[src_idx] = tgt_pos

        loss_ce = -(logits * target_sim).sum(-1)

        eos_coef = torch.full(loss_ce.shape, self.eos_coef, device=target_sim.device)
        eos_coef[src_idx] = 1

        loss_ce = loss_ce * eos_coef
        loss_ce = loss_ce.sum() / num_boxes

        losses = {"loss_ce": loss_ce}

        return losses

    def loss_contrastive_align(
        self, outputs, targets, positive_map, indices, num_boxes
    ):
        bs = outputs["proj_queries"].shape[0]
        tokenized = outputs["tokenized"]

        normalized_text_emb = outputs["proj_tokens"]  # BS x (num_tokens) x hdim
        normalized_img_emb = outputs["proj_queries"]  # BS x (num_queries) x hdim

        logits = (
            torch.matmul(normalized_img_emb, normalized_text_emb.transpose(-1, -2))
            / self.temperature
        )  # BS x (num_queries) x (num_tokens)

        # construct a map such that positive_map[k, i,j] = True iff query i is associated to token j in batch item k
        # For efficency, the construction happens on CPU, then the whole matrix is transferred to GPU in one go.
        positive_map = torch.zeros(logits.shape, dtype=torch.bool)
        for i, ((idx_src, idx_tgt), tgt) in enumerate(zip(indices, targets)):
            if "tokens_positive" in tgt:
                cur_tokens = [tgt["tokens_positive"][j] for j in idx_tgt]
            else:
                cur_tokens = [tgt["tokens"][j] for j in idx_tgt]

            for j, tok_list in enumerate(cur_tokens):
                for beg, end in tok_list:
                    beg_pos = tokenized.char_to_token(i, beg)
                    end_pos = tokenized.char_to_token(i, end - 1)
                    if beg_pos is None:
                        try:
                            beg_pos = tokenized.char_to_token(beg + 1)
                            if beg_pos is None:
                                beg_pos = tokenized.char_to_token(beg + 2)
                        except:
                            beg_pos = None
                    if end_pos is None:
                        try:
                            end_pos = tokenized.char_to_token(end - 2)
                            if end_pos is None:
                                end_pos = tokenized.char_to_token(end - 3)
                        except:
                            end_pos = None
                    if beg_pos is None or end_pos is None:
                        continue

                    assert beg_pos is not None and end_pos is not None
                    positive_map[i, idx_src[j], beg_pos : end_pos + 1].fill_(True)

        positive_map = positive_map.to(logits.device)
        positive_logits = -logits.masked_fill(~positive_map, 0)
        negative_logits = logits  # .masked_fill(positive_map, -1000000)

        boxes_with_pos = positive_map.any(2)
        pos_term = positive_logits.sum(2)
        neg_term = negative_logits.logsumexp(2)

        nb_pos = positive_map.sum(2) + 1e-6

        box_to_token_loss = (
            ((pos_term / nb_pos + neg_term)).masked_fill(~boxes_with_pos, 0).sum()
        )

        tokens_with_pos = positive_map.any(1)
        pos_term = positive_logits.sum(1)
        neg_term = negative_logits.logsumexp(1)

        nb_pos = positive_map.sum(1) + 1e-6

        tokens_to_boxes_loss = (
            ((pos_term / nb_pos + neg_term)).masked_fill(~tokens_with_pos, 0).sum()
        )
        tot_loss = (box_to_token_loss + tokens_to_boxes_loss) / 2

        return {"loss_contrastive_align": tot_loss / num_boxes}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, positive_map, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        ## Count the number of predictions that are NOT "no-object" (which is the last class)
        # normalized_text_emb = outputs["proj_tokens"]  # BS x (num_tokens) x hdim
        # normalized_img_emb = outputs["proj_queries"]  # BS x (num_queries) x hdim

        # logits = torch.matmul(
        #    normalized_img_emb, normalized_text_emb.transpose(-1, -2)
        # )  # BS x (num_queries) x (num_tokens)
        # card_pred = (logits[:, :, 0] > 0.5).sum(1)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, positive_map, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, positive_map, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = NestedTensor.from_tensor_list(
            [t["masks"] for t in targets]
        ).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(
            src_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(
        self, loss, outputs, targets, positive_map, indices, num_boxes, **kwargs
    ):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
            "isfinal": self.loss_isfinal,
            "contrastive_align": self.loss_contrastive_align,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](
            outputs, targets, positive_map, indices, num_boxes, **kwargs
        )

    def forward(self, outputs, targets, positive_map):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, positive_map)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if dist.is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / dist.get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, outputs, targets, positive_map, indices, num_boxes)
            )

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets, positive_map)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(
                        loss,
                        aux_outputs,
                        targets,
                        positive_map,
                        indices,
                        num_boxes,
                        **kwargs,
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
