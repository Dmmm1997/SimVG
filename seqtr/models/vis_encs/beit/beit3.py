# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import torch
import torch.nn as nn
from .modeling_utils import BEiT3Wrapper, _get_base_config, _get_large_config
from seqtr.models import VIS_ENCODERS
from .utils import load_state_dict


class Pooler(nn.Module):
    def __init__(self, input_features, output_features, norm_layer):
        super().__init__()
        self.norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.Tanh()

    def forward(self, x):
        cls_rep = x[:, 0, :]
        cls_rep = self.norm(cls_rep)
        pooled_output = self.dense(cls_rep)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@VIS_ENCODERS.register_module()
class BEIT3(BEiT3Wrapper):
    def __init__(
        self,
        img_size=384,
        patch_size=32,
        vit_type="base",
        drop_path_rate=0.1,
        vocab_size=64010,
        norm_layer=nn.LayerNorm,
        freeze_layer=-1,
        vision_embed_proj_interpolate=False,
        pretrain="/home/dmmm/demo_mirror/vlm/unilm/beit3/pretrain_weights/beit3_base_patch16_384_coco_retrieval.zip",
    ):
        if vit_type == "base":
            args = _get_base_config(
                img_size=img_size,
                patch_size=patch_size,
                drop_path_rate=drop_path_rate,
                vocab_size=vocab_size,
            )
        elif vit_type == "large":
            args = _get_large_config(
                img_size=img_size,
                patch_size=patch_size,
                rop_path_rate=drop_path_rate,
                vocab_size=vocab_size,
            )
        else:
            raise TypeError("please select the <vit_type> from ['base','large']")

        super(BEIT3, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.pooler = Pooler(
            input_features=embed_dim,
            output_features=embed_dim,
            norm_layer=norm_layer,
        )
        self.pooler.apply(self._init_weights)
        self.hidden_size = embed_dim
        self.vision_embed_proj_interpolate = vision_embed_proj_interpolate
        # load pretrain checkpoint
        if isinstance(pretrain, str):
            self.load_model_and_may_interpolate(pretrain)
        # freeze the encoder
        if freeze_layer >= 0:
            self.frozen_stages = freeze_layer if freeze_layer <= len(self.beit3.encoder.layers) else len(self.beit3.encoder.layers)
            self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.beit3.text_embed.eval()
            self.beit3.vision_embed.eval()
            for param in self.beit3.text_embed.parameters():
                param.requires_grad = False
            for param in self.beit3.vision_embed.parameters():
                param.requires_grad = False
            for i in range(1, self.frozen_stages + 1):
                m = self.beit3.encoder.layers[i - 1]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def load_model_and_may_interpolate(self, ckpt_path, model_key="model|module", model_prefix=""):
        if ckpt_path.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(ckpt_path, map_location="cpu", check_hash=True)
        else:
            checkpoint = torch.load(ckpt_path, map_location="cpu")

        print("Load ckpt from %s" % ckpt_path)
        checkpoint_model = None
        for model_key in model_key.split("|"):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break

        if checkpoint_model is None:
            checkpoint_model = checkpoint

        state_dict = self.state_dict()
        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        for pos_embed_key in (
            "vision_pos_embed",
            "pos_embed",
            "beit3.encoder.embed_positions.A.weight",
        ):
            if pos_embed_key in checkpoint_model:
                pos_embed_checkpoint = checkpoint_model[pos_embed_key]
                embedding_size = pos_embed_checkpoint.shape[-1]
                if pos_embed_key == "beit3.encoder.embed_positions.A.weight":
                    # being consistent with Fairseq, which starts from 2 for position embedding
                    torchscale_model = True
                    num_patches = self.beit3.vision_embed.num_patches
                    num_extra_tokens = self.beit3.vision_embed.num_position_embeddings() + 2 - num_patches
                else:
                    torchscale_model = False
                    num_patches = self.patch_embed.num_patches
                    num_extra_tokens = getattr(self, pos_embed_key).shape[-2] - num_patches
                # height (== width) for the checkpoint position embedding
                orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int(num_patches**0.5)
                # class_token and dist_token are kept unchanged
                if orig_size != new_size:
                    print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                    if torchscale_model:
                        extra_tokens = pos_embed_checkpoint[:num_extra_tokens].unsqueeze(0)
                        # only the position tokens are interpolated
                        pos_tokens = pos_embed_checkpoint[num_extra_tokens:]
                    else:
                        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                        # only the position tokens are interpolated
                        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2).float()
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens,
                        size=(new_size, new_size),
                        mode="bicubic",
                        align_corners=False,
                    )
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    if torchscale_model:
                        new_pos_embed = new_pos_embed.squeeze(0)
                    checkpoint_model[pos_embed_key] = new_pos_embed

        if (
            checkpoint_model["beit3.vision_embed.proj.weight"].shape != self.beit3.vision_embed.proj.weight.shape
        ) and self.vision_embed_proj_interpolate:
            vision_embed_proj_weight = checkpoint_model["beit3.vision_embed.proj.weight"]
            new_size = self.beit3.vision_embed.proj.weight.shape[-2:]
            vision_embed_proj_weight = torch.nn.functional.interpolate(
                vision_embed_proj_weight.float(),
                size=new_size,
                mode="bicubic",
                align_corners=False,
            )
            checkpoint_model["beit3.vision_embed.proj.weight"] = vision_embed_proj_weight

        load_state_dict(self, checkpoint_model, prefix=model_prefix)

    def forward(self, image, question, padding_mask, **kwargs):
        outputs = self.beit3(
            textual_tokens=question,
            visual_tokens=image,
            text_padding_position=padding_mask,
        )
        x = outputs["encoder_out"]
        img_feat, text_feat = x[:, 1 : -question.shape[-1]], x[:, -question.shape[-1] :]
        cls_rep = self.pooler(x)
        return img_feat, text_feat, cls_rep
