import torch
import torch.nn as nn
from .vision_transformer import vit_base_patch32_384

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from .objectives import init_weights
from simvg.models import VIS_ENCODERS


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@VIS_ENCODERS.register_module()
class ViLTransformerSS(nn.Module):
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=768 * 4,
        max_position_embeddings=40,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pretrain="/home/dmmm/demo_mirror/pretrain/ViLT/pretrain_weights/vilt_200k_mlm_itm.ckpt",
    ):
        super().__init__()
        
        self.hidden_size = hidden_size

        bert_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(init_weights)

        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        self.token_type_embeddings.apply(init_weights)

        self.transformer = vit_base_patch32_384(
            pretrained=False, drop_rate=attention_probs_dropout_prob
        )

        self.pooler = Pooler(hidden_size)
        self.pooler.apply(init_weights)

        # ===================== Downstream ===================== #
        ckpt = torch.load(pretrain, map_location="cpu")

        state_dict = ckpt["state_dict"]
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        print("missing keys:{}".format(missing_keys))
        print("unexpected keys:{}".format(unexpected_keys))
        print("Loading pretrained ViT done.")

        # self.eval()
        # for param in self.parameters():
        #     param.requires_grad = False

    def infer(
        self,
        img,
        word_id,
        # word_mask,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        text_ids = word_id
        # text_masks = word_mask
        text_masks = ~(torch.abs(word_id) == 0).long()
        text_embeds = self.text_embeddings(text_ids)

        (
            image_embeds,
            image_masks,
            patch_index,
            image_labels,
        ) = self.transformer.visual_embed(
            img,
            max_image_len=-1,
            mask_it=mask_image,
        )

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        cls_feats = self.pooler(x)

        return image_feats, text_feats, cls_feats

    def forward(self, img, word_id, mask=None):
        ret = self.infer(img, word_id)

        return ret
