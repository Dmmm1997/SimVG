import torch
import torch.nn as nn
from simvg.models import LAN_ENCODERS
from transformers import RobertaModel, RobertaTokenizerFast
from transformers import AutoModel, AutoTokenizer

class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


@LAN_ENCODERS.register_module()
class ALBERTA(nn.Module):
    def __init__(self, text_encoder_type="roberta-base", freeze_text_encoder=False, output_cfg=dict(type="max"), word_emb=0, num_token=0):
        super(ALBERTA, self).__init__()
        # self.tokenizer = RobertaTokenizerFast.from_pretrained(text_encoder_type)
        # self.text_encoder = RobertaModel.from_pretrained(text_encoder_type)
        self.text_encoder = AutoModel.from_pretrained(text_encoder_type)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
        self.output_type = output_cfg["type"]

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        # self.resizer = FeatureResizer(
        #     input_feat_size=768,
        #     output_feat_size=1024,
        #     dropout=0.1,
        # )

    def forward(self, img_metas):
        """Args:
            ref_expr_inds (tensor): [batch_size, max_token],
                integer index of each word in the vocabulary,
                padded tokens are 0s at the last.

        Returns:
            y (tensor): [batch_size, 1, C_l].

            y_word (tensor): [batch_size, max_token, C_l].

            y_mask (tensor): [batch_size, max_token], dtype=torch.bool,
                True means ignored position.
        """
        # Encode the text
        text = [info["expression"] for info in img_metas]
        tokenized = self.tokenizer.batch_encode_plus(text, padding="longest", return_tensors="pt").to("cuda")
        encoded_text = self.text_encoder(**tokenized)

        # Transpose memory because pytorch's attention expects sequence first
        text_memory = encoded_text.last_hidden_state
        # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
        text_attention_mask = tokenized.attention_mask

        # Resize the encoder hidden states to be of the same d_model as the decoder
        # text_memory = self.resizer(text_memory)

        if self.output_type == "mean":
            y = torch.cat(list(map(lambda feat, mask: torch.mean(feat[mask, :], dim=0, keepdim=True), text_memory, ~text_attention_mask))).unsqueeze(1)
        elif self.output_type == "max":
            y = torch.cat(list(map(lambda feat, mask: torch.max(feat[mask, :], dim=0, keepdim=True)[0], text_memory, ~text_attention_mask))).unsqueeze(1)
        elif self.output_type == "default":
            h = h.transpose(0, 1)
            y = h.flatten(1).unsqueeze(1)

        return y
