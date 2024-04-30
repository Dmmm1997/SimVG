import torch
import torch.nn as nn
from simvg.models import LAN_ENCODERS
from .rnn import PhraseAttention

@LAN_ENCODERS.register_module()
class LSTM(nn.Module):
    def __init__(
        self,
        num_token,
        word_emb,
        lstm_cfg=dict(type="gru", num_layers=1, dropout=0.0, hidden_size=512, bias=True, bidirectional=True, batch_first=True),
        output_cfg=dict(type="max"),
        freeze_emb=True,
        out_dim=256
    ):
        super(LSTM, self).__init__()
        self.fp16_enabled = False
        self.num_token = num_token

        assert len(word_emb) > 0
        lstm_input_ch = word_emb.shape[-1]
        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(word_emb),
            freeze=freeze_emb,
        )

        assert lstm_cfg.pop("type") in ["gru"]
        self.lstm = nn.GRU(**lstm_cfg, input_size=lstm_input_ch)

        output_type = output_cfg.pop("type")
        assert output_type in ["mean", "default", "max", "original", "query"]
        self.output_type = output_type
        
        if self.output_type == "query":
            self.parser = nn.ModuleList([PhraseAttention(input_dim=lstm_cfg["hidden_size"] * 2)
                       for _ in range(4)])
            self.linear = nn.Linear(lstm_cfg["hidden_size"] * 2, out_dim)
        
    def forward(self, ref_expr_inds):
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
        y_mask = torch.abs(ref_expr_inds) == 0

        y_word = self.embedding(ref_expr_inds)

        y_word, h = self.lstm(y_word)

        if self.output_type == "mean":
            y = torch.cat(list(map(lambda feat, mask: torch.mean(feat[mask, :], dim=0, keepdim=True), y_word, ~y_mask))).unsqueeze(1)
        elif self.output_type == "max":
            y = torch.cat(list(map(lambda feat, mask: torch.max(feat[mask, :], dim=0, keepdim=True)[0], y_word, ~y_mask))).unsqueeze(1)
        elif self.output_type == "default":
            h = h.transpose(0, 1)
            y = h.flatten(1).unsqueeze(1)
        elif self.output_type == "query":
            y = [module(y_word, y_word, ref_expr_inds)[-1] for module in self.parser]
            y = torch.stack(y, dim=1)
            y = self.linear(y)
        elif self.output_type == "original":
            res = {
                "text_feat":y_word,
                "text_mask":y_mask
            }
            return res
            
        return y
