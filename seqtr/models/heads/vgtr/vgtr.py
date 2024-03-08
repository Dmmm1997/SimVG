import torch.nn as nn
from .vg_encoder import VGEncoder
from .vg_decoder import VGDecoder
from .position_encoding import PositionEmbeddingSine

from seqtr.models import HEADS


@HEADS.register_module()
class VGTRHead(nn.Module):

    def __init__(
        self, input_dim, hidden_dim, dropout, dim_feedforward, enc_layers, dec_layers, nheads
    ):
        super().__init__()

        self.input_proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)

        self.encoder = VGEncoder(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
        )

        self.decoder = VGDecoder(
            n_layers=dec_layers, n_heads=nheads, d_model=hidden_dim
        )

        self.pos_encoder = PositionEmbeddingSine(hidden_dim // 2, normalize=False)

    def forward(self, img, sent, sent_id):

        pos_feature = self.pos_encoder(img)

        # encoder
        fused_vis_feature, fused_exp_feature = self.encoder(
            self.input_proj(img), pos_feature, sent
        )

        # decoder
        out = self.decoder(
            fused_vis_feature.transpose(0, 1),
            fused_exp_feature,
            pos_feature=pos_feature.flatten(2).permute(2, 0, 1),
        )

        return out.transpose(0, 1)
