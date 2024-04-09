import torch
from torch import nn
from seqtr.models.heads.tgqs_kd_detr_head.transformer import DetrTransformer, DetrTransformerEncoder, DetrTransformerDecoder


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 示例
# model = DetrTransformer(
#     decoder=DetrTransformerDecoder(
#         embed_dim=256,
#         num_heads=8,
#         attn_dropout=0.1,
#         feedforward_dim=2048,
#         ffn_dropout=0.1,
#         num_layers=3,
#         return_intermediate=True,
#         post_norm=True,
#     ),
#     only_decoder=True,
# )


model = DetrTransformerDecoder(
    embed_dim=256,
    num_heads=8,
    attn_dropout=0.1,
    feedforward_dim=512,
    ffn_dropout=0.1,
    num_layers=2,
    return_intermediate=False,
    post_norm=True,
)

print(f"模型参数量: {count_parameters(model)}")
