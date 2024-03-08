d_model = 256
num_bin = 1000
# pretrained = "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth"  # noqa
pretrained = "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b1.pth"

model = dict(
    type="MIX",
    vis_enc=dict(
        type="PyramidVisionTransformerV2MMMix",
        embed_dims=64,
        # num_layers=[3, 4, 6, 3],
        num_layers=[2, 2, 2, 2],
        out_indices=[1, 2, 3],
        frozen_stages=-1,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    lan_enc=dict(
        type="RNN",
        hidden_dim=256,
        rnn_hidden_dim=128,
        word_embedding_size=1024
    ),
    fusion=None,
    head=None
)

optimizer_config = dict(type="Adam", lr=0.0005, betas=(0.9, 0.98), eps=1e-9, weight_decay=0, amsgrad=True)
# optimizer_config = dict(type="AdamW", lr=0.0002)
grad_norm_clip = 0.15
scheduler_config = dict(type="MultiStepLRWarmUp", warmup_epochs=5, decay_steps=[25], decay_ratio=0.1, max_epoch=30)
