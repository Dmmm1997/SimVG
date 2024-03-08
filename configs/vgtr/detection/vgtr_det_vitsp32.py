# pretrained = "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth"  # noqa
pretrained = "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b1.pth"

model = dict(
    type="VGTR",
    vis_enc=dict(
        type="VIT",
        freeze_layer=-1,
        model_name="vit_base_patch32_384",
        pretrained=True,
    ),
    lan_enc=dict(
        type="RNN",
        hidden_dim=256,
        rnn_hidden_dim=128,
        word_embedding_size=1024
    ),
    fusion=None,
    head=dict(
        type="VGTRHead",
        input_dim=384,
        hidden_dim=256,
        dropout=0.1,
        dim_feedforward=2048,
        enc_layers=2,
        dec_layers=2,
        nheads=8,
    ),
)

optimizer_config = dict(
    type="Adam", lr=0.0005, betas=(0.9, 0.98), eps=1e-9, weight_decay=0, amsgrad=True
)
# optimizer_config = dict(type="AdamW", lr=0.0002)
grad_norm_clip = 0.15
scheduler_config = dict(
    type="MultiStepLRWarmUp",
    warmup_epochs=5,
    decay_steps=[25],
    decay_ratio=0.1,
    max_epoch=30,
)
