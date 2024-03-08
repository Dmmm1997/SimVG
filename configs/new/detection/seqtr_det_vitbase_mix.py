d_model = 256
num_bin = 1000
pretrained = "pretrain_weights/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz"  # noqa
# pretrained = "pretrain_weights/jx_vit_base_p16_384-83fb41ba.pth"
# pretrained = "pretrain_weights/jx_vit_large_p32_384-9b920ba8.pth"
# pretrained = "pretrain_weights/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz"
fusion_dim = 768

model = dict(
    type="SeqTR",
    vis_enc=dict(
        type="VisionTransformerMix",
        pretrain=pretrained,
        frozen_stages=None,
        patch_size=32, 
        embed_dim=768, depth=12, num_heads=12, 
        # embed_dim=1024, depth=24, num_heads=16,
        # embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=4, 
        qkv_bias=True,
        drop_path_rate=0.1
    ),
    lan_enc=dict(
        type="LSTM",
        lstm_cfg=dict(type="gru", num_layers=1, hidden_size=fusion_dim//2, dropout=0.0, bias=True, bidirectional=True, batch_first=True),
        freeze_emb=True,
        output_cfg=dict(type="max"),
    ),
    fusion=dict(type="SimpleFusionv2", vis_chs=[768], fusion_dim=fusion_dim, direction="none"),
    head=dict(
        type="SeqHead",
        in_ch=fusion_dim,
        num_bin=num_bin,
        multi_task=False,
        shuffle_fraction=-1,
        mapping="relative",
        top_p=-1,
        num_ray=-1,
        det_coord=[0],
        det_coord_weight=1.5,
        loss=dict(type="LabelSmoothCrossEntropyLoss", neg_factor=0.1),
        predictor=dict(
            num_fcs=3,
            in_chs=[d_model, d_model, d_model],
            out_chs=[d_model, d_model, num_bin + 1],
            fc=[
                dict(linear=dict(type="Linear", bias=True), act=dict(type="ReLU", inplace=True), drop=None),
                dict(linear=dict(type="Linear", bias=True), act=dict(type="ReLU", inplace=True), drop=None),
                dict(linear=dict(type="Linear", bias=True), act=None, drop=None),
            ],
        ),
        transformer=dict(
            type="AutoRegressiveTransformer",
            encoder=dict(
                num_layers=6, layer=dict(d_model=d_model, nhead=8, dim_feedforward=4 * d_model, dropout=0.1, activation="relu", batch_first=True)
            ),
            decoder=dict(
                num_layers=3,
                layer=dict(d_model=d_model, nhead=8, dim_feedforward=4 * d_model, dropout=0.1, activation="relu", batch_first=True),
            ),
        ),
        x_positional_encoding=dict(type="SinePositionalEncoding2D", num_feature=d_model // 2, normalize=True),
        seq_positional_encoding=dict(type="LearnedPositionalEncoding1D", num_embedding=4 + 1, num_feature=d_model),
    ),
)

optimizer_config = dict(type="Adam", lr=0.0005, betas=(0.9, 0.98), eps=1e-9, weight_decay=0, amsgrad=True)
# optimizer_config = dict(type="AdamW", lr=0.0002)
grad_norm_clip = 0.15
scheduler_config = dict(type="MultiStepLRWarmUp", warmup_epochs=5, decay_steps=[25], decay_ratio=0.1, max_epoch=30)
