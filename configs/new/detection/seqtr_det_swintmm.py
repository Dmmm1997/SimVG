d_model = 256
num_bin = 1000
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
    type="SeqTR",
    vis_enc=dict(
        type="SwinTransformerMM",
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        frozen_stages=2,
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    lan_enc=dict(
        type="LSTM",
        lstm_cfg=dict(type="gru", num_layers=1, hidden_size=512, dropout=0.0, bias=True, bidirectional=True, batch_first=True),
        freeze_emb=True,
        output_cfg=dict(type="max"),
    ),
    fusion=dict(type="SimpleFusionv2", vis_chs=[192, 384, 768], direction="bottom_up"),
    head=dict(
        type="SeqHead",
        in_ch=1024,
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

optimizer_config = dict(type="Adam", lr=0.00025, betas=(0.9, 0.98), eps=1e-9, weight_decay=0, amsgrad=True)
grad_norm_clip = 0.15
scheduler_config = dict(type="MultiStepLRWarmUp", warmup_epochs=5, decay_steps=[50], decay_ratio=0.1, max_epoch=60)
