_base_ = [
    "../../_base_/datasets/detection/refcoco-unc.py",
    "../../_base_/misc.py",
]

num_bin = 1000
d_model = 256
max_token = 20
img_size = 512

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(
        type="LoadImageAnnotationsFromFile",
        max_token=max_token,
        with_bbox=True,
        dataset="RefCOCOUNC",
        use_token_type="beit3",
    ),
    dict(type="LargeScaleJitter", out_max_size=img_size, jitter_min=0.3, jitter_max=1.4),
    dict(type="Resize", img_scale=(img_size, img_size), keep_ratio=False),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    # dict(type='Pad', pad_to_square=True),
    dict(type="DefaultFormatBundle"),
    dict(
        type="CollectData",
        keys=["img", "ref_expr_inds", "gt_bbox", "text_attention_mask"],
    ),
]

val_pipeline = [
    dict(
        type="LoadImageAnnotationsFromFile",
        max_token=max_token,
        with_bbox=True,
        dataset="RefCOCOUNC",
        use_token_type="beit3",
    ),
    dict(type="Resize", img_scale=(img_size, img_size), keep_ratio=False),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    # dict(type='Pad', pad_to_square=True),
    dict(type="DefaultFormatBundle"),
    dict(
        type="CollectData",
        keys=["img", "ref_expr_inds", "gt_bbox", "text_attention_mask"],
    ),
]
test_pipeline = val_pipeline.copy()

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        pipeline=train_pipeline,
    ),
    val=dict(
        pipeline=val_pipeline,
    ),
    testA=dict(
        pipeline=test_pipeline,
    ),
    testB=dict(
        pipeline=test_pipeline,
    ),
)

model = dict(
    type="MIXSeqTR",
    vis_enc=dict(
        type="BEIT3",
        img_size=img_size,
        patch_size=32,
        vit_type="base",
        drop_path_rate=0.1,
        vocab_size=64010,
        freeze_layer=-1,
        vision_embed_proj_interpolate=True,
    ),
    lan_enc=None,
    fusion=None,
    head=dict(
        type="SeqHead",
        in_ch=768,
        num_bin=num_bin,
        onlydecoder=True,
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
                dict(
                    linear=dict(type="Linear", bias=True),
                    act=dict(type="ReLU", inplace=True),
                    drop=None,
                ),
                dict(
                    linear=dict(type="Linear", bias=True),
                    act=dict(type="ReLU", inplace=True),
                    drop=None,
                ),
                dict(linear=dict(type="Linear", bias=True), act=None, drop=None),
            ],
        ),
        transformer=dict(
            type="AutoRegressiveTransformer",
            encoder=dict(
                num_layers=6,
                layer=dict(
                    d_model=d_model,
                    nhead=8,
                    dim_feedforward=4 * d_model,
                    dropout=0.1,
                    activation="relu",
                    batch_first=True,
                ),
            ),
            decoder=dict(
                num_layers=3,
                layer=dict(
                    d_model=d_model,
                    nhead=8,
                    dim_feedforward=4 * d_model,
                    dropout=0.1,
                    activation="relu",
                    batch_first=True,
                ),
            ),
        ),
        x_positional_encoding=dict(type="SinePositionalEncoding2D", num_feature=d_model // 2, normalize=True),
        seq_positional_encoding=dict(type="LearnedPositionalEncoding1D", num_embedding=4 + 1, num_feature=d_model),
    ),
)

grad_norm_clip = 0.15
use_fp16 = False
ema = True
# work_dir = "work_dir/seqtr_det_refcoco-unc_pvtv2mmb1_mix_type1_detectionpretrain_nofreeze_fusionv3_lr0.0003_ema_ep30"
work_dir = "work_dir/beit3/beit3_seqtrheadonlydecoder_det_refcoco-unc_vitbp32_51_maxtoken20"

lr = 0.0005
optimizer_config = dict(
    type="Adam",
    lr=lr,
    lr_vis_enc=lr / 10.0,
    lr_lan_enc=lr,
    betas=(0.9, 0.98),
    eps=1e-9,
    weight_decay=0,
    amsgrad=True,
)

scheduler_config = dict(
    type="MultiStepLRWarmUp",
    warmup_epochs=3,
    decay_steps=[27],
    decay_ratio=0.1,
    max_epoch=30,
)
