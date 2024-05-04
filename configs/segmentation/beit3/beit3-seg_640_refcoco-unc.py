_base_ = [
    "../../_base_/datasets/segmentation/refcoco-unc.py",
    "../../_base_/misc.py",
]
dataset= "RefCOCOUNC"
max_token = 20
img_size = 640
num_ray = 18
d_model = 256
num_bin = 1000

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(
        type="LoadImageAnnotationsFromFile",
        max_token=max_token,
        with_mask=True,
        dataset=dataset,
        use_token_type="beit3",
    ),
    dict(type="LargeScaleJitter", out_max_size=img_size, jitter_min=0.3, jitter_max=1.4),
    dict(type="Resize", img_scale=(img_size, img_size), keep_ratio=False),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type='SampleMaskVertices', num_ray=18, center_sampling=False),
    # dict(type='Pad', pad_to_square=True),
    dict(type="DefaultFormatBundle"),
    dict(
        type="CollectData",
        keys=["img", "ref_expr_inds", "text_attention_mask", 'is_crowd','gt_mask_vertices', 'mass_center', 'gt_mask_rle'],
    ),
]

val_pipeline = [
    dict(
        type="LoadImageAnnotationsFromFile",
        max_token=max_token,
        with_mask=True,
        dataset=dataset,
        use_token_type="beit3",
    ),
    dict(type="Resize", img_scale=(img_size, img_size), keep_ratio=False),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    # dict(type='Pad', pad_to_square=True),
    dict(type="DefaultFormatBundle"),
    dict(
        type="CollectData",
        keys=["img", "ref_expr_inds", "text_attention_mask", 'is_crowd', 'gt_mask_rle'],
    ),
]
test_pipeline = val_pipeline.copy()

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,
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
        pretrain="pretrain_weights/beit3_base_patch16_224.zip",
    ),
    lan_enc=None,
    fusion=None,
    head=dict(
        type='SeqHead',
        in_ch=768,
        num_bin=num_bin,
        multi_task=False,
        shuffle_fraction=-1,
        mapping='relative',
        top_p=-1,
        num_ray=num_ray,
        det_coord=[-1],
        det_coord_weight=1.,
        onlydecoder=True, ######
        loss=dict(
            type="LabelSmoothCrossEntropyLoss",
            neg_factor=0.1
        ),
        predictor=dict(
            num_fcs=3, in_chs=[d_model, d_model, d_model], out_chs=[d_model, d_model, num_bin+1],
            fc=[
                dict(
                    linear=dict(type='Linear', bias=True),
                    act=dict(type='ReLU', inplace=True),
                    drop=None
                ),
                dict(
                    linear=dict(type='Linear', bias=True),
                    act=dict(type='ReLU', inplace=True),
                    drop=None
                ),
                dict(
                    linear=dict(type='Linear', bias=True),
                    act=None,
                    drop=None
                )
            ]
        ),
        transformer=dict(
            type='AutoRegressiveTransformer',
            encoder=dict(
                num_layers=6,
                layer=dict(
                    d_model=d_model,
                    nhead=8,
                    dim_feedforward=4*d_model,
                    dropout=0.1,
                    activation='relu',
                    batch_first=True)),
            decoder=dict(
                num_layers=3,
                layer=dict(
                    d_model=d_model,
                    nhead=8,
                    dim_feedforward=4*d_model,
                    dropout=0.1,
                    activation='relu',
                    batch_first=True),
            )),
        x_positional_encoding=dict(
            type='SinePositionalEncoding2D',
            num_feature=d_model//2,
            normalize=True),
        seq_positional_encoding=dict(
            type='LearnedPositionalEncoding1D',
            num_embedding=2*num_ray+1,
            num_feature=d_model
        )
    ),
)

grad_norm_clip = 0.15
use_fp16 = False
ema = False
# work_dir = "work_dir/seqtr_det_refcoco-unc_pvtv2mmb1_mix_type1_detectionpretrain_nofreeze_fusionv3_lr0.0003_ema_ep30"
# work_dir = "work_dir/paper_exp/decoder_ablation/ViTBaseP32-1.0decoder-40ep-512hw-refcocounc"

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
    decay_steps=[50],
    decay_ratio=0.1,
    max_epoch=60,
)

log_interval = 50
