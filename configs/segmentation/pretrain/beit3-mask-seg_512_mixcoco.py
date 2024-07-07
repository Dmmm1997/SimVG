_base_ = [
    "../../_base_/datasets/segmentation/mixed-seg.py",
    "../../_base_/misc.py",
]
dataset= "MixedSeg"
max_token = 20
img_size = 512
patch_size = 32

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
    # dict(type='SampleMaskVertices', num_ray=18, center_sampling=False),
    # dict(type='Pad', pad_to_square=True),
    dict(type="DefaultFormatBundle"),
    dict(
        type="CollectData",
        keys=["img", "ref_expr_inds", "text_attention_mask", 'is_crowd', 'gt_mask_rle'],
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
    workers_per_gpu=4,
    train=dict(
        pipeline=train_pipeline,
    ),
    val_refcoco_unc=dict(
        pipeline=val_pipeline,
    ),
    testA_refcoco_unc=dict(
        pipeline=val_pipeline,
    ),
    testB_refcoco_unc=dict(
        pipeline=val_pipeline,
    ),
    val_refcocoplus_unc=dict(
        pipeline=test_pipeline,
    ),
    testA_refcocoplus_unc=dict(
        pipeline=test_pipeline,
    ),
    testB_refcocoplus_unc=dict(
        pipeline=test_pipeline,
    ),
    val_refcocog_umd=dict(
        pipeline=test_pipeline,
    ),
    test_refcocog_umd=dict(
        pipeline=test_pipeline,
    ),
    val_refcocog_google=dict(
        pipeline=test_pipeline,
    ),
)

model = dict(
    type="MIXUnetSeg",
    vis_enc=dict(
        type="BEIT3",
        img_size=img_size,
        patch_size=patch_size,
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
        type='UnetHead',
        input_channels=768,
        patch_size = 32
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