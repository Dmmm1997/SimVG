_base_ = [
    "../../../_base_/datasets/detection/refcocoplus-unc.py",
    "../../../_base_/misc.py",
]
dataset = 'RefCOCOPlusUNC'
max_token = 20
img_size = 512

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(
        type="LoadImageAnnotationsFromFile",
        max_token=max_token,
        with_bbox=True,
        dataset=dataset,
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
        keys=["img", "ref_expr_inds", "gt_bbox", "text_attention_mask"],
    ),
]

test_pipeline = val_pipeline.copy()

data = dict(
    samples_per_gpu=32,
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
    )
)

model = dict(
    type="MIXDETR",
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
        type="DETRHead",
        num_queries=1,
        in_channels=768,
        embed_dim=256,
        num_classes=1,
        aux_loss=True,
        num_encoder_layers=6,
        num_decoder_layers=3,
        only_decoder=True,
    ),
)

grad_norm_clip = 0.15
use_fp16 = False
ema = True
# work_dir = "work_dir/seqtr_det_refcoco-unc_pvtv2mmb1_mix_type1_detectionpretrain_nofreeze_fusionv3_lr0.0003_ema_ep30"
work_dir = "work_dir/beit3_finetune_coco/beit3_detronlydecoder_det_coco_all_vitbp32_512_maxtoken20_oq1_finetune_refcoco+"

lr = 0.0001
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
    warmup_epochs=0,
    decay_steps=[8],
    decay_ratio=0.1,
    max_epoch=10,
)

load_from = "work_dir/beit3_mix_pretrain/beit3_detronlydecoder_det_coco_all_vitbp32_512_maxtoken20_oq1/det_best.pth"