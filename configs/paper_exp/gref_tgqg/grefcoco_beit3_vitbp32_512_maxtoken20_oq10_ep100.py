_base_ = [
    "../../_base_/datasets/detection/grefcoco.py",
    "../../_base_/misc.py",
]

max_token = 20
img_size = 512
epoch=100

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(
        type="LoadImageAnnotationsFromFile",
        max_token=max_token,
        with_bbox=True,
        dataset="GRefCOCO",
        use_token_type="beit3",
    ),
    # dict(type="LargeScaleJitter", out_max_size=img_size, jitter_min=0.3, jitter_max=1.4),
    dict(type="Resize", img_scale=(img_size, img_size), keep_ratio=False),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    # dict(type='Pad', pad_to_square=True),
    dict(type="DefaultFormatBundle"),
    dict(
        type="CollectData",
        keys=["img", "ref_expr_inds", "gt_bbox", "text_attention_mask"],
        meta_keys=('filename', 'expression', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor',"target"), 
    ),
]

val_pipeline = [
    dict(
        type="LoadImageAnnotationsFromFile",
        max_token=max_token,
        with_bbox=True,
        dataset="GRefCOCO",
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
        meta_keys=('filename', 'expression', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor',"target"), 
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
    ),
)

model = dict(
    type="MIXDETRMB",
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
        type="TextGuidedQuerySelectKDDETRHead",
        num_queries=10,
        text_max_token=max_token,
        in_channels=768,
        embed_dim=256,
        decoder_freeze=False,
        num_classes=1,
        aux_loss=True,
        num_encoder_layers=6,
        num_decoder_layers=3,
        only_decoder=True,
        branch_loss_weight={"decoder": 1.0},
        distill_type="hard_weighted", # "hard", "hard_weighted", "soft"
        prepare_target_mode="score_iou_weighted", # "score_weighted", "score_iou_weighted"
        share_predicthead=False,
        num_token_mlp_layers=1,
        mlp_aux_loss=False,
        text_guided_query_generation=False,
        num_tgqg_layers=2,
    ),
)

grad_norm_clip = 0.15
use_fp16 = False
ema = False
# work_dir = "work_dir/beit3_grefcoco/grefcoco_beit3_tgqshead_vitbp32_512_maxtoken20_oq20_ep100"

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
    decay_steps=[int(epoch*0.9)],
    decay_ratio=0.1,
    max_epoch=epoch,
)

evaluate_interval=2
start_evaluate_epoch=50
