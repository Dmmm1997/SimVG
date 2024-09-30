_base_ = [
    "../../../_base_/datasets/detection/referitgame-berkeley.py",
    "../../../_base_/misc.py",
]

dataset = 'ReferItGameBerkeley'
max_token = 20
img_size = 640

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
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        pipeline=train_pipeline,
    ),
    val=dict(
        pipeline=val_pipeline,
    ),
    test=dict(
        pipeline=test_pipeline,
    )
)

model = dict(
    type="MIXDETRMB",
    vis_enc=dict(
        type="BEIT3",
        img_size=img_size,
        patch_size=32,
        vit_type="large",
        drop_path_rate=0.1,
        vocab_size=64010,
        freeze_layer=-1,
        vision_embed_proj_interpolate=True,
        pretrain="pretrain_weights/beit3_large_patch16_224.zip",
    ),
    lan_enc=None,
    fusion=None,
    head=dict(
        type="TextGuidedQuerySelectKDDETRHead",
        num_queries=1,
        text_max_token=max_token,
        in_channels=1024,
        embed_dim=256,
        decoder_freeze=False,
        num_classes=1,
        aux_loss=True,
        num_encoder_layers=6,
        num_decoder_layers=3,
        only_decoder=True,
        text_embed_aug=False,
        branch_loss_weight={"decoder": 1.0, "balanced_distill": {"token":2.0, "distill":1.0}},
        distill_type="hard_weighted", # "hard", "hard_weighted", "soft"
        prepare_target_mode="score_iou_weighted", # "score_weighted", "score_iou_weighted"
        share_predicthead=False,
        num_token_mlp_layers=1,
        mlp_aux_loss=False,
        tgqs_mid_dim=512,
        text_guided_query_generation=True,
        num_tgqg_layers=2
    ),
)

grad_norm_clip = 0.15
use_fp16 = False
ema = False
# work_dir = "work_dir/seqtr_det_refcoco-unc_pvtv2mmb1_mix_type1_detectionpretrain_nofreeze_fusionv3_lr0.0003_ema_ep30"
# work_dir = "work_dir/pretrain_exp/ema#1.0decoder#1.0token"

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
    decay_steps=[25],
    decay_ratio=0.1,
    max_epoch=30,
)

log_interval = 50

# finetune_from="work_dir/paper_exp/sota_model/sota_large/ViTLargeP32-1.0decoder-30ep-640hw-tgqg_layer2_referit/20240428_123538/det_best.pth"