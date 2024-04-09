_base_ = [
    "../../../_base_/datasets/detection/refcoco-unc.py",
    "../../../_base_/misc.py",
]
dataset = "RefCOCOUNC"
max_token = 40
img_size = 512

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(
        type="LoadImageAnnotationsFromFile",
        max_token=40,
        with_bbox=True,
        dataset="RefCOCOUNC",
        use_token_type="bert",
    ),
    dict(type="LargeScaleJitter", out_max_size=512, jitter_min=0.3, jitter_max=1.4),
    dict(type="Resize", img_scale=(512, 512), keep_ratio=False),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    # dict(type='Pad', pad_to_square=True),
    dict(type="DefaultFormatBundle"),
    dict(type="CollectData", keys=["img", "ref_expr_inds", "gt_bbox"]),
]

val_pipeline = [
    dict(
        type="LoadImageAnnotationsFromFile",
        max_token=40,
        with_bbox=True,
        dataset="RefCOCOUNC",
        use_token_type="bert",
    ),
    dict(type="Resize", img_scale=(512, 512), keep_ratio=False),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    # dict(type='Pad', pad_to_square=True),
    dict(type="DefaultFormatBundle"),
    dict(type="CollectData", keys=["img", "ref_expr_inds", "gt_bbox"]),
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
    type="MIX",
    vis_enc=dict(
        type="ViLTransformerSS",
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=768 * 4,
        max_position_embeddings=40,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pretrain="/home/dmmm/demo_mirror/vlm/ViLT/pretrain_weights/vilt_200k_mlm_itm.ckpt",
    ),
    lan_enc=None,
    fusion=None,
    head=None,
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
    decay_steps=[21, 27],
    decay_ratio=0.1,
    max_epoch=30,
)

log_interval = 50
