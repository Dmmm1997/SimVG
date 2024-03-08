_base_ = [
    "../../../_base_/datasets/detection/refcoco-unc.py",
    "../../../_base_/misc.py",
    "../vgtr_det_vitsp32.py",
]

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(
        type="LoadImageAnnotationsFromFile",
        max_token=20,
        with_bbox=True,
        dataset="RefCOCOUNC",
        use_token_type="copus",
    ),
    dict(type="LargeScaleJitter", out_max_size=512, jitter_min=0.3, jitter_max=1.4),
    dict(type="Resize", img_scale=(512, 512), keep_ratio=False),
    dict(type="Normalize", **img_norm_cfg),
    # dict(type="Pad", size_divisor=32),
    # dict(type='Pad', pad_to_square=True),
    dict(type="DefaultFormatBundle"),
    dict(type="CollectData", keys=["img", "ref_expr_inds", "gt_bbox"]),
]

data = dict(samples_per_gpu=64, workers_per_gpu=4, train=dict(pipeline=train_pipeline))

model = dict(
    vis_enc=dict(
        type="VIT",
        freeze_layer=-1,
        model_name="vit_base_patch32_384",
        pretrained=True,
        img_size=(512, 512),
        dynamic_img_size=False,
    ),
    # lan_enc=dict(
    #     _delete_=True,
    #     type="LSTM",
    #     lstm_cfg=dict(
    #         type="gru",
    #         num_layers=1,
    #         hidden_size=512,
    #         dropout=0.0,
    #         bias=True,
    #         bidirectional=True,
    #         batch_first=True,
    #     ),
    #     freeze_emb=True,
    #     output_cfg=dict(type="query"),
    #     out_dim=256,
    # ),
    head=dict(
        type="VGTRHead",
        input_dim=768,
        hidden_dim=256,
        dropout=0.1,
        dim_feedforward=2048,
        enc_layers=2,
        dec_layers=2,
        nheads=8,
    ),
)

use_fp16 = False
ema = True
# work_dir = "work_dir/seqtr_det_refcoco-unc_pvtv2mmb1_mix_type1_detectionpretrain_nofreeze_fusionv3_lr0.0003_ema_ep30"
work_dir = "work_dir/vgtr/vgtr_det_refcoco-unc_vitbp32_rnn_lr0.0002_frozonbackbone"

# optimizer_config = dict(type="Adam", lr=0.0002, betas=(0.9, 0.98), eps=1e-9, weight_decay=0, amsgrad=True)
lr = 0.0002
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
    decay_steps=[21, 27],
    decay_ratio=0.3,
    max_epoch=30,
)
