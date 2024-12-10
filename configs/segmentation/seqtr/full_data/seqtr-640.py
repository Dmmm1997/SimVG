_base_ = [
    "../../../_base_/datasets/segmentation/mixed-seg_nogoogle.py",
    "../../../_base_/misc.py",
]
dataset = "MixedSeg"
max_token = 20
img_size = 640

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(
        type="LoadImageAnnotationsFromFile",
        max_token=max_token,
        with_mask=True,
        with_bbox=True,
        dataset=dataset,
        use_token_type="default",
    ),
    dict(type="LargeScaleJitter", out_max_size=img_size, jitter_min=0.3, jitter_max=1.4),
    dict(type="Resize", img_scale=(img_size, img_size), keep_ratio=False),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type='SampleMaskVertices', num_ray=18, center_sampling=False),
    dict(type="DefaultFormatBundle"),
    # dict(
    #     type="CollectData",
    #     keys=["img", "ref_expr_inds", "text_attention_mask", "is_crowd", "gt_mask_rle", "gt_bbox"],
    # ),
    dict(type="CollectData", keys=["img", "ref_expr_inds", "gt_mask_rle", "is_crowd", "gt_mask_vertices", "mass_center", "gt_bbox"]),
]

val_pipeline = [
    dict(
        type="LoadImageAnnotationsFromFile",
        max_token=max_token,
        with_mask=True,
        with_bbox=True,
        dataset=dataset,
        use_token_type="default",
    ),
    dict(type="Resize", img_scale=(img_size, img_size), keep_ratio=False),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    # dict(
    #     type="CollectData",
    #     keys=["img", "ref_expr_inds", "text_attention_mask", "is_crowd", "gt_mask_rle", "gt_bbox"],
    # ),
    dict(type="CollectData", keys=["img", "ref_expr_inds", "is_crowd", "gt_mask_rle", "gt_bbox"]),
]
test_pipeline = val_pipeline.copy()

data = dict(
    samples_per_gpu=64,
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
)

num_ray = 18
d_model = 256
num_bin = 1000
model = dict(
    type='SeqTR',
    vis_enc=dict(
        type='DarkNet53',
        pretrained='data/seqtr_type/weights/darknet.weights',
        freeze_layer=2,
        out_layer=(6, 8, 13, )
    ),
    lan_enc=dict(
        type='LSTM',
        lstm_cfg=dict(
            type='gru',
            num_layers=1,
            hidden_size=512,
            dropout=0.,
            bias=True,
            bidirectional=True,
            batch_first=True),
        freeze_emb=True,
        output_cfg=dict(type="max")
    ),
    fusion=dict(
        type="SimpleFusion",
        direction="bottom_up",
        vis_chs=[256, 512, 1024]
    ),
    head=dict(
        type='SeqHead',
        in_ch=1024,
        num_bin=num_bin,
        multi_task=True,
        shuffle_fraction=-1,
        mapping='relative',
        top_p=-1,
        num_ray=num_ray,
        det_coord=[0],
        det_coord_weight=1.5,
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
            num_embedding=1+4+1+2*num_ray,
            num_feature=d_model
        )
    )
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
    decay_steps=[25],
    decay_ratio=0.1,
    max_epoch=30,
)

log_interval = 50
threshold = 0.5
evaluate_interval = 1
start_evaluate_epoch = 0
start_save_checkpoint = 3