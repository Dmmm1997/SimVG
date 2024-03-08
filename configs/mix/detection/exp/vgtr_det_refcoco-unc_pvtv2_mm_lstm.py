_base_ = [
    "../../../_base_/datasets/detection/refcoco-unc.py",
    "../../../_base_/misc.py",
    "../vgtr_det_pvtv2_mm_mix.py",
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
)

pretrained = (
    "https://download.openmmlab.com/mmdetection/v2.0/pvt/retinanet_pvtv2-b1_fpn_1x_coco/retinanet_pvtv2-b1_fpn_1x_coco_20210831_103318-7e169a7d.pth"
    # "https://download.openmmlab.com/mmdetection/v2.0/pvt/retinanet_pvtv2-b2_fpn_1x_coco/retinanet_pvtv2-b2_fpn_1x_coco_20210901_174843-529f0b9a.pth"
)

model = dict(
    vis_enc=dict(
        type="PyramidVisionTransformerV2MMMix",
        frozen_stages=-1,
        text_in_channels=256,
        # num_layers=[3, 4, 6, 3],
        init_cfg=dict(type="Pretrained", checkpoint=pretrained, prefix="backbone."),
    ),
    lan_enc=dict(
        _delete_=True,
        type="LSTM",
        lstm_cfg=dict(type="gru", num_layers=1, hidden_size=512, dropout=0.0, bias=True, bidirectional=True, batch_first=True),
        freeze_emb=True,
        output_cfg=dict(type="query"),
        out_dim=256,
    ),
)

use_fp16 = False
ema = True
# work_dir = "work_dir/seqtr_det_refcoco-unc_pvtv2mmb1_mix_type1_detectionpretrain_nofreeze_fusionv3_lr0.0003_ema_ep30"
work_dir = "work_dir/mix/test"

# optimizer_config = dict(type="Adam", lr=0.0002, betas=(0.9, 0.98), eps=1e-9, weight_decay=0, amsgrad=True)
optimizer_config = dict(
    type="Adam",
    lr=0.0002,
    lr_vis_enc=0.0002,
    lr_lan_enc=0.0002,
    betas=(0.9, 0.98),
    eps=1e-9,
    weight_decay=0,
    amsgrad=True,
)

scheduler_config = dict(
    type="MultiStepLRWarmUp",
    warmup_epochs=3,
    decay_steps=[21, 27],
    decay_ratio=0.3,
    max_epoch=30,
)
