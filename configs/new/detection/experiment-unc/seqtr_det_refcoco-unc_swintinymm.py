_base_ = ["../../../_base_/datasets/detection/refcoco-unc.py", "../../../_base_/misc.py", "../seqtr_det_swintmm.py"]

pretrain = "https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_20210906_131725-bacf6f7b.pth"

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
)

model = dict(
    vis_enc=dict(
        type="SwinTransformerMM",
        frozen_stages=-1,
        out_indices=[1, 2, 3],
        init_cfg=dict(type="Pretrained", checkpoint=pretrain),
    )
)

use_fp16 = False
ema = True
work_dir = "work_dir/seqtr_det_refcoco-unc_swintinymm_detectionpretrain_nofreeze_lr0.0002_ema_ep40"

optimizer_config = dict(
    type="Adam", lr=0.0002, lr_vis_enc=0.00002, lr_lan_enc=0.0002, betas=(0.9, 0.98), eps=1e-9, weight_decay=0, amsgrad=True
)
# optimizer_config = dict(type="AdamW", lr=0.0002, lr_vis_enc=0.00002, lr_lan_enc=0.0002)
scheduler_config = dict(type="MultiStepLRWarmUp", warmup_epochs=3, decay_steps=[25, 35], decay_ratio=0.3, max_epoch=40)
