_base_ = ["../../../_base_/datasets/detection/refcoco-unc.py", "../../../_base_/misc.py", "../seqtr_det_pvtv2_mm_concatfusion.py"]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
)

pretrained = "https://download.openmmlab.com/mmdetection/v2.0/pvt/retinanet_pvtv2-b1_fpn_1x_coco/retinanet_pvtv2-b1_fpn_1x_coco_20210831_103318-7e169a7d.pth"

model = dict(
    vis_enc=dict(
        type="PyramidVisionTransformerV2MM",
        frozen_stages=-1,
        out_indices=[1, 2, 3],
        init_cfg=dict(type="Pretrained", checkpoint=pretrained, prefix="backbone."),
    )
)

use_fp16 = False
ema = True
# work_dir = "work_dir/seqtr_det_refcoco-unc_pvtv2mm_concatfusionvisionaug_detectionpretrain_nofreeze_lr0.0002_ema"
work_dir = "work_dir/test"

optimizer_config = dict(
    type="Adam", lr=0.0002, lr_vis_enc=0.00002, lr_lan_enc=0.0002, betas=(0.9, 0.98), eps=1e-9, weight_decay=0, amsgrad=True
)
scheduler_config = dict(type="MultiStepLRWarmUp", warmup_epochs=3, decay_steps=[25, 35], decay_ratio=0.3, max_epoch=40)
