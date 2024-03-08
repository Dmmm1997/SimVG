_base_ = ["../../../_base_/datasets/detection/refcoco-unc.py", "../../../_base_/misc.py", "../seqtr_det_resnetmm.py"]

pretrain = "https://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth"

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
)

model = dict(
    vis_enc=dict(
        type="ResNetMM",
        frozen_stages=-1,
        out_indices=[3],
        init_cfg=dict(type="Pretrained", checkpoint=pretrain, prefix="backbone."),
    ),
    fusion=dict(type="SimpleFusionv2", vis_chs=[2048], direction="none"),
)

use_fp16 = False
ema = True
work_dir = "work_dir/seqtr_det_refcoco-unc_resnet50mm_nofreeze_lr0.0001"

lr = 0.0002
optimizer_config = dict(
    type="Adam", lr=lr, lr_vis_enc=lr / 10.0, lr_lan_enc=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0, amsgrad=True
)
scheduler_config = dict(type="MultiStepLRWarmUp", warmup_epochs=0, decay_steps=[28, 36], decay_ratio=0.3, max_epoch=40)
