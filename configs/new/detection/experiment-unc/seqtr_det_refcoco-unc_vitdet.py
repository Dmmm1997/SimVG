_base_ = ["../../../_base_/datasets/detection/refcoco-unc.py", "../../../_base_/misc.py", "../seqtr_det_vitdet.py"]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)

model = dict(
    type="SeqTR",
    vis_enc=dict(
        type="VITDet",
    ),
    fusion=dict(type="SimpleFusionv2", vis_chs=[768], fusion_dim=1024, direction="none"),
)

use_fp16 = False
ema = True
work_dir = "work_dir/seqtr_det_refcoco-unc_vitdet_nofreeze_adam_lr0.0002_ema"

lr = 0.0002
optimizer_config = dict(
    type="Adam", lr=lr, lr_vis_enc=lr / 10.0, lr_lan_enc=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0, amsgrad=True
)
# optimizer_config = dict(type="AdamW", lr=lr, lr_vis_enc=lr/10.0, lr_lan_enc=lr)
# optimizer_config = dict(type="SGD", lr=0.01, lr_vis_enc=0.003, lr_lan_enc=0.01, momentum=0.9)
scheduler_config = dict(type="MultiStepLRWarmUp", warmup_epochs=0, decay_steps=[28, 36], decay_ratio=0.3, max_epoch=40)
