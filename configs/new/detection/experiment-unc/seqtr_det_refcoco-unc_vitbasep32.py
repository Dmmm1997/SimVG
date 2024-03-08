_base_ = [
    "../../../_base_/datasets/detection/refcoco-unc.py",
    "../../../_base_/misc.py",
    "../seqtr_det_vits.py",
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
)

model = dict(
    type="SeqTR",
    vis_enc=dict(
        type="VIT",
        # freeze_layer=6,
        model_name="vit_base_patch32_384",
        pretrained=True,
        img_size=(512, 512),
        dynamic_img_size=False,
    ),
    fusion=dict(type="SimpleFusionv2", vis_chs=[768], fusion_dim=1024, direction="none"),
)

use_fp16 = False
ema = True
work_dir = "work_dir/seqtr/seqtr_det_refcoco-unc_vitbp32_nofreeze"

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
    decay_ratio=0.3,
    max_epoch=30,
)
