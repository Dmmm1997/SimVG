_base_ = ["../../../_base_/datasets/detection/refcoco-unc.py", "../../../_base_/misc.py", "../seqtr_det_vitbase_mix.py"]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
)

pretrained = (
    "pretrain_weights/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz"
)

model = dict(
    vis_enc=dict(
        type="VisionTransformerMix",
        pretrain=pretrained,
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        frozen_stages=-1,
    ),
    fusion=dict(type="SimpleFusionv2", vis_chs=[768], fusion_dim=768, direction="none"),
)

use_fp16 = False
ema = True
work_dir = "work_dir/seqtr_det_refcoco-unc_vitb32_mix_mixformerpretrain_nofreeze_lr0.0003_ema_ep30"
lr = 0.0002
# optimizer_config = dict(type="Adam", lr=0.0002, betas=(0.9, 0.98), eps=1e-9, weight_decay=0, amsgrad=True)
optimizer_config = dict(
    type="Adam", lr=lr, lr_vis_enc=lr / 10.0, lr_lan_enc=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0, amsgrad=True
)

scheduler_config = dict(type="MultiStepLRWarmUp", warmup_epochs=0, decay_steps=[21, 27], decay_ratio=0.3, max_epoch=30)
