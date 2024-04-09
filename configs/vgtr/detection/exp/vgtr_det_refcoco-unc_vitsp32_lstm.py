_base_ = [
    "../../../_base_/datasets/detection/refcoco-unc.py",
    "../../../_base_/misc.py"
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
)

model = dict(
    type="VGTR",
    vis_enc=dict(
        type="VIT",
        freeze_layer=-1,
        model_name="vit_base_patch32_384",
        pretrained=True,
        img_size=(512, 512),
        dynamic_img_size=False,
    ),
    lan_enc=dict(
        _delete_=True,
        type="LSTM",
        lstm_cfg=dict(
            type="gru",
            num_layers=1,
            hidden_size=512,
            dropout=0.0,
            bias=True,
            bidirectional=True,
            batch_first=True,
        ),
        freeze_emb=True,
        output_cfg=dict(type="query"),
        out_dim=256,
    ),
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
ema = False
# work_dir = "work_dir/seqtr_det_refcoco-unc_pvtv2mmb1_mix_type1_detectionpretrain_nofreeze_fusionv3_lr0.0003_ema_ep30"
# work_dir = "work_dir/vgtr/vgtr_det_refcoco-unc_vitbp32_lstm_size512"

# optimizer_config = dict(type="Adam", lr=0.0002, betas=(0.9, 0.98), eps=1e-9, weight_decay=0, amsgrad=True)
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
