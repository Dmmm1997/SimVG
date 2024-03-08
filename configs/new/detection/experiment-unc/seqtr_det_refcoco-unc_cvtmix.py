_base_ = ["../../../_base_/datasets/detection/refcoco-unc.py", "../../../_base_/misc.py", "../seqtr_det_cvtmix.py"]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
)

pretrained = "pretrain_weights/CvT-13-384x384-IN-22k.pth"

model = dict(
    type="SeqTR",
    vis_enc=dict(
        type="PyramidVisionTransformerV2MMMix",
        type_name="cvt13",
        pretrained=pretrained,
    ),
)
use_fp16 = False
ema = True
work_dir = "work_dir/seqtr_det_refcoco-unc_cvtmix_nofreeze_lr0.0002_ema"

optimizer_config = dict(type="Adam", lr=0.0002, betas=(0.9, 0.98), eps=1e-9, weight_decay=0, amsgrad=True)
