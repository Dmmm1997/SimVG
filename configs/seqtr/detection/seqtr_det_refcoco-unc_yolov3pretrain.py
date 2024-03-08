_base_ = ["../../_base_/datasets/detection/refcoco-unc.py", "../../_base_/misc.py", "./seqtr_det_darknet.py"]

model = dict(type="SeqTR", vis_enc=dict(type="DarkNet53", pretrained="./data/weights/yolov3.weights"))

work_dir = "work_dir/seqtr/seqtr_det_refcoco-unc_yolov3pretrain_size512"


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

ema = True
