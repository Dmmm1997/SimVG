_base_ = [
    '../../../_base_/datasets/detection/refcoco-unc.py',
    '../../../_base_/misc.py',
    '../seqtr_det_resnet.py'
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
)
