_base_ = [
    '../../_base_/datasets/detection/mixed.py',
    '../../_base_/misc.py',
    './seqtr_det_darknet.py'
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
)