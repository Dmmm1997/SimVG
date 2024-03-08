_base_ = [
    '../../../_base_/datasets/detection/refcoco-unc.py',
    '../../../_base_/misc.py',
    '../seqtr_det_swint.py'
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
)

work_dir = "work_dir/seqtr_det_refcoco-unc_swin"
