_base_ = [
    '../../_base_/datasets/detection/refcoco-unc.py',
    '../../_base_/misc.py',
    './seqtr_det_cspdarknet.py'
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
)

work_dir = "work_dir/seqtr/seqtr_det_refcoco-unc_cspdarknet"
