_base_ = [
    '../../../_base_/datasets/detection/refcoco-unc.py',
    '../../../_base_/misc.py',
    '../seqtr_det_darknetmm.py'
]

model = dict(
    type='SeqTR',
    vis_enc=dict(
        type='DarknetMM',
        frozen_stages=4,
        )
    )

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
)

optimizer_config = dict(type="Adam", lr=0.0002, betas=(0.9, 0.98), eps=1e-9, weight_decay=0, amsgrad=True)

work_dir = "work_dir/seqtr_det_refcoco-unc_darknetmm_freeze4_lr0.0002"