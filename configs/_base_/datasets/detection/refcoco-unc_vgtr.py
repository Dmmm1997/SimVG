dataset = 'VGTRDataset'

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type=dataset,
        data_root="/home/dmmm/demo_mirror/REC/vgtr/store/ln_data",
        split_root="/home/dmmm/demo_mirror/REC/vgtr/store/data",
        dataset="refcoco",
        imsize=512,
        transform=None,
        testmode=False,
        split="train",
        max_query_len=20,
        augment=True),
    val=dict(
        type=dataset,
        data_root="/home/dmmm/demo_mirror/REC/vgtr/store/ln_data",
        split_root="/home/dmmm/demo_mirror/REC/vgtr/store/data",
        dataset="refcoco",
        imsize=512,
        transform=None,
        testmode=False,
        split="val",
        max_query_len=20,
        augment=False)
)
