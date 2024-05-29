dataset = "RefCOCOUNC"
data_root = "./data/cris_type/"
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(type="LoadImageAnnotationsFromFileCRIS", max_token=15, with_mask=True, dataset="RefCOCOUNC"),
    dict(type="LargeScaleJitter", out_max_size=512, jitter_min=0.3, jitter_max=1.4),
    dict(type="Resize", img_scale=(512, 512), keep_ratio=False),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="SampleMaskVertices", num_ray=18, center_sampling=False),
    dict(type="DefaultFormatBundle"),
    dict(type="CollectData", keys=["img", "ref_expr_inds", "gt_mask_rle", "is_crowd", "gt_mask_vertices", "mass_center"]),
]
val_pipeline = [
    dict(type="LoadImageAnnotationsFromFileCRIS", max_token=15, with_mask=True, dataset="RefCOCOUNC"),
    dict(type="Resize", img_scale=(512, 512), keep_ratio=False),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="CollectData", keys=["img", "ref_expr_inds", "is_crowd", "gt_mask_rle"]),
]
test_pipeline = val_pipeline.copy()

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type=dataset,
        which_set="train",
        imgsfile=data_root + "images/train2014",
        lmdb_dir=data_root + "lmdb/refcoco/train.lmdb",
        mask_dir=data_root + "masks/refcoco",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset,
        which_set="val",
        imgsfile=data_root + "images/train2014",
        lmdb_dir=data_root + "lmdb/refcoco/val.lmdb",
        mask_dir=data_root + "masks/refcoco",
        pipeline=val_pipeline,
    ),
    testA=dict(
        type=dataset,
        which_set="testA",
        imgsfile=data_root + "images/mscoco/train2014",
        lmdb_dir=data_root + "lmdb/refcoco/testA.lmdb",
        mask_dir=data_root + "masks/refcoco",
        pipeline=test_pipeline,
    ),
    testB=dict(
        type=dataset,
        which_set="testB",
        imgsfile=data_root + "images/mscoco/train2014",
        lmdb_dir=data_root + "lmdb/refcoco/testB.lmdb",
        mask_dir=data_root + "masks/refcoco",
        pipeline=test_pipeline,
    ),
)
