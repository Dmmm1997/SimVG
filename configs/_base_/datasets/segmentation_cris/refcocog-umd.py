dataset = "RefCOCOgUMD"
data_root = "./data/cris_type/"
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(type="LoadImageAnnotationsFromFile", max_token=20, with_mask=True, dataset="RefCOCOgUMD"),
    dict(type="LargeScaleJitter", out_max_size=512, jitter_min=0.3, jitter_max=1.4),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="SampleMaskVertices", num_ray=18, center_sampling=False),
    dict(type="DefaultFormatBundle"),
    dict(type="CollectData", keys=["img", "ref_expr_inds", "gt_mask_rle", "is_crowd", "gt_mask_vertices", "mass_center"]),
]
val_pipeline = [
    dict(type="LoadImageAnnotationsFromFile", max_token=20, with_mask=True, dataset="RefCOCOgUMD"),
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
        lmdb_dir=data_root + "lmdb/refcocog_u/train.lmdb",
        mask_dir=data_root + "masks/refcocog_u",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset,
        which_set="val",
        imgsfile=data_root + "images/train2014",
        lmdb_dir=data_root + "lmdb/refcocog_u/val.lmdb",
        mask_dir=data_root + "masks/refcocog_u",
        pipeline=val_pipeline,
    ),
    test=dict(
        type=dataset,
        which_set="test",
        imgsfile=data_root + "images/train2014",
        lmdb_dir=data_root + "lmdb/refcocog_u/test.lmdb",
        mask_dir=data_root + "masks/refcocog_u",
        pipeline=test_pipeline,
    ),
)
