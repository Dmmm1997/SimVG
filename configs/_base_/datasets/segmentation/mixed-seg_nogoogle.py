dataset = "MixedSeg"
data_root = "./data/seqtr_type/"
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(type="LoadImageAnnotationsFromFile", max_token=20, with_mask=True, dataset=dataset),
    dict(type="Resize", img_scale=(640, 640)),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="SampleMaskVertices", num_ray=18, center_sampling=False),
    dict(type="DefaultFormatBundle"),
    dict(type="CollectData", keys=["img", "ref_expr_inds", "gt_mask_rle", "is_crowd", "gt_mask_vertices", "mass_center"]),
]
val_pipeline = [
    dict(type="LoadImageAnnotationsFromFile", max_token=20, with_mask=True, dataset=dataset),
    dict(type="Resize", img_scale=(640, 640), keep_ratio=False),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="CollectData", keys=["img", "ref_expr_inds", "is_crowd", "gt_mask_rle"]),
]
test_pipeline = val_pipeline.copy()

word_emb_cfg = dict(type="GloVe")
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type=dataset,
        which_set="train",
        img_source=["coco"],
        annsfile=data_root + "annotations/mixed-seg/instances_nogoogle.json",
        imgsfile=data_root + "images/mscoco/train2014",
        pipeline=train_pipeline,
        word_emb_cfg=word_emb_cfg,
    ),
    val_refcoco_unc=dict(
        type=dataset,
        which_set="val_refcoco_unc",
        img_source=["coco"],
        annsfile=data_root + "annotations/mixed-seg/instances_nogoogle.json",
        imgsfile=data_root + "images/mscoco/train2014",
        pipeline=val_pipeline,
        word_emb_cfg=word_emb_cfg,
    ),
    testA_refcoco_unc=dict(
        type=dataset,
        which_set="testA_refcoco_unc",
        img_source=["coco"],
        annsfile=data_root + "annotations/mixed-seg/instances_nogoogle.json",
        imgsfile=data_root + "images/mscoco/train2014",
        pipeline=val_pipeline,
        word_emb_cfg=word_emb_cfg,
    ),
    testB_refcoco_unc=dict(
        type=dataset,
        which_set="testB_refcoco_unc",
        img_source=["coco"],
        annsfile=data_root + "annotations/mixed-seg/instances_nogoogle.json",
        imgsfile=data_root + "images/mscoco/train2014",
        pipeline=val_pipeline,
        word_emb_cfg=word_emb_cfg,
    ),
    val_refcocoplus_unc=dict(
        type=dataset,
        which_set="val_refcocoplus_unc",
        img_source=["coco"],
        annsfile=data_root + "annotations/mixed-seg/instances_nogoogle.json",
        imgsfile=data_root + "images/mscoco/train2014",
        pipeline=test_pipeline,
        word_emb_cfg=word_emb_cfg,
    ),
    testA_refcocoplus_unc=dict(
        type=dataset,
        which_set="testA_refcocoplus_unc",
        img_source=["coco"],
        annsfile=data_root + "annotations/mixed-seg/instances_nogoogle.json",
        imgsfile=data_root + "images/mscoco/train2014",
        pipeline=test_pipeline,
        word_emb_cfg=word_emb_cfg,
    ),
    testB_refcocoplus_unc=dict(
        type=dataset,
        which_set="testB_refcocoplus_unc",
        img_source=["coco"],
        annsfile=data_root + "annotations/mixed-seg/instances_nogoogle.json",
        imgsfile=data_root + "images/mscoco/train2014",
        pipeline=test_pipeline,
        word_emb_cfg=word_emb_cfg,
    ),
    val_refcocog_umd=dict(
        type=dataset,
        which_set="val_refcocog_umd",
        img_source=["coco"],
        annsfile=data_root + "annotations/mixed-seg/instances_nogoogle.json",
        imgsfile=data_root + "images/mscoco/train2014",
        pipeline=test_pipeline,
        word_emb_cfg=word_emb_cfg,
    ),
    test_refcocog_umd=dict(
        type=dataset,
        which_set="test_refcocog_umd",
        img_source=["coco"],
        annsfile=data_root + "annotations/mixed-seg/instances_nogoogle.json",
        imgsfile=data_root + "images/mscoco/train2014",
        pipeline=test_pipeline,
        word_emb_cfg=word_emb_cfg,
    ),
)
