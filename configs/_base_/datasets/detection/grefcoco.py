dataset = 'GRefCOCO'
data_root = './data/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(type='LoadImageAnnotationsFromFile',
         max_token=15, with_bbox=True, dataset="GRefCOCO"),
    dict(type='LargeScaleJitter', out_max_size=512,
         jitter_min=0.3, jitter_max=1.4),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    # dict(type='Pad', pad_to_square=True),
    dict(type='DefaultFormatBundle'),
    dict(type='CollectData', keys=[
         'img', 'ref_expr_inds', 'gt_bbox'])
]
val_pipeline = [
    dict(type='LoadImageAnnotationsFromFile',
         max_token=15, with_bbox=True, dataset="GRefCOCO"),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    # dict(type='Pad', pad_to_square=True),
    dict(type='DefaultFormatBundle'),
    dict(type='CollectData', keys=[
         'img', 'ref_expr_inds', 'gt_bbox'])
]
test_pipeline = val_pipeline.copy()

word_emb_cfg = dict(type='GloVe')
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type=dataset,
        which_set='train',
        img_source=['coco'],
        annsfile=data_root + 'annotations/grefs/instances.json',
        imgsfile=data_root + 'images/mscoco/train2014',
        pipeline=train_pipeline,
        word_emb_cfg=word_emb_cfg),
    val=dict(
        type=dataset,
        which_set='val',
        img_source=['coco'],
        annsfile=data_root + 'annotations/grefs/instances.json',
        imgsfile=data_root + 'images/mscoco/train2014',
        pipeline=val_pipeline,
        word_emb_cfg=word_emb_cfg),
    testA=dict(
        type=dataset,
        which_set='testA',
        img_source=['coco'],
        annsfile=data_root + 'annotations/grefs/instances.json',
        imgsfile=data_root + 'images/mscoco/train2014',
        pipeline=test_pipeline,
        word_emb_cfg=word_emb_cfg),
    testB=dict(
        type=dataset,
        which_set='testB',
        img_source=['coco'],
        annsfile=data_root + 'annotations/grefs/instances.json',
        imgsfile=data_root + 'images/mscoco/train2014',
        pipeline=test_pipeline,
        word_emb_cfg=word_emb_cfg)
)
