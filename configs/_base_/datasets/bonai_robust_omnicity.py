dataset_type = 'BONAI'
data_root = '/root/autodl-tmp/data/OmniCityView3WithOffset/'
# data_root = 'D:/off_nadir_building/data/OmniCityView3WithOffset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations',
         with_bbox=True,
         with_mask=True,
         with_offset=True,
         with_roof_bbox=True,
         with_footprint_bbox=True,
         ),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal', 'vertical']),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_offsets', 'gt_roof_bboxes', 'gt_footprint_bboxes']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations',
         with_bbox=True,
         with_mask=False,
         with_offset=False,
         with_roof_bbox=True,
         with_footprint_bbox=True,
         ),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_roof_bboxes', 'gt_footprint_bboxes']),
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/coco/OmniCityView3WithOffset_trainval.json',
        img_prefix=data_root + "/trainval/images",
        bbox_type='building',
        mask_type='roof',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/coco/OmniCityView3WithOffset_test_roof.json',
        img_prefix=data_root + "/test/images",
        gt_footprint_csv_file="",
        bbox_type='building',
        mask_type='roof',
        pipeline=test_pipeline),

    test=dict(
        type=dataset_type,
        ann_file=data_root + '/coco/OmniCityView3WithOffset_test_roof.json',
        img_prefix=data_root + "/test/images",
        gt_footprint_csv_file="",
        bbox_type='building',
        mask_type='roof',
        pipeline=test_pipeline),
    )
evaluation = dict(interval=50, metric=['bbox', 'segm'])