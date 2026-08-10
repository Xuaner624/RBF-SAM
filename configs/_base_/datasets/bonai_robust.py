dataset_type = 'BONAI'
data_root = '/root/autodl-tmp/data/BONAI/'
# data_root = 'D:/off_nadir_building/data/BONAI/'
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
cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
# cities = ['shanghai', 'beijing', 'haerbin', 'chengdu']

train_ann_file = []    # 训练集标注文件路径
train_img_prefix = []        # 训练集图像路径

val_ann_file = []      # 验证集标注文件路径
val_img_prefix = []  # 验证集图像路径

test_ann_file = []
test_img_prefix = []

for city in cities:
    train_ann_file.append(data_root + 'coco/bonai_{}_trainval.json'.format(city))
    train_img_prefix.append(data_root + "trainval/images/")

val_ann_file.append(data_root + 'coco/bonai_jinan_trainval.json')
val_img_prefix.append(data_root + "/trainval/images/")

test_ann_file.append(data_root + 'coco/bonai_shanghai_xian_test.json')
test_img_prefix.append(data_root + "/test")

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=train_ann_file,
        img_prefix=train_img_prefix,
        bbox_type='building',
        mask_type='roof',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=val_ann_file[0],
        img_prefix=val_img_prefix[0],
        gt_footprint_csv_file="",
        bbox_type='building',
        mask_type='roof',
        pipeline=test_pipeline),

    test=dict(
        type=dataset_type,
        ann_file=test_ann_file[0],
        img_prefix=test_img_prefix[0],
        gt_footprint_csv_file="",
        bbox_type='building',
        mask_type='roof',
        pipeline=test_pipeline),
    )
evaluation = dict(interval=50, metric=['bbox', 'segm'])