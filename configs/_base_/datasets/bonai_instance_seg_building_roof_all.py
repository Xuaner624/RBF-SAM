# 数据集类型，定义为 BONAI_OBM
# dataset_type = 'BONAI_OBM'
dataset_type = 'BONAI_OBM'

# 数据根目录
data_root = r'D:\off_nadir_building\data\BONAI'

# # ImageNet 上常用的归一化均值和方差（基于 RGB 通道的统计值）
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# ImageNet 上常用的归一化均值和方差（基于 RGB 三通道的统计值）
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],  # 均值
    std=[58.395, 57.12, 57.375],  # 方差
    to_rgb=True  # 是否转换为 RGB 格式
)

# -------------------- 训练数据处理流程 --------------------
train_pipeline = [
    # 从文件加载图像
    dict(type='LoadImageFromFile'),

    dict(type='LoadAnnotations',
         with_bbox=True,  # 加载边界框
         with_mask=True,  # 加载分割掩膜
         with_offset=True,  # 加载偏移量信息
         with_building_mask=True),  # 加载建筑物掩膜

    # dict(type='Corrupt',
    #     corruption='gaussian_noise',ge
    #     severity=1),
    # dict(type='Corrupt',
    #     corruption='brightness',
    #     severity=1),

    # 数据增强：随机水平/垂直翻转，概率 0.5
    dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal', 'vertical']),
    # 调整图像大小为 (1024, 1024)，保持比例
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    # 图像归一化
    dict(type='Normalize', **img_norm_cfg),
    # 填充到 32 的倍数，方便模型输入
    dict(type='Pad', size_divisor=32),
    # 转换为默认格式
    dict(type='DefaultFormatBundle'),
    # 收集需要的字段，供模型输入
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_offsets', 'gt_building_masks']),
]


# -------------------- 测试/验证数据处理流程(有TTA) --------------------
test_pipeline = [
    # 从文件加载图像
    dict(type='LoadImageFromFile'),

    dict(type='LoadAnnotations',
         with_bbox=True,  # 只加载边界框
         with_mask=False,  # 不加载分割掩膜
         with_offset=False,  # 不加载偏移量
         with_building_mask=False),  # 不加载建筑物掩膜

    dict(
        type='MultiScaleFlipAug',    # 命名
        img_scale=(1024, 1024),      # 图像缩放大小
        flip=False,                   # 是否开启 TTA（测试时增强）
        transforms=[
            dict(type='Resize', keep_ratio=True),            # 调整大小，保持比例
            dict(type='RandomFlip', flip_ratio=0.5),         # 随机翻转
            dict(type='Normalize', **img_norm_cfg),          # 归一化
            dict(type='Pad', size_divisor=32),               # 填充
            dict(type='ImageToTensor', keys=['img']),        # 转换为 Tensor
            dict(type='Collect', keys=['img', 'gt_bboxes'])  # 收集图像和边界框
        ])
]


# # -------------------- 测试/验证数据处理流程（无 TTA） --------------------
# test_pipeline = [
#     # 从文件加载图像
#     dict(type='LoadImageFromFile'),
#
#     dict(type='LoadAnnotations',
#          with_bbox=True,   # 只加载边界框
#          with_mask=False,  # 不加载分割掩膜
#          with_offset=False,  # 不加载偏移量
#          with_building_mask=False),  # 不加载建筑物掩膜
#
#     dict(
#         type='MultiScaleFlipAug',      # 命名
#         img_scale=(1024, 1024),        # 单一尺度
#         flip=False,                    # 关闭翻转增强（无 TTA）
#         transforms=[
#             dict(type='Resize', keep_ratio=True),            # 调整大小，保持比例
#             dict(type='Normalize', **img_norm_cfg),          # 归一化
#             dict(type='Pad', size_divisor=32),               # 填充
#             dict(type='ImageToTensor', keys=['img']),        # 转换为 Tensor
#             dict(type='Collect', keys=['img', 'gt_bboxes'])  # 收集图像（无 gt_bboxes）
#         ])
# ]


# -------------------- 数据集路径配置 --------------------
cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']  # 训练城市
# cities = ['shanghai', 'beijing', 'haerbin', 'chengdu']

train_ann_file = []    # 训练集标注文件路径
img_prefix = []        # 训练集图像路径

val_ann_file = []      # 验证集标注文件路径
val_image_prefix = []  # 验证集图像路径

# 遍历城市，构造训练集的标注文件和图像路径
# for city in cities:
#     train_ann_file.append(data_root + 'coco_building_seg/building_seg_bonai_{}_trainval.json'.format(city))
#     img_prefix.append("../BONAI_data/trainval-001/trainval/images")
for city in cities:
    train_ann_file.append(data_root + 'coco/bonai_{}_trainval.json'.format(city))
    img_prefix.append(data_root + "/trainval/images")

# 验证集：只使用济南 (jinan)
# val_ann_file.append(data_root + 'coco_building_seg/building_seg_bonai_jinan_trainval.json')
# val_image_prefix.append("../BONAI_data/trainval-001/trainval/images")
val_ann_file.append(data_root + 'coco/bonai_jinan_trainval.json')
val_image_prefix.append(data_root + "/trainval/images")


# -------------------- 数据加载配置 --------------------
data = dict(
    samples_per_gpu=1,   # 每个 GPU 的 batch size
    workers_per_gpu=1,   # 每个 GPU 的 dataloader 线程数
    train=dict(
        type=dataset_type,         # 数据集类型
        ann_file=train_ann_file,   # 标注文件（多个城市）
        img_prefix=img_prefix,     # 图像路径
        bbox_type='building',      # 边界框类型：建筑物
        mask_type='roof',          # 掩膜类型：屋顶
        pipeline=train_pipeline),  # 训练数据处理流程
    val=dict(
        type=dataset_type,
        ann_file=val_ann_file[0],
        img_prefix=val_image_prefix[0],
        gt_footprint_csv_file="",
        bbox_type='building',
        mask_type='roof',
        pipeline=test_pipeline),

    test=dict(
        type='BONAI',
        # ann_file='/config_data/BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco_building_seg/building_seg_bonai_shanghai_xian_test.json',
        # ann_file='../BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco/bonai_shanghai_xian_test.json',
        # img_prefix='../BONAI_data/BONAI-20230403T091731Z-002/BONAI/test/test',
        ann_file=data_root + '/coco/bonai_shanghai_xian_test.json',
        img_prefix=data_root + "/test",
        gt_footprint_csv_file="",
        bbox_type='building',
        mask_type='roof',
        pipeline=test_pipeline),
    # val=dict(
    #     type=dataset_type,
    #     ann_file='/config_data/BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco/bonai_shanghai_xian_test.json',
    #     img_prefix='/config_data/BONAI_data/BONAI-20230403T091731Z-002/BONAI/test/test',
    #     gt_footprint_csv_file="",
    #     bbox_type='building',
    #     mask_type='roof',
    #     pipeline=test_pipeline)
)


# -------------------- 评估配置 --------------------
# 评估配置：每 1 个 epoch 评估一次，指标包括 bbox 和 segm
evaluation = dict(interval=1, metric=['bbox', 'segm'])
