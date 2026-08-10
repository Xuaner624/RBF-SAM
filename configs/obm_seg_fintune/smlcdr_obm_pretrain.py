"""The training process of OBM is divided into two stages.
First stage mainly concentrates on those long offsets, the next
stage trained on all data. Both stage needs 48 epochs."""

# 基础配置文件列表，当前配置会继承这些基础配置
_base_ = [
    '../_base_/models/obm_seg.py',
    '../_base_/datasets/bonai_instance_seg_building_roof_all.py',
    '../_base_/schedules/schedule_4x.py',
    '../_base_/default_runtime.py'
]
# 预训练模型路径，用于加载初始权重, First stage mainly concentrates on those long offsets
load_from = 'pretrained/obm_long.pth'

# 训练配置
train_cfg = dict(
    max_num=60,  # 每张图像最多采样的目标数量（比如最多分割 60 个实例）
)


# 模型结构配置（在基础模型 obm_seg.py 的基础上进行修改）
model = dict(
    mask_decoder=dict(
        offset_aug=[dict(
            type='DeltaXYOffsetCoder_Transformer',
            image_size=(150, 150),
            target_means=[0.0, 0.0],
            target_stds=[0.5, 0.5]),
            dict(
                type='DeltaXYOffsetCoder_Transformer',
                image_size=(300, 300),
                target_means=[0.0, 0.0],
                target_stds=[0.5, 0.5]),
            dict(
                type='DeltaXYOffsetCoder_Transformer',
                image_size=(400, 400),
                target_means=[0.0, 0.0],
                target_stds=[0.5, 0.5]),
        ],
        # hidden_dim = 1024,
    )

)
