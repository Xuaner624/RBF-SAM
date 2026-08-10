# 每隔 1 个 epoch 保存一次模型权重
checkpoint_config = dict(interval=1)
# yapf:disable  表示下面的代码块不使用 yapf 自动格式化
log_config = dict(
    interval=10,  # 每隔 10 个 iteration 打印一次日志
    hooks=[
        dict(type='TextLoggerHook'),  # 使用文本日志输出
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

# 分布式训练参数，通信后端使用 NCCL
dist_params = dict(backend='nccl')
# 日志级别设为 INFO
log_level = 'INFO'
# 不从已有模型加载（设为路径可加载预训练模型）
load_from = None
# 不从之前的训练断点继续训练（设为路径可恢复断点）
resume_from = None
# 训练流程配置：表示执行 'train' 阶段，循环 1 次（即每个 epoch 运行一次训练）
workflow = [('train', 1)]
