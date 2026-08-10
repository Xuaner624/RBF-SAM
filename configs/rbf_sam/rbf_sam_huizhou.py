_base_ = [
    '../_base_/models/rbf_sam.py',
    '../_base_/datasets/bonai_robust_huizhou.py',
    '../_base_/schedules/schedule_4x_bonai.py',
    '../_base_/default_runtime.py'
]

load_from = 'pretrain/sam_vit_b_01ec64.pth'

find_unused_parameters = True