import numpy as np
import mmcv
import torch
import cv2
import math

from ..builder import DETECTORS
from .loft import LOFT


@DETECTORS.register_module()
class prompt_LOFT(LOFT):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(prompt_LOFT, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        self.anchor_bbox_vis = [[287, 433, 441, 541]]
        self.with_vis_feat = True
        
    def simple_test(self, img, img_metas, gt_bboxes, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, gt_bboxes, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        # ------------------------------------------------------
        # Step 1: 重新提取特征（节省显存）
        # ------------------------------------------------------
        x = self.extract_feats(imgs)  # 多张图像提取特征
        x = x[0]  # 这里只保留第一张的特征（适配 ROI Head）
        # ------------------------------------------------------
        # Step 2: 处理 gt_bboxes，作为 proposals 使用
        # ------------------------------------------------------
        gt_bboxes = gt_bboxes[0]  # 取第一组增强对应的 GT 框
        gt_bboxes = [b.squeeze(0) for b in gt_bboxes]  # 移除多余维度
        # proposal_list: list(tensor(N,5))
        # 给每个 bbox 增加一个置信度维度（此处置为 1），
        # 模拟 RPN 生成的 proposals 格式：[x1, y1, x2, y2, score]
        gt_bboxes = [torch.cat((b, torch.ones(b.shape[0]).unsqueeze(1).cuda()), 1) for b in gt_bboxes]
        # p = p
        # gt_bboxes = torch.cat((gt_bboxes,p),1)
        # 将这些 GT 框作为 proposal_list 输入 ROI Head
        # ------------------------------------------------------
        # Step 3: 处理元信息并进入 ROI Head 的增强测试流程
        # ------------------------------------------------------
        proposal_list = gt_bboxes
        # return self.roi_head.aug_test(
        #     x, proposal_list, img_metas, rescale=rescale)
        # img_metas = [i[0] for i in img_metas]
        img_metas = img_metas[0]  # 取对应增强图像的元信息
        return self.roi_head.aug_test2(x, proposal_list, img_metas, rescale=rescale)
    
    

        

