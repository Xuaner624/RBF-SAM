# -*- encoding: utf-8 -*-
'''
@File    :   offset_roi_head.py
@Time    :   2021/01/17 21:10:35
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2021
@Desc    :   RoI head for offset model training
'''

import numpy as np
import torch
from abc import abstractmethod

from mmdet.core import bbox2roi, bbox2result, roi2bbox
from ..builder import HEADS, build_head, build_roi_extractor
from .standard_roi_head import StandardRoIHead
from .test_mixins import OffsetTestMixin


@HEADS.register_module()
class LoftRoIHead(StandardRoIHead, OffsetTestMixin):
    """
        LoftRoIHead 类，继承自 StandardRoIHead 和 OffsetTestMixin。
        - 在常规目标检测（bbox）、分割（mask）的基础上，增加 offset 偏移预测功能。
        - 主要用于倾斜影像建筑物提取任务
    """

    def __init__(self,
                 offset_roi_extractor=None,  # ROI 特征提取器（用于 offset 分支）
                 offset_head=None,  # offset 分支的 head
                 **kwargs):
        assert offset_head is not None
        super(LoftRoIHead, self).__init__(**kwargs)

        if offset_head is not None:
            # 初始化 offset 分支
            self.init_offset_head(offset_roi_extractor, offset_head)

        # 是否可视化特征图（调试用）
        self.with_vis_feat = False

    def init_offset_head(self, offset_roi_extractor, offset_head):
        # 构建 offset 分支的 ROI 特征提取器
        self.offset_roi_extractor = build_roi_extractor(offset_roi_extractor)
        self.offset_head = build_head(offset_head)

    def init_weights(self, pretrained):
        # 初始化权重
        super(LoftRoIHead, self).init_weights(pretrained)
        self.offset_head.init_weights()

    def forward_train(self,
                      x,
                      img_metas,  # 图像元信息（大小、缩放比例、翻转等）
                      proposal_list,  # 候选框 proposals
                      gt_bboxes,  # 真实框 ground truth bboxes
                      gt_labels,  # 真实标签
                      gt_bboxes_ignore=None,  # 忽略的框
                      gt_masks=None,  # 实例分割掩膜
                      gt_offsets=None):  # 偏移量 ground truth
        """
        Args:
            x (list[Tensor]): list of multi-level img features.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposals (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        """
               前向训练过程：
               - 生成正负样本（assigner + sampler）
               - 分别计算 bbox、mask、offset 的 loss
        """
        # assign gts and sample proposals
        # ------------------ 1. 分配 proposals 与 GT ------------------
        # 如果模型有 bbox 分支 或 mask 分支，才需要做正负样本的分配
        if self.with_bbox or self.with_mask:
            # 当前 batch 中图像的数量（通常一个 batch 可以包含多张图像）
            num_imgs = len(img_metas)
            # 如果没有给出要忽略的 GT 框，就初始化为 None
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            # 用来存放每张图片的采样结果（正负样本）
            sampling_results = []
            for i in range(num_imgs):
                # ---------- 1.1 分配 (assign) ----------
                # bbox_assigner 的作用：
                #   将 proposals（候选框）与 gt_bboxes（真实框）进行匹配，
                #   判断每个 proposal 属于：
                #       - 正样本（与某个 GT 匹配，IoU 超过阈值）
                #       - 负样本（没有匹配上 GT，或者 IoU 低于阈值）
                #       - 忽略样本（gt_bboxes_ignore 中的框）
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i],  # 当前图片的候选框 proposals
                    gt_bboxes[i],  # 当前图片的真实框 GT
                    gt_bboxes_ignore[i],  # 需要忽略的 GT 框
                    gt_labels[i])  # 真实框对应的类别标签
                # ---------- 1.2 采样 (sample) ----------
                # bbox_sampler 的作用：
                #   在 assign_result 的基础上，从正负样本中按照一定比例采样，
                #   比如：正样本 128，负样本 128（具体由配置文件 train_cfg 决定），
                #   这样保证训练时正负样本平衡。
                sampling_result = self.bbox_sampler.sample(
                    assign_result,  # 上一步的分配结果
                    proposal_list[i],  # 当前图片的 proposals
                    gt_bboxes[i],  # 真实框
                    gt_labels[i],  # 真实类别标签
                    # feats: 当前图片在所有 FPN 特征层上的特征，用于后续的 RoIAlign 操作。
                    # lvl_feat[i][None] 的意思是：取出第 i 张图片的特征图，再在维度上加一层，
                    # 方便后续的 batch 操作
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # 初始化一个空字典，用于存放各个分支的损失函数结果
        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            # TODO: Support empty tensor input. #2280
            if mask_results['loss_mask'] is not None:
                losses.update(mask_results['loss_mask'])

        if self.with_offset:
            # print("mask_results['mask_pred']: ", mask_results['mask_pred'].shape)
            # print("mask_results['mask_targets']: ", mask_results['mask_targets'].shape)
            # print("bbox_results['bbox_feats']: ", bbox_results['bbox_feats'].shape)
            offset_results = self._offset_forward_train(x, sampling_results,
                                                        bbox_results['bbox_feats'],
                                                        gt_offsets, img_metas)
            # TODO: Support empty tensor input. #2280
            if offset_results['loss_offset'] is not None:
                losses.update(offset_results['loss_offset'])

        return losses

    def _offset_forward_train(self,
                              x,
                              sampling_results,
                              bbox_feats,
                              gt_offsets,
                              img_metas):
        # 将所有正样本的 bbox（即和 GT 对应上的 proposals）转换成 RoI 格式
        # bbox2roi 会把多个图片的 bboxes 拼接成一个二维张量，格式 [img_idx, x1, y1, x2, y2]
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        # if pos_rois.shape[0] == 0:
        #     return dict(loss_offset=None)
        # 前向传播 offset head，得到预测结果
        offset_results = self._offset_forward(x, pos_rois)
        # 生成 offset 的监督目标
        #   - sampling_results：里面保存了正负样本信息
        #   - gt_offsets：真实的偏移量标注
        #   - self.train_cfg：训练配置（比如正负样本比例、损失函数参数等）
        offset_targets = self.offset_head.get_targets(sampling_results, gt_offsets,
                                                      self.train_cfg)
        # 计算 offset 的损失函数
        # offset_results['offset_pred'] 是预测值
        # offset_targets 是监督目标
        loss_offset = self.offset_head.loss(offset_results['offset_pred'], offset_targets)

        offset_results.update(loss_offset=loss_offset, offset_targets=offset_targets)
        return offset_results

    def _offset_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        # 这里用到了逻辑异或 ^：
        #   - 要么传 rois（常见情况），
        #   - 要么传 pos_inds + bbox_feats（二选一，不能同时为空，也不能同时都有）
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        # 如果给的是 RoIs，则通过 RoIExtractor 从特征图中提取 offset 特征
        #   - x: FPN 多层特征
        #   - self.offset_roi_extractor.num_inputs 表示需要用几层特征
        if rois is not None:
            offset_feats = self.offset_roi_extractor(
                x[:self.offset_roi_extractor.num_inputs], rois)
        else:
            # 如果没有 rois，而是直接传进来 bbox_feats + pos_inds，
            # 就直接在已有的 bbox_feats 中取正样本的特征
            assert bbox_feats is not None
            offset_feats = bbox_feats[pos_inds]

        # self._show_offset_feat(rois, offset_feats)
        # 将特征输入 offset_head，得到 offset 的预测结果
        offset_pred = self.offset_head(offset_feats)
        offset_results = dict(offset_pred=offset_pred, offset_feats=offset_feats)
        return offset_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)
            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels, )

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if self.with_mask:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)

            if self.with_vis_feat:
                offset_results = self.simple_test_offset_rotate_feature(
                    x, img_metas, det_bboxes, det_labels, rescale=rescale)
                return bbox_results, segm_results, offset_results, self.vis_featuremap
            else:
                offset_results = self.simple_test_offset(
                    x, img_metas, det_bboxes, det_labels, rescale=rescale)

                return bbox_results, segm_results, offset_results
        else:
            offset_results = self.simple_test_offset(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)

            return bbox_results, None, offset_results
