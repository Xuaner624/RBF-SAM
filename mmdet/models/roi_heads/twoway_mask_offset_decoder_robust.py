# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from mmdet.core import auto_fp16
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from typing import List, Tuple, Type
from mmcv.cnn import normal_init
from ..utils.common import LayerNorm2d
from ..builder import HEADS
from ..builder import build_head

from mmdet.core import (build_bbox_coder, multi_apply, force_fp32, mask_target)
from mmdet.models.builder import HEADS, build_loss


@HEADS.register_module()
class MaskDecoder_Robust(nn.Module):
    # __init__方法:
    # 1. 输入参数:
    #     - transformer_dim: transformer 的通道维度
    #     - transformer: 使用的 transformer
    #     - num_multimask_outputs: 在消除掩码歧义时预测的掩码数量。
    #     - activation: 上采样掩码时使用的激活函数类型
    #     - iou_head_depth: 用于预测掩码质量的 MLP 的深度
    #     - iou_head_hidden_dim: 用于预测掩码质量的 MLP 的隐藏维度
    # 2. 记录 transformer_dim 和 transformer。
    # 3. 记录 num_multimask_outputs。
    # 4. 嵌入 iou_token 和 mask_tokens。
    # 5. 定义 output_upscaling 为上采样器,用于上采样 transformer 的输出以得到掩码。
    # 6. 定义 output_hypernetworks_mlps 为 MLP 列表,个数为 num_mask_tokens, 用于从 transformer 的输出生成掩码通道。
    # 7. 定义 iou_prediction_head 为 MLP,用于从 transformer 的输出预测掩码的 IOU。
    def __init__(
            self,
            *,
            transformer_dim: int,
            transformer: nn.Module,
            num_multimask_outputs: int = 3,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
            hidden_dim=256,
    ):
        """-> None
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        # offset_aug=[dict(
        #     type='DeltaXYOffsetCoder_Transformer',
        #     image_size = (100,100),
        #     target_means=[0.0, 0.0],
        #     target_stds=[0.5, 0.5]), dict(
        #     type='DeltaXYOffsetCoder_Transformer',
        #     image_size = (400,400),
        #     target_means=[0.0, 0.0],
        #     target_stds=[0.5, 0.5]),],
        """
        super().__init__()
        activation = nn.GELU
        self.transformer_dim = transformer_dim
        self.relu = nn.ReLU()
        if isinstance(transformer, dict):
            self.transformer = build_head(transformer)
        else:
            self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        self.offset_query = nn.Embedding(1, transformer_dim)
        nn.init.zeros_(self.offset_query.weight)
        self.offset_wavelet = nn.Embedding(4, transformer_dim)
        nn.init.zeros_(self.offset_wavelet.weight)
        self.offset_prediction_global = MLP(
            transformer_dim, hidden_dim, 2, 5
        )
        self.offset_prediction_ll = MLP(
            transformer_dim, hidden_dim, 2, 5
        )
        self.offset_prediction_lh = MLP(
            transformer_dim, hidden_dim, 2, 5
        )
        self.offset_prediction_hl = MLP(
            transformer_dim, hidden_dim, 2, 5
        )
        self.offset_prediction_hh = MLP(
            transformer_dim, hidden_dim, 2, 5
        )
        self.w1 = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        self.w3 = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        self.w4 = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        self.enhanced_token = nn.Embedding(1, transformer_dim)
        self.enhance_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1
        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 8, 3, 1, 1),
            LayerNorm2d(transformer_dim // 8),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 8, transformer_dim // 8, 3, 1, 1))

    def get_targets(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    def get_masks(self, gt_masks, masks, scale, ):
        device = masks.device
        mask = torch.from_numpy(gt_masks.masks).to(device)
        mask = F.interpolate(mask[:, None, :, :], scale_factor=scale, mode='nearest')
        return mask.squeeze(1) > 0

    def forward_test(
            self,
            image_embeddings,
            enhanced_features,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            wavelet_offset_feature,
            vit_sample_feature,
            multimask_output,
            # gt_boxes,
    ):
        masks_sam, masks_enhanced, prob, offset_list = self.predict_offset_masks(image_embeddings,
                                                                                 enhanced_features,
                                                                                 image_pe,
                                                                                 sparse_prompt_embeddings,
                                                                                 dense_prompt_embeddings,
                                                                                 wavelet_offset_feature,
                                                                                 vit_sample_feature,
                                                                                 multimask_output,
                                                                                 )
        return masks_sam, masks_enhanced, prob, offset_list

    @torch.no_grad()
    def iou(self, gt, pred):
        device = pred.device
        c, h, w = gt.shape
        gt = gt.view(c, -1).cpu().numpy() > 0.5
        pred = pred.view(c, -1).cpu().numpy() > 0.5
        intersection = np.bitwise_and(gt, pred)
        union = np.bitwise_or(gt, pred)
        iou_score = intersection.sum(axis=1) / union.sum(axis=1)
        return torch.from_numpy(iou_score).to(device)

    @torch.no_grad()
    def mask_iou(self, mask1, mask2):
        """
        mask1: [m1,n] m1 means number of predicted objects 
        mask2: [m2,n] m2 means number of gt objects
        Note: n means image_w x image_h
        """
        c, w, h = mask1.shape
        mask1, mask2 = mask1.view(c, w * h), mask2.view(c, w * h)
        intersection = torch.matmul(mask1, mask2.t())
        area1 = torch.sum(mask1, dim=1).view(1, -1)
        area2 = torch.sum(mask2, dim=1).view(1, -1)
        union = (area1.t() + area2) - intersection
        iou = intersection / union
        return iou

    @auto_fp16(apply_to=('image_embeddings', 'image_pe', 'sparse_prompt_embeddings', 'dense_prompt_embeddings'))
    def forward_train(self,
                      image_embeddings,
                      enhanced_features,
                      image_pe,
                      sparse_prompt_embeddings,
                      dense_prompt_embeddings,
                      wavelet_offset_feature,
                      vit_sample_feature,
                      multimask_output,
                      **kwargs):
        # print(f'length of offset: {gt_offsets[0].shape[0]}')
        masks_sam, masks_enhanced, prob, offset_list = self.predict_offset_masks(image_embeddings,
                                                                                 enhanced_features,
                                                                                 image_pe,
                                                                                 sparse_prompt_embeddings,
                                                                                 dense_prompt_embeddings,
                                                                                 wavelet_offset_feature,
                                                                                 vit_sample_feature,
                                                                                 multimask_output,)
        return masks_sam, masks_enhanced, prob, offset_list

    def predict_offset_masks(
            self,
            image_embeddings: torch.Tensor,
            enhanced_features: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            wavelet_offset_feature: torch.Tensor,
            vit_sample_feature: torch.Tensor,
            multimask_output: bool,
    ):
        """-> Tuple[torch.Tensor, torch.Tensor]
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          enhanced_features (torch.Tensor): the embeddings from the enhanced features
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          wavelet_offset_feature (torch.Tensor): the embeddings of the wavelet features
          vit_sample_feature (torch.Tensor): the embeddings of the resample features
          multimask_output (bool): Whether to return multiple masks or a single mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred, offset_tokens = self.predict_masks(
            image_embeddings=image_embeddings,
            enhanced_features=enhanced_features,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            wavelet_offset_feature=wavelet_offset_feature,
            vit_sample_feature=vit_sample_feature,
        )

        offset_global = self.offset_prediction_global(offset_tokens[0])
        offset_ll = self.offset_prediction_ll(offset_tokens[1])
        offset_lh = self.offset_prediction_lh(offset_tokens[2])
        offset_hl = self.offset_prediction_hl(offset_tokens[3])
        offset_hh = self.offset_prediction_hh(offset_tokens[4])
        offset_final = (self.w1 * offset_global + self.w2 * offset_ll +
                        self.w3 * offset_lh + self.w4 * offset_hl +
                        (1 - self.w1 - self.w2 - self.w3 - self.w4) * offset_hh)
        if multimask_output:
            # mask with highest score
            mask_slice = slice(1, self.num_mask_tokens - 1)
            iou_pred = iou_pred[:, mask_slice]
            iou_pred, max_iou_idx = torch.max(iou_pred, dim=1)
            iou_pred = iou_pred.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)), max_iou_idx].unsqueeze(1)
        else:
            # singale mask output, default
            mask_slice = slice(0, 1)
            masks_sam = masks[:, mask_slice]
        masks_enhanced = masks[:, slice(self.num_mask_tokens - 1, self.num_mask_tokens), :, :]
        offset_list = [offset_global, offset_ll, offset_lh, offset_hl, offset_hh, offset_final]
        return masks_sam, masks_enhanced, iou_pred, offset_list

    # 这个 predict_masks 方法的作用是预测掩码。它包含:
    # 1. 输入参数:
    #     - image_embeddings: 图像编码器的输出
    #     - image_pe: 与 image_embeddings 形状相同的位置编码
    #     - sparse_prompt_embeddings: 点和框的 embedding
    #     - dense_prompt_embeddings: 掩码输入的 embedding
    # 2. 拼接 iou_token 和 mask_tokens 作为输出 tokens, 扩展至 batch 大小, 与 sparse_prompt_embeddings 拼接作为 tokens。
    # 3. 通过 torch.repeat_interleave 扩展 src 和 pos_src 至与 tokens 相同的 batch 大小。
    # 4. 将 src 和 pos_src 以及 tokens 输入 transformer, 获得 hs 和 src。
    # 5. 获得 iou_token_out 和 mask_tokens_out 作为 transformer 的输出。
    # 6. 上采样 src 得到 upscaled_embedding。
    # 7. 对 mask_tokens_out 中的每个 token, 使用对应 MLP 得到 hyper_in_list 中的 tensor。
    # 8. 使用 torch.stack 将 hyper_in_list 拼接为 hyper_in。
    # 9. 计算 masks=(hyper_in @ upscaled_embedding.view(b, c, h * w)), 形状为 (b, num_mask_tokens, h, w)。
    # 10. 使用 iou_prediction_head 从 iou_token_out 预测 iou_pred。
    # 11. 返回 masks 和 iou_pred。
    # 所以,这个 predict_masks 方法实现了根据prompt预测掩码的功能。
    # 它发挥 transformer 和上采样器的功能,可以从 prompt 学习生成模型的参数
    # 这个 predict_masks 方法提供了根据 prompt 预测掩码的具体实现。
    # 它利用 MaskDecoder 的强大功能,可以解码出复杂的定制化掩码,为实现高质量的 prompt 驱动生成模型提供关键支持。

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,  # Bx(embed_dim=256 in vit-h)x(embed_H)x(embed_W)
            enhanced_features: torch.Tensor,
            image_pe: torch.Tensor,  # Bx(embed_dim=256 in vit-h)x(embed_H)x(embed_W)
            sparse_prompt_embeddings: torch.Tensor,  # BxNx(embed_dim)
            dense_prompt_embeddings: torch.Tensor,  # Bx(embed_dim)x(embed_H)x(embed_W)
            wavelet_offset_feature: torch.Tensor,
            vit_sample_feature: torch.Tensor,
    ):
        """-> Tuple[torch.Tensor, torch.Tensor]
        Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = [self.iou_token.weight, self.mask_tokens.weight, self.enhanced_token.weight, self.offset_query.weight]
        output_tokens = torch.cat(output_tokens, dim=0)  # [5,256] <- [1,256], [4,256], [1, 256]
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        wavelet_offset_token = wavelet_offset_feature + self.offset_wavelet.weight
        wavelet_offset_token = wavelet_offset_token.expand(sparse_prompt_embeddings.size(0), -1, -1)
        # (n, 5+2, 256) <- (n, 5, 256) + (n, 2, 256)
        tokens = torch.cat([output_tokens, wavelet_offset_token, sparse_prompt_embeddings], dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)  # (n,256,64,64) <- (1,256,64,64)
        vit_sample_feature = torch.repeat_interleave(vit_sample_feature, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)  # (n,256,64,64) <- (1,256,64,64)
        b, c, h, w = src.shape

        # Run the transformer
        # queries:(1,7,256) ,  keys:(1,4096,256)  <-  (图片特征， 位置信息，（output，prompt）)
        hs, src = self.transformer(src, pos_src, tokens, vit_sample_feature)
        iou_token_out = hs[:, 0, :]  # [1,256]
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]  # [1,4,256]
        offset_tokens_out = [hs[:, (1 + self.num_mask_tokens), :]]
        for i in range(wavelet_offset_token.shape[1]):
            offset_tokens_out.append(hs[:, (2 + self.num_mask_tokens + i), :])

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)  # [1,256,64,64] <- [1,4096,256]
        upscaled_embedding = self.output_upscaling(src)  # [1,32,256,256] <- [1,256,64,64]
        upscaled_embedding_enhanced = self.embedding_maskfeature(upscaled_embedding) + enhanced_features

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < 4:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))  # list: [[1,32]*4]
            else:
                hyper_in_list.append(self.enhance_mlp(mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [1,4,32]
        b, c, h, w = upscaled_embedding.shape  # [1,32,256,256]
        # [1,4,256,256] <- [1,4,256*256] <- [1,4,32] * [1,32,256*256]
        masks_sam = (hyper_in[:, :4] @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        masks_enhanced = (hyper_in[:, 4:] @ upscaled_embedding_enhanced.view(b, c, h * w)).view(b, -1, h, w)
        masks = torch.cat([masks_sam, masks_enhanced], dim=1)
        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)  # [1,4]<-[1,256]

        return masks, iou_pred, offset_tokens_out


class MLP(nn.Module):
    # __init__方法:
    # 1. 输入参数:
    #     - input_dim: 输入维度
    #     - hidden_dim: 隐藏层维度
    #     - output_dim: 输出维度
    #     - num_layers: 隐藏层数
    #     - sigmoid_output: 是否使用 sigmoid 激活函数
    # 2. 记录 num_layers 和 h 为 num_layers-1 个隐藏层维度。
    # 3. 实例化 nn.ModuleList 由 nn.Linear 组成的列表,用于实现 MLP 的线性变换。
    # 4. 记录 sigmoid_output 以决定是否使用 sigmoid 激活函数。
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    # forward 方法: 
    # 1. 对输入 x 重复 num_layers 次线性变换和激活。
    # 2. 最后一层只使用线性变换,不使用激活函数。
    # 3. 如果 sigmoid_output 为 True, 使用 sigmoid 激活函数。
    # 4. 返回 MLP 的输出。
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
