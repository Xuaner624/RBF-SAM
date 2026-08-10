from mmcv.runner import auto_fp16
import torch
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from torch import nn
from torch.nn import functional as F
import random
from mmdet.models.backbones import ImageEncoderViT
from mmdet.models.dense_heads import PromptEncoder
from mmdet.utils import get_root_logger
from typing import Any, Dict, List, Tuple
from ..builder import DETECTORS
from ..backbones import ImageEncoderViT_Robust, Wavelet_Block, Feature_Resample_Block
from ..dense_heads import PromptEncoder_Robust
from ..roi_heads import MaskDecoder_Robust
from .base import BaseDetector
from mmdet.core import build_sampler, build_bbox_coder
from copy import deepcopy
from mmdet.models.builder import build_loss
import hashlib
import cv2
import os

def select_valid_box(
        i,
        gt_bboxes,
        gt_roof_bboxes,
        gt_footprint_bboxes,
        gt_bboxes_perturb,
):
    """
    从四类 bbox 中随机选择一个，
    若 width 或 height 小于等于 1，则自动使用 gt_bboxes（building）。
    box_type:
        0 -> building (gt_bboxes)
        1 -> roof
        2 -> footprint
        3 -> perturb
    """

    candidates = [
        (gt_bboxes[0][i:i + 1], 0),  # building
        (gt_roof_bboxes[0][i:i + 1], 1),  # roof
        (gt_footprint_bboxes[0][i:i + 1], 2),  # footprint
        (gt_bboxes_perturb[i:i + 1], 3),  # perturb
    ]
    selected_box, box_type = random.choice(candidates)
    x1 = selected_box[..., 0]
    y1 = selected_box[..., 1]
    x2 = selected_box[..., 2]
    y2 = selected_box[..., 3]
    w = x2 - x1
    h = y2 - y1

    if (w <= 1).any() or (h <= 1).any():
        selected_box = gt_bboxes[0][i:i + 1]
        box_type = 0

    return selected_box, box_type


def perturb_bbox(bboxes, max_shift=4, img_size=1024):
    """
    bboxes: [N,4] (x1, y1, x2, y2)
    max_shift: 单方向最大随机变化像素
    """
    shift = (torch.rand_like(bboxes) * 2 - 1) * max_shift
    shift = shift.round().to(torch.int)

    new_bboxes = bboxes.clone()
    new_bboxes[:, 0] += shift[:, 0]   # x1
    new_bboxes[:, 1] += shift[:, 1]   # y1
    new_bboxes[:, 2] += shift[:, 2]   # x2
    new_bboxes[:, 3] += shift[:, 3]   # y2

    new_bboxes[:, 0] = new_bboxes[:, 0].clamp(0, img_size - 1)
    new_bboxes[:, 1] = new_bboxes[:, 1].clamp(0, img_size - 1)

    new_bboxes[:, 2] = torch.max(new_bboxes[:, 2], new_bboxes[:, 0] + 1)
    new_bboxes[:, 3] = torch.max(new_bboxes[:, 3], new_bboxes[:, 1] + 1)

    new_bboxes[:, 2] = new_bboxes[:, 2].clamp(1, img_size)
    new_bboxes[:, 3] = new_bboxes[:, 3].clamp(1, img_size)

    return new_bboxes


@DETECTORS.register_module()
class rbf_sam(BaseDetector):
    mask_threshold: float = 0.0  # 掩码二值化阈值
    image_format: str = "RGB"  # 输入图像格式
    def __init__(
            self,
            image_encoder: ImageEncoderViT_Robust,
            wavelet_block: Wavelet_Block,
            resample_block: Feature_Resample_Block,
            prompt_encoder: PromptEncoder_Robust,
            mask_decoder: MaskDecoder_Robust,
            loss_masks=dict(type='SAMHQLoss', diceloss_weight=0.5, sigmoidcdloss_weight=10.0),
            loss_offset=dict(type='SmoothL1Loss', loss_weight=16.0),
            loss_offset_final=dict(type='SmoothL1Loss', loss_weight=32.0),
            offset_coder=dict(
                type='DeltaXYOffsetCoder_Transformer',
                image_size=(256, 256),
                target_means=[0.0, 0.0],
                target_stds=[0.5, 0.5]),
            pretrained=None,
            train_cfg=None,
            test_cfg=None,
            sampler=None,
            pixel_mean=[0, 0, 0],
            pixel_std=[1, 1, 1],
            noise_box=None,
    ):
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.sampler = build_sampler(sampler, context=self) if sampler is not None else None
        """-> None
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.noise_box = noise_box
        if isinstance(image_encoder, nn.Module):
            self.image_encoder = image_encoder
        else:
            self.image_encoder = build_backbone(image_encoder)
        if isinstance(wavelet_block, nn.Module):
            self.wavelet_block = wavelet_block
        else:
            self.wavelet_block = build_backbone(wavelet_block)
        if isinstance(resample_block, nn.Module):
            self.resample_block = resample_block
        else:
            self.resample_block = build_backbone(resample_block)
        if isinstance(prompt_encoder, nn.Module):
            self.prompt_encoder = prompt_encoder
        else:
            self.prompt_encoder = build_head(prompt_encoder)
        if isinstance(mask_decoder, nn.Module):
            self.mask_decoder = mask_decoder
        else:
            self.mask_decoder = build_head(mask_decoder)
        self.offset_coder = build_bbox_coder(offset_coder)
        self.loss_offset = build_loss(loss_offset)
        self.loss_offset_final = build_loss(loss_offset_final)
        self.loss_mask_roof = build_loss(loss_masks)
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        # self.freeze_model()

    @property
    def device(self):  # -> Any
        return self.pixel_mean.device

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_roof_bboxes=None,
                      gt_footprint_bboxes=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_offsets=None,
                      **kwargs):

        assert gt_masks or gt_bboxes
        # ====== 1. 提取图像特征 ======
        features, early_embedding = self.image_encoder(img)
        enhanced_features, wavelet_feature = self.wavelet_block(early_embedding, features)
        vit_sample = self.resample_block(features)

        # ====== 2. 用于收集所有 box 的预测 ======
        offset_global_preds = []
        offset_ll_preds = []
        offset_lh_preds = []
        offset_hl_preds = []
        offset_hh_preds = []
        offset_final_preds = []
        mask_sam_preds = []
        mask_enhanced_preds = []

        # ----------- 如果目标数仍然过多，再随机截断 -----------
        prompt_len = len(gt_labels[0])
        gt_bboxes_perturb = perturb_bbox(gt_bboxes[0], max_shift=15, img_size=1024)

        if prompt_len > self.train_cfg['max_num']:
            # 随机采样 max_num 个目标
            loc = random.sample(range(0, prompt_len), self.train_cfg['max_num'])
            # 按采样索引截断GT
            gt_bboxes[0] = gt_bboxes[0][loc]
            gt_roof_bboxes[0] = gt_roof_bboxes[0][loc]
            gt_footprint_bboxes[0] = gt_footprint_bboxes[0][loc]
            gt_labels[0] = gt_labels[0][loc]
            gt_masks[0] = gt_masks[0][loc]
            gt_offsets[0] = gt_offsets[0][loc]
            gt_bboxes_perturb = gt_bboxes_perturb[loc]

        # ===== 生成每个 box 的随机候选来源 =====
        num_boxes = len(gt_bboxes[0])
        box_candidates_list = []
        for i in range(num_boxes):
            selected_box, box_type = select_valid_box(
                i,
                gt_bboxes,
                gt_roof_bboxes,
                gt_footprint_bboxes,
                gt_bboxes_perturb
            )
            box_candidates_list.append({
                "box": selected_box,
                "box_type": box_type
            })

        # ===== 3. 对 batch 中每个 box 独立训练 =====
        for i, box_item in enumerate(box_candidates_list):
            sparse_embeddings_i, dense_embeddings_i = self.prompt_encoder(
                points=None,
                boxes=box_item["box"],
                masks=None,
                box_type=box_item["box_type"],
            )
            masks_sam_i, masks_enhanced_i, prob_i, offset_list_i = self.mask_decoder.forward_train(
                image_embeddings=features,
                enhanced_features=enhanced_features,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings_i,
                dense_prompt_embeddings=dense_embeddings_i,
                wavelet_offset_feature=wavelet_feature,
                vit_sample_feature=vit_sample,
                multimask_output=False,
                **kwargs
            )
            offset_global_preds.append(offset_list_i[0])
            offset_ll_preds.append(offset_list_i[1])
            offset_lh_preds.append(offset_list_i[2])
            offset_hl_preds.append(offset_list_i[3])
            offset_hh_preds.append(offset_list_i[4])
            offset_final_preds.append(offset_list_i[5])
            mask_sam_preds.append(masks_sam_i)
            mask_enhanced_preds.append(masks_enhanced_i)

        # ===== 4. 拼接所有 box 的预测 =====
        pred_masks_sam_cat = torch.cat(mask_sam_preds, dim=0)
        pred_masks_enhanced_cat = torch.cat(mask_enhanced_preds, dim=0)

        # ====== 5. 构造 offset target ======
        offset_targets = self.offset_coder.encode(gt_offsets)
        # ===== 6. offset loss =====
        offset_global = torch.cat(offset_global_preds, dim=0)
        offset_ll = torch.cat(offset_ll_preds, dim=0)
        offset_lh = torch.cat(offset_lh_preds, dim=0)
        offset_hl = torch.cat(offset_hl_preds, dim=0)
        offset_hh = torch.cat(offset_hh_preds, dim=0)
        offset_final = torch.cat(offset_final_preds, dim=0)
        loss_offset_global = self.loss_offset(offset_global, offset_targets)
        loss_offset_ll = self.loss_offset(offset_ll, offset_targets)
        loss_offset_lh = self.loss_offset(offset_lh, offset_targets)
        loss_offset_hl = self.loss_offset(offset_hl, offset_targets)
        loss_offset_hh = self.loss_offset(offset_hh, offset_targets)
        loss_offset_final = self.loss_offset_final(offset_final, offset_targets)

        # ===== 7. mask loss =====
        # roof_predict = self.postprocess_masks(roof_predict, (1024, 1024), (1024, 1024)).squeeze(1)
        gt_masks = self.get_masks(gt_masks[0], pred_masks_sam_cat, pred_masks_sam_cat.shape[-1] / gt_masks[0].width)
        loss_mask_roof_sam, loss_dice_roof_sam = self.loss_mask_roof(pred_masks_sam_cat, gt_masks.float())
        loss_mask_roof_enhanced, loss_dice_roof_enhanced = self.loss_mask_roof(pred_masks_enhanced_cat, gt_masks.float())

        # ===== 8. 输出所有 loss =====
        losses = dict(
            loss_offset_final=loss_offset_final,
            loss_offset_global=loss_offset_global,
            loss_offset_ll=loss_offset_ll,
            loss_offset_lh=loss_offset_lh,
            loss_offset_hl=loss_offset_hl,
            loss_offset_hh=loss_offset_hh,
            loss_mask_roof_sam=loss_mask_roof_sam,
            loss_dice_roof_sam=loss_dice_roof_sam,
            loss_mask_roof_enhanced=loss_mask_roof_enhanced,
            loss_dice_roof_enhanced=loss_dice_roof_enhanced,
        )
        return losses


    def forward_test(self,
                     img,
                     img_metas,
                     gt_bboxes=None,
                     gt_labels=None,
                     gt_bboxes_ignore=None,
                     gt_roof_bboxes=None,
                     gt_footprint_bboxes=None,
                     gt_masks=None,
                     proposals=None,
                     **kwargs):

        # ===== 固定随机种子 =====
        BASE_SEED = 42
        filename = img_metas[0][0]["filename"]
        hash_obj = hashlib.md5(filename.encode('utf-8'))
        fname_hash_int = int(hash_obj.hexdigest(), 16)
        img_seed = (BASE_SEED + fname_hash_int) % (2**32 - 1)
        torch.manual_seed(img_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(img_seed)

        # ===== 0. 备份box，选择不同框提示的时候记得更换！！gt_bboxes默认为gt_building_boxes =====
        return_box = deepcopy(gt_bboxes)
        gt_bboxes_perturb = perturb_bbox(gt_bboxes[0][0], max_shift=15, img_size=1024)
        return_box_robust = deepcopy(gt_bboxes)
        # return_box_robust = deepcopy(gt_roof_bboxes)
        # return_box_robust = deepcopy(gt_footprint_bboxes)
        # return_box_robust = [[deepcopy(gt_bboxes_perturb)]]

        # ===== 1. 图像特征 =====
        features, early_embedding = self.image_encoder(img[0])
        enhanced_features, wavelet_feature = self.wavelet_block(
            early_embedding, features
        )
        vit_sample = self.resample_block(features)
        # ===== 2. 选择不同框提示的时候记得更换！！Prompt Encoder（全部 batch）=====
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=gt_bboxes[0][0],
            masks=None,
            box_type=0,
        )
        # sparse_embeddings, dense_embeddings = self.prompt_encoder(
        #     points=None,
        #     boxes=gt_roof_bboxes[0][0],
        #     masks=None,
        #     box_type=1,
        # )
        # sparse_embeddings, dense_embeddings = self.prompt_encoder(
        #     points=None,
        #     boxes=gt_footprint_bboxes[0][0],
        #     masks=None,
        #     box_type=2,
        # )
        # sparse_embeddings, dense_embeddings = self.prompt_encoder(
        #     points=None,
        #     boxes=gt_bboxes_perturb,
        #     masks=None,
        #     box_type=3,
        # )
        # ===== 3. Mask Decoder（全部 batch）=====
        masks_sam, masks_enhanced, prob, offset_list = \
            self.mask_decoder.forward_test(
                image_embeddings=features,
                enhanced_features=enhanced_features,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                wavelet_offset_feature=wavelet_feature,
                vit_sample_feature=vit_sample,
                multimask_output=False,
            )
        # ===== 4. Offset decode（全部 batch）=====
        offset_global = self.offset_coder.decode(offset_list[0])
        offset_ll = self.offset_coder.decode(offset_list[1])
        offset_lh = self.offset_coder.decode(offset_list[2])
        offset_hl = self.offset_coder.decode(offset_list[3])
        offset_hh = self.offset_coder.decode(offset_list[4])
        offset_final = self.offset_coder.decode(offset_list[5])
        # # ===== 5. Mask 后处理（OmniCity-view3） =====
        # pred_masks_cat = torch.cat(
        #     [masks_sam, masks_enhanced], dim=1
        # )  # [B, 2, H, W]
        # pred_masks_cat = self.postprocess_masks(
        #     pred_masks_cat, (1024, 1024), (512, 512)
        # )
        # pred_masks_cat = (
        #     pred_masks_cat[:, :2, :, :].mean(dim=1, keepdim=True)
        # )
        # pred_masks_cat = pred_masks_cat > self.mask_threshold
        # # ===== 6. 返回（OmniCity-view3） =====
        # return return_box[0][0] / 2, pred_masks_cat, offset_final, offset_global, offset_ll, offset_lh, offset_hl, offset_hh, return_box_robust[0][0] / 2

        # ===== 5. Mask 后处理（BONAI、Huizhou） =====
        pred_masks_cat = torch.cat(
            [masks_sam, masks_enhanced], dim=1
        )  # [B, 2, H, W]
        pred_masks_cat = self.postprocess_masks(
            pred_masks_cat, (1024, 1024), (1024, 1024)
        )
        pred_masks_cat = (
            pred_masks_cat[:, :2, :, :].mean(dim=1, keepdim=True)
        )
        pred_masks_cat = pred_masks_cat > self.mask_threshold
        # ===== 6. 返回（BONAI、Huizhou）=====
        return return_box[0][0], pred_masks_cat, offset_final, offset_global, offset_ll, offset_lh, offset_hl, offset_hh, return_box_robust[0][0]


    def get_masks(self, gt_masks, masks, scale, ):
        device = masks.device
        mask = torch.from_numpy(gt_masks.masks).to(device)
        if scale == 1:
            mask = mask
        else:
         mask = F.interpolate(mask[:, None, :, :], scale_factor=scale, mode='nearest')
        return mask.squeeze(1) > 0

    def freeze_model(self):
        """
        冻结 image_encoder、prompt_encoder、decoder，
        decoder 中部分模块可手动排除。
        """
        decoder_unfreeze_list = [
            "mask_decoder.w1",
            "mask_decoder.w2",
            "mask_decoder.w3",
            "mask_decoder.w4",
            "mask_decoder.transformer.layers.0.cross_attn_token_to_image.mlp_k",
            "mask_decoder.transformer.layers.0.cross_attn_token_to_image.mlp_v",
            "mask_decoder.transformer.layers.1.cross_attn_token_to_image.mlp_k",
            "mask_decoder.transformer.layers.1.cross_attn_token_to_image.mlp_v",
            "mask_decoder.offset_query",
            "mask_decoder.offset_wavelet",
            "mask_decoder.offset_prediction_global",
            "mask_decoder.offset_prediction_ll",
            "mask_decoder.offset_prediction_lh",
            "mask_decoder.offset_prediction_hl",
            "mask_decoder.offset_prediction_hh",
            "mask_decoder.enhanced_token",
            "mask_decoder.enhance_mlp",
            "mask_decoder.embedding_maskfeature",
        ]

        for name, param in self.named_parameters():
            # ========== 1) 冻结 image_encoder + prompt_encoder ==========
            if name.startswith("image_encoder.") or name.startswith("prompt_encoder."):
                param.requires_grad = False
                continue

            # ========== 2) 冻结 mask_decoder（默认冻结） ==========
            if name.startswith("mask_decoder."):
                # 看看是否是我们要“排除、不冻结”的模块
                if any(name.startswith(prefix) for prefix in decoder_unfreeze_list):
                    param.requires_grad = True  # 特定子模块保持可训练
                else:
                    param.requires_grad = False  # 其他全部冻结
                continue

            # ========== 3) 其他模块（保持训练） ==========
            param.requires_grad = True

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
    ):
        """-> torch.Tensor
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    @auto_fp16(apply_to=('img',))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        This function mimics the inference forward pass (forward_test),
        but removes unnecessary logic such as noise, post-processing,
        and thresholding.

        Args:
            img (Tensor): Input tensor of shape (N, C, H, W).
            img_metas (list[dict], optional): Image metadata. If None,
                a dummy one will be created to avoid errors.
        Returns:
            tuple: dummy outputs (e.g., masks or feature maps)
        """

        dummy_box = torch.tensor(
            [[100.0, 100.0, 400.0, 400.0]],
            device=img.device
        ).repeat(50, 1).unsqueeze(0)
        gt_bboxes = [dummy_box]
        x, early_embedding = self.image_encoder(img)
        enhanced_features, wavelet_feature = self.wavelet_block(early_embedding, x)
        vit_sample = self.resample_block(x)
        num_boxes = len(gt_bboxes[0][0])
        for i in range(num_boxes):
            box_i = gt_bboxes[0][0][i:i + 1]

            sparse_embeddings_i, dense_embeddings_i = self.prompt_encoder(
                points=None,
                boxes=box_i,
                masks=None,
                box_type=0,
            )  # 1,2,256

            masks_sam_i, masks_enhanced_i, prob_i, offset_list_i = self.mask_decoder.forward_test(
                image_embeddings=x,
                enhanced_features=enhanced_features,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings_i,
                dense_prompt_embeddings=dense_embeddings_i,
                wavelet_offset_feature=wavelet_feature,
                vit_sample_feature=vit_sample,
                multimask_output=False,
            )
        return 0

    def aug_test(self, ):
        pass

    def extract_feat(self):
        pass

    def simple_test(self):
        pass
