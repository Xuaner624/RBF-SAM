import torch
from torch.nn import functional as F
from typing import List, Optional
import torch.nn as nn
from ..builder import LOSSES


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    # 做归一化映射
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def get_uncertain_point_coords_with_randomness(
        coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.
    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.
    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    # batch size
    num_boxes = coarse_logits.shape[0]
    # 计算过采样后的点数
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    # [N, 1, num_sampled]
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def loss_masks(src_masks, target_masks, num_masks, oversample_ratio=3.0):
    """Compute the losses related to the masks: the focal loss and the dice loss.
    targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
    """

    # No need to upsample predictions as we are using normalized coordinates :)

    with torch.no_grad():
        # sample point_coords
        point_coords = get_uncertain_point_coords_with_randomness(
            src_masks,
            lambda logits: calculate_uncertainty(logits),
            112 * 112,
            oversample_ratio,
            0.75,
        )
        # get gt labels
        point_labels = point_sample(
            target_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

    point_logits = point_sample(
        src_masks,
        point_coords,
        align_corners=False,
    ).squeeze(1)

    loss_mask = sigmoid_ce_loss_jit(point_logits, point_labels, num_masks)
    loss_dice = dice_loss_jit(point_logits, point_labels, num_masks)

    del src_masks
    del target_masks
    return loss_mask, loss_dice


@LOSSES.register_module()
class SAMHQLoss(nn.Module):
    def __init__(self, oversample_ratio=3.0,
                 diceloss_weight=1.0,
                 sigmoidcdloss_weight=1.0):
        super(SAMHQLoss, self).__init__()
        # 过采样比例，用于在采样点时增加不确定区域的采样概率
        self.oversample_ratio = oversample_ratio
        self.diceloss_weight = diceloss_weight
        self.sigmoidcdloss_weight = sigmoidcdloss_weight

    def forward(self, src_masks, target_masks):
        # [N, H, W]->[N, 1, H, W]
        if src_masks.dim() == 3:
            src_masks = src_masks[:, None, :, :]
        if target_masks.dim() == 3:
            target_masks = target_masks[:, None, :, :]
        # 记录 batch 中 mask 的数量
        num_masks = len(src_masks)
        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                112 * 112,
                self.oversample_ratio,
                0.75,
            )
            # get gt labels

            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        loss_mask = sigmoid_ce_loss_jit(point_logits, point_labels, num_masks) * self.sigmoidcdloss_weight
        loss_dice = dice_loss_jit(point_logits, point_labels, num_masks)*self.diceloss_weight

        del src_masks
        del target_masks
        return loss_mask, loss_dice


class BCELoss_ignore(nn.Module):
    def __init__(self, loss_weight=torch.tensor(1), ignore_index=None):
        super(BCELoss_ignore, self).__init__()
        self.ignore_index = ignore_index
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss(pos_weight=loss_weight, reduction='none')
        # self.bce = nn.BCEWithLogitsLoss(pos_weight=loss_weight)

    def forward(self, logits, targets):
        loss = self.bce_with_logits_loss(logits, targets)
        if self.ignore_index is not None:
            mask = torch.ones_like(targets, dtype=torch.float32)
            mask[targets == self.ignore_index] = 0
            loss = loss * mask
            loss = loss.sum() / mask.sum()
        # loss1 = self.bce(logits, targets)
        return loss


def dice_loss_ignore(input, target, ignore_index=None, smooth=1.0):
    """
    Computes the Dice Loss.

    Args:
    - input (torch.Tensor): Predicted tensor of shape (N, *).
    - target (torch.Tensor): Ground truth tensor of shape (N, *).
    - ignore_index (int, optional): Specifies a target value that is ignored
                                    and does not contribute to the input gradient.
    - smooth (float, optional): Smoothing value to avoid division by zero. Default is 1.0.

    Returns:
    - torch.Tensor: Dice loss.
    """

    n = input.size(0)  # Batch size
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)

    if ignore_index is not None:
        mask = (tflat != ignore_index)
        iflat = iflat * mask
        tflat = tflat * mask

    intersection = (iflat * tflat).sum(1)
    dice_score = (2. * intersection + smooth) / (iflat.sum(1) + tflat.sum(1) + smooth)
    loss = 1 - dice_score

    return loss.mean()


@LOSSES.register_module()
class OBM_Wavelet_Loss(nn.Module):
    def __init__(self, loss_weight=torch.tensor(1), ignore_index=-1, expand=3):
        super(OBM_Wavelet_Loss, self).__init__()

        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.expand = expand
        # BCE with ignore
        self.bcecriterion = BCELoss_ignore(loss_weight=loss_weight, ignore_index=ignore_index)

    def forward(self, inputs, target, gt_boxes):
        """
        返回: bce_loss, dice_loss
        """

        # ---- 处理 target 形状 ----
        if target.dim() == 3:
            target = target.unsqueeze(1)
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(1)
        B, C, H_pred, W_pred = inputs.shape
        is_1024 = (H_pred == 1024 and W_pred == 1024)
        if not is_1024:
            target = F.interpolate(
                target.float(),
                size=(H_pred, W_pred),
                mode='nearest'
            )
        else:
            target = target.float()

        ignore_mask = torch.zeros_like(target, dtype=torch.bool)

        for i in range(B):
            x1, y1, x2, y2 = gt_boxes[i]
            expand = self.expand
            x1 = max(0, int(x1) - expand)
            y1 = max(0, int(y1) - expand)
            x2 = min(W_pred, int(x2) + expand)
            y2 = min(H_pred, int(y2) + expand)
            # box 区域内保持原样，box 外全部 ignore
            ignore_mask[i, 0, :y1, :] = True
            ignore_mask[i, 0, y2:, :] = True
            ignore_mask[i, 0, y1:y2, :x1] = True
            ignore_mask[i, 0, y1:y2, x2:] = True

        # 设置为 ignore_index = -1
        target = target.clone()  # 防止原地修改
        target[ignore_mask] = -1

        # ---- BCE ----
        loss_bce = self.bcecriterion(inputs.squeeze(1), target.squeeze(1))

        # ---- Dice ----
        prob = torch.sigmoid(inputs.squeeze(1))
        loss_dice = dice_loss_ignore(
            prob, target.squeeze(1).float(),
            ignore_index=self.ignore_index
        )

        del inputs
        del target
        del gt_boxes
        return loss_bce, loss_dice