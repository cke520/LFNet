# utils/func.py
import torch
import torch.nn.functional as F
import torch.nn as nn


def clip_gradient(optimizer, grad_clip):
    parameters = []
    for group in optimizer.param_groups:
        parameters.extend(group['params'])
    nn.utils.clip_grad_norm_(parameters, grad_clip)


def weighted_bce_loss(pred, mask):
    """
    Weighted BCE Loss (Wei et al. CVPR2020)
    自动增大边缘像素的权重，SOTA 提分核心。
    """
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')

    # 动态计算权重: 局部方差越大(边缘)，权重越大
    ave = F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)
    weight = 1 + 5 * torch.abs(ave - mask)

    loss = (wbce * weight).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    return loss.mean()


def weighted_iou_loss(pred, mask):
    """
    Weighted IoU Loss
    """
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return wiou.mean()


def structure_loss(pred, mask):
    """
    [Updated] Weighted Structure Loss
    """
    if mask.dtype != pred.dtype:
        mask = mask.type_as(pred)

    wbce = weighted_bce_loss(pred, mask)
    wiou = weighted_iou_loss(pred, mask)

    return wbce + wiou