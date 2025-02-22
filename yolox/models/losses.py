#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import cxcywh2xyxy


def cross_entropy_soft_targets(input, target, reduction='none'):
    log_input = torch.nn.functional.log_softmax(input, dim=-1)
    loss = -torch.sum(target * log_input, dim=-1)

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise NotImplementedError('Unsupported reduction mode.')


class NormalizedCrossEntropy(nn.Module):
    def __init__(self, num_classes, class_weight=None, scale=1.):
        super(NormalizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        if class_weight is not None:
            assert len(class_weight) == num_classes
            class_weight = torch.tensor(class_weight).view((1, num_classes))
        self.weights = class_weight
        self.scale = scale

    def forward(self, pred, labels):
        pred = pred.permute(0, 2, 3, 1).reshape((-1, self.num_classes))
        labels = labels.view(-1)
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(labels.device)
        if self.weights is not None:
            nce = -1 * torch.sum(label_one_hot * pred * self.weights.to(labels.device), dim=1) / (- pred.sum(dim=1))
        else:
            nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return nce.mean() * self.scale

    @property
    def loss_name(self):
        return 'nce'

class ReverseCrossEntropy(nn.Module):
    def __init__(self, num_classes, class_weight=None, scale=1.):
        super(ReverseCrossEntropy, self).__init__()
        self.num_classes = num_classes
        if class_weight is not None:
            assert len(class_weight) == num_classes
            class_weight = torch.tensor(class_weight).view((1, num_classes))
        self.weights = class_weight
        self.scale = scale

    def forward(self, pred, labels):
        pred = pred.permute(0, 2, 3, 1).reshape((-1, self.num_classes))
        labels = labels.view(-1)
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(labels.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        if self.weights is not None:
            rce = (-1 * torch.sum(pred * torch.log(label_one_hot) * self.weights.to(labels.device), dim=1))
        else:
            rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        return rce.mean() * self.scale

    @property
    def loss_name(self):
        return 'rce'

class JaccardLoss(nn.Module):
    def __init__(self, num_classes, scale=1., log_loss=True, eps=1e-7, class_weight=None):
        super(JaccardLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.log_loss = log_loss
        self.scale = scale
        if class_weight is not None:
            assert len(class_weight) == num_classes
            class_weight = torch.tensor(class_weight)
        self.weights = class_weight

    def forward(self, pred, labels):
        # pred (NCHW), labels (NHW)
        pred = F.log_softmax(pred, dim=1).exp()
        bs = labels.size(0)
        dims = (0, 2)

        y_true = labels.view(bs, -1)
        y_pred = pred.view(bs, self.num_classes, -1)

        y_true = F.one_hot(y_true, self.num_classes).to(device=y_pred.device, dtype=y_pred.dtype)  # N,H*W -> N,H*W, C
        y_true = y_true.permute(0, 2, 1)  # H, C, H*W

        # scores = soft_jaccard_score(y_pred, y_true.type(y_pred.dtype), eps=self.eps, dims=dims)
        intersection = torch.sum(y_pred * y_true, dim=dims)
        cardinality = torch.sum(y_pred + y_true, dim=dims)
        union = cardinality - intersection
        scores = intersection / union.clamp_min(self.eps)
        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # IoU loss is defined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss
        mask = y_true.sum(dims) > 0
        loss *= mask.float()
        # if self.weights is not None:
        #     loss = loss * self.weights.to(labels.device)

        return loss.mean() * self.scale

    @property
    def loss_name(self):
        return 'jaccard'

class OhemCrossEntropy(nn.Module):
    """OhemCrossEntropy loss.

    This func is modified from
    `PIDNet <https://github.com/XuJiacong/PIDNet/blob/main/utils/criterion.py#L43>`_.  # noqa

    Licensed under the MIT License.

    Args:
        ignore_label (int): Labels to ignore when computing the loss.
            Default: 255
        thresh (float, optional): The threshold for hard example selection.
            Below which, are prediction with low confidence. If not
            specified, the hard examples will be pixels of top ``min_kept``
            loss. Default: 0.7.
        min_kept (int, optional): The minimum number of predictions to keep.
            Default: 100000.
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_name (str): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_boundary'.
    """

    def __init__(self,
                 ignore_label: int = 255,
                 thres: float = 0.9,
                 min_kept: int = 100000,
                 loss_weight: float = 1.0,
                 class_weight=None,
                 loss_name: str = 'ohem'):
        super().__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.loss_weight = loss_weight
        self.loss_name_ = loss_name
        self.class_weight = class_weight

    def forward(self, score, target):
        """Forward function.
        Args:
            score (Tensor): Predictions of the segmentation head.
            target (Tensor): Ground truth of the image.

        Returns:
            Tensor: Loss tensor.
        """
        # score: (N, C, H, W)
        pred = F.softmax(score, dim=1)
        if self.class_weight is not None:
            class_weight = score.new_tensor(self.class_weight)
        else:
            class_weight = None
        pixel_losses = F.cross_entropy(
            score,
            target,
            weight=class_weight,
            ignore_index=self.ignore_label,
            reduction='none').contiguous().view(-1)  # (N*H*W)
        mask = target.contiguous().view(-1) != self.ignore_label  # (N*H*W)

        tmp_target = target.clone()  # (N, H, W)
        tmp_target[tmp_target == self.ignore_label] = 0
        # pred: (N, C, H, W) -> (N*H*W, C)
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        # pred: (N*H*W, C) -> (N*H*W), ind: (N*H*W)
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        if pred.numel() > 0:
            min_value = pred[min(self.min_kept, pred.numel() - 1)]
        else:
            return score.new_tensor(0.0)
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return self.loss_weight * pixel_losses.mean()

    @property
    def loss_name(self):
        return self.loss_name_


# if __name__ == '__main__':
    # import time
    # bu_loss = BoxUncertainLoss(reduction='mean', loss_type='dmm')
    # pred_mean = torch.tensor([[10., 20, 10, 20], [400, 200, 50, 80]], requires_grad=True)
    # pred_var = torch.tensor([[1., 0.5, -1, 0.4], [-3, 1.5, 2, -1]], requires_grad=True)
    # target = torch.tensor([[4, 3, 10, 10, 20, 0, 0, 0, 0, 0, 0, 0, 0],
    #                        [12, 380, 175, 420, 225, 379, 180, 433, 212, 382, 173, 411, 230]])
    # st = time.time()
    # print(bu_loss(pred_mean.repeat(1000,1), pred_var.repeat(1000,1), target.repeat(1000,1)))
    # print(time.time() - st)

class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
