#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead, YOLOXSegAuxHead, resize
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None, seg_aux_head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head
        self.seg_aux_head = seg_aux_head

    def forward(self, x, targets=None, aux_targets=None, aux_masks=None, seg_inference=False):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            seg_outs = None
            if self.seg_aux_head is not None:
                assert aux_targets is not None and aux_masks is not None
                fpn_outs, aux_fpn_outs = (fpn_outs[0][~aux_masks], fpn_outs[1][~aux_masks], fpn_outs[2][~aux_masks]), (fpn_outs[0][aux_masks],),
                seg_loss, seg_losses = self.seg_aux_head(aux_fpn_outs, aux_targets)
                if self.head.connect_aux:
                    seg_outs = self.seg_aux_head.forward_only((fpn_outs[0],))
                # targets, aux_targets = targets[~aux_masks], aux_targets[aux_masks]
                x = x[~aux_masks]
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x, seg_outs=seg_outs,
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
            if self.seg_aux_head is not None:
                outputs["seg_loss"] = seg_loss
                outputs["total_loss"] = outputs["total_loss"] + seg_loss
                loss_count = len(self.seg_aux_head.criterion)
                for i in range(len(aux_fpn_outs)):
                    for j in range(loss_count):
                        outputs["fpn{}_seg_{}_loss".format(i, self.seg_aux_head.criterion[j].loss_name)] = seg_losses[i*loss_count+j]
        else:
            seg_outs = None
            if self.seg_aux_head is not None and self.head.connect_aux:
                seg_outs = self.seg_aux_head.forward_only((fpn_outs[0],))
            outputs = self.head(fpn_outs, seg_outs=seg_outs)
            if self.seg_aux_head is not None and seg_inference:
                if seg_outs is None:
                    seg_outs = self.seg_aux_head(fpn_outs[:1])
                else:
                    seg_outs = resize(
                                seg_outs,
                                size=self.seg_aux_head.test_size,
                                mode='bilinear',
                                align_corners=self.seg_aux_head.align_corners) # return logit
                outputs = [outputs, seg_outs]

        return outputs

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        fpn_outs = self.backbone(x)
        self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)
