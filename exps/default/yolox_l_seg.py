#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.num_classes = 20
        self.data_dir = '<data directory>/coco'

        self.max_epoch = 200
        self.eval_interval = 1
        # save history checkpoint or not.
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = False

        self.basic_lr_per_img = 0.001 / 32
        self.no_aug_epochs = 50

        self.semantic_aux_branch = True
        self.semantic_train_image_dir = '<data directory>/coco_stuff10k/images/train2014'
        self.semantic_test_image_dir = '<data directory>/coco_stuff10k/images/test2014'
        self.semantic_train_ann_dir = '<data directory>/coco_stuff10k/annotations/train2014'
        self.semantic_test_ann_dir = '<data directory>/coco_stuff10k/annotations/test2014'
        self.semantic_num_classes = 7 # should include bg class
        self.semantic_class_names = ['Enamel', 'Dentin', 'Root Dentin', 'Pulp', 'Bone', 'Restoration', 'Background']
        self.semantic_bg_cls_id = 6 # usually is semantic_num_classes - 1
        self.semantic_class_weights = [2.39, 3.3, 1.24,
                                       7.3, 0.35, 6.01, 0.2,]
        self.semantic_connect_aux = True
