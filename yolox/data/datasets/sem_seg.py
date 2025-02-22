#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import copy
import os

import cv2
import numpy as np
from pycocotools.coco import COCO

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import CacheDataset, cache_read_img


class SemSegDataset(CacheDataset):
    """
    Semantic segmentation dataset class.
    """

    def __init__(
        self,
        img_dir,
        ann_dir,
        name="train2014",
        img_size=(640, 640),
        preproc=None,
        cache=False,
        cache_type="ram",
        pad_val=-1,
    ):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.pad_val = pad_val

        self.file_lists = [x for x in os.listdir(img_dir) if x.lower().endswith(".jpg") or x.lower().endswith(".png")]
        self.num_imgs = len(self.file_lists)

        self.name = name
        self.img_size = img_size
        self.preproc = preproc

        self.annotations = self._load_annotations()

        path_filename = [os.path.join(name, anno[3]) for anno in self.annotations]
        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            cache_dir_name=f"cache_{name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type
        )

    def __len__(self):
        return self.num_imgs

    def _load_annotations(self):
        return [self.load_anno_from_ids(id_) for id_ in range(self.num_imgs)]

    def load_anno_from_ids(self, id_):
        file_name = self.file_lists[id_]
        ann_name = file_name[:-4] + '_labelTrainIds.png'
        seg_mask = cv2.cvtColor(cv2.imread(os.path.join(self.ann_dir, ann_name)), cv2.COLOR_BGR2GRAY) + self.pad_val
        height, width = seg_mask.shape

        r = min(self.img_size[0] / height, self.img_size[1] / width)

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        seg_mask = cv2.resize(
            seg_mask,
            (int(seg_mask.shape[1] * r), int(seg_mask.shape[0] * r)),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)

        return (seg_mask, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]

        img_file = os.path.join(self.img_dir, file_name)

        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"

        return img

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        return self.load_resized_img(index)

    def pull_item(self, index):
        label, origin_image_size, _, _ = self.annotations[index]
        img = self.read_img(index)

        return img, copy.deepcopy(label), origin_image_size, np.array([index])

    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            targets (torch.Tensor): pre-processed label data.
                shape is same as img
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        # assert img.shape[0] == target.shape[0] and img.shape[1] == target.shape[1]
        return img, target, img_info, img_id
