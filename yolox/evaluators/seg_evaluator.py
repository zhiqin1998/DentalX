#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import itertools
import time
from collections import ChainMap, defaultdict, OrderedDict
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

import numpy as np

import torch

from yolox.utils import (
    gather,
    is_main_process,
    synchronize,
    time_synchronized,
)


def per_class_res_table(res_dict, class_names, metrics=['IoU', 'Dice', 'Acc']):
    row_pair = []
    for idx, name in enumerate(class_names):
        row_pair.append([name] + [res_dict[m][idx] for m in metrics])

    table_headers = ['Class'] + metrics
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table

class SegEvaluator:
    def __init__(
        self,
        dataloader,
        img_size: int,
        confthre: float,
        class_names: list,
        testdev: bool = False,
        ignore_index = 255,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre: confidence threshold ranging from 0 to 1, which
                is defined in the config file.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.testdev = testdev
        self.ignore_index = ignore_index


    def evaluate(
        self, model, distributed=False, half=False, trt_file=None, test_size=None, return_outputs=False,
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        results = []
        output_data = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        for cur_iter, (imgs, targets, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)
                targets = targets.cuda(non_blocking=True).long()

                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs, seg_inference=True)[-1]

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                for output, target, _id in zip(outputs, targets, ids):
                    pred = output.argmax(dim=0) # H,W
                    results.append(self.intersect_and_union(pred, target, self.num_classes, self.ignore_index))
                    output_data[_id] = (pred.cpu().numpy(), output.cpu().numpy())


        statistics = torch.cuda.FloatTensor([inference_time, n_samples])
        if distributed:
            # different process/device might have different speed,
            # to make sure the process will not be stucked, sync func is used here.
            synchronize()
            results = gather(results, dst=0)
            output_data = gather(output_data, dst=0)
            results = list(itertools.chain(*results))
            output_data = dict(ChainMap(*output_data))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(results, statistics)
        synchronize()

        if return_outputs:
            return eval_results, output_data
        return eval_results

    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                            num_classes: int, ignore_index: int):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    @staticmethod
    def total_area_to_metrics(total_area_intersect: np.ndarray,
                              total_area_union: np.ndarray,
                              total_area_pred_label: np.ndarray,
                              total_area_label: np.ndarray,
                              metrics = ['mIoU'],
                              nan_to_num = None,
                              beta: int = 1):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
                'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | torch.Tensor): The precision value.
                recall (float | torch.Tensor): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [torch.tensor]: The f-score value.
            """
            score = (1 + beta ** 2) * (precision * recall) / (
                    (beta ** 2 * precision) + recall)
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({'aAcc': all_acc})
        for metric in metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics['IoU'] = iou
                ret_metrics['Acc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / (
                        total_area_pred_label + total_area_label)
                acc = total_area_intersect / total_area_label
                ret_metrics['Dice'] = dice
                ret_metrics['Acc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor([
                    f_score(x[0], x[1], beta) for x in zip(precision, recall)
                ])
                ret_metrics['Fscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall

        ret_metrics = {
            metric: value.numpy()
            for metric, value in ret_metrics.items()
        }
        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })
        return ret_metrics

    def evaluate_prediction(self, results, statistics):
        if not is_main_process():
            return 0, 0, 0, None

        logger.info("Evaluate in main process...")

        metrics = ["mIoU", "mDice"]

        inference_time = statistics[0].item()
        n_samples = statistics[1].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["inference"],
                    [a_infer_time],
                )
            ]
        )

        info = time_info + "\n"

        if len(results) > 0:
            results = tuple(zip(*results))
            assert len(results) == 4
            total_area_intersect = sum(results[0])
            total_area_union = sum(results[1])
            total_area_pred_label = sum(results[2])
            total_area_label = sum(results[3])
            ret_metrics = self.total_area_to_metrics(
                total_area_intersect, total_area_union, total_area_pred_label, total_area_label, metrics)

            class_names = self.class_names

            # summary table
            ret_metrics_summary = OrderedDict({
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            })
            metrics = dict()
            for key, val in ret_metrics_summary.items():
                if key == 'aAcc':
                    metrics[key] = val
                else:
                    metrics['m' + key] = val

            # each class table
            ret_metrics.pop('aAcc', None)
            ret_metrics_class = OrderedDict({
                ret_metric: np.round(ret_metric_value * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            })
            pc_metrics = list(ret_metrics_class.keys())
            ret_metrics_class.update({'Class': class_names})
            ret_metrics_class.move_to_end('Class', last=False)

            info += f"mIoU: {metrics['mIoU']}, mDice: {metrics['mDice']}, aAcc: {metrics['aAcc']}\n"
            per_class_table = per_class_res_table(ret_metrics_class, class_names, pc_metrics)
            info += "per class metrics:\n" + per_class_table + "\n"
            return metrics['mIoU'], metrics['mDice'], metrics['aAcc'], info
        else:
            return 0, 0, 0, info
