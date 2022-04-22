#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
import math
import numpy as np
from common.data_struct import DetResult
from detection.nanodet.nanodet.model.head.gfl_head import Integral
from detection.nanodet.nanodet.util import distance2bbox
from detection.nanodet.nanodet.model.module.nms import multiclass_nms
from detection.nanodet.nanodet.data.transform.warp import warp_boxes


# @REGISTERED_POST_PROCESS.register_module(PostProcessName.YoloPostProcess)
class NanoDetPostProcess():

    def __init__(self):
        super().__init__()
        self.num_classes = 1
        self.reg_max = 7
        self.conf_thresh = 0.6
        self.strides = [8, 16, 32, 64]
        self.warp_matrix = np.array([[0.21667, 0, 0], [0, 0.38519, 0], [0, 0, 1]])
        self.distribution_project = Integral(self.reg_max)

    def get_single_level_center_priors(
        self, batch_size, featmap_size, stride, dtype, device
    ):
        """Generate centers of a single stage feature map.
        Args:
            batch_size (int): Number of images in one batch.
            featmap_size (tuple[int]): height and width of the feature map
            stride (int): down sample stride of the feature map
            dtype (obj:`torch.dtype`): data type of the tensors
            device (obj:`torch.device`): device of the tensors
        Return:
            priors (Tensor): center priors of a single level feature map.
        """
        h, w = featmap_size
        x_range = (torch.arange(w, dtype=dtype, device=device)) * stride
        y_range = (torch.arange(h, dtype=dtype, device=device)) * stride
        y, x = torch.meshgrid(y_range, x_range)
        y = y.flatten()
        x = x.flatten()
        strides = x.new_full((x.shape[0],), stride)
        proiors = torch.stack([x, y, strides, strides], dim=-1)
        return proiors.unsqueeze(0).repeat(batch_size, 1, 1)

    def get_bboxes(self, cls_preds, reg_preds, img_metas):
        """Decode the outputs to bboxes.
        Args:
            cls_preds (Tensor): Shape (num_imgs, num_points, num_classes).
            reg_preds (Tensor): Shape (num_imgs, num_points, 4 * (regmax + 1)).
            img_metas (dict): Dict of image info.

        Returns:
            results_list (list[tuple]): List of detection bboxes and labels.
        """
        device = cls_preds.device
        b = cls_preds.shape[0]
        input_height, input_width = img_metas.shape[2:]
        input_shape = (input_height, input_width)

        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width) / stride)
            for stride in self.strides
        ]
        # get grid cells of one image
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                b,
                featmap_sizes[i],
                stride,
                dtype=torch.float32,
                device=device,
            )
            for i, stride in enumerate(self.strides)
        ]
        center_priors = torch.cat(mlvl_center_priors, dim=1)
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
        scores = cls_preds.sigmoid()
        result_list = []
        for i in range(b):
            # add a dummy background class at the end of all labels
            # same with mmdetection2.0
            score, bbox = scores[i], bboxes[i]
            padding = score.new_zeros(score.shape[0], 1)
            score = torch.cat([score, padding], dim=1)
            results = multiclass_nms(
                bbox,
                score,
                score_thr=0.05,
                nms_cfg=dict(type="nms", iou_threshold=0.6),
                max_num=100,
            )
            result_list.append(results)
        return result_list

    def __call__(self, prediction, input_data, origin_data):
        result = []
        cls_scores, bbox_preds = prediction[0].split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
        )
        preds = self.get_bboxes(cls_scores, bbox_preds, input_data)
        img_height, img_width = origin_data.shape[:2]
        det_bboxes, det_labels = preds[0]
        det_bboxes = det_bboxes.numpy()
        det_bboxes[:, :4] = warp_boxes(
            det_bboxes[:, :4], np.linalg.inv(self.warp_matrix), img_width, img_height
        )
        print(det_labels.shape, det_bboxes.shape)

        for det_label, det_bbox in zip(det_labels, det_bboxes):
            if det_bbox[4] > self.conf_thresh:
                det_result = DetResult()
                det_result.class_id = det_label.numpy()
                det_result.confidence = det_bbox[4]
                det_result.head_location = []
                det_result.object_location = det_bbox[0:4]
                result.append(det_result)
        return result
