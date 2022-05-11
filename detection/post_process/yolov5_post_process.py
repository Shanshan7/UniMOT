#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from common.data_struct import DetResult
from detection.yolov5.utils.general import non_max_suppression, scale_coords


class YoloV5PostProcess():

    def __init__(self):
        super().__init__()
        self.conf_thres=0.25
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.classes = None # filter by class: --class 0, or --class 0 2 3
        self.agnostic = False # class-agnostic NMS

    def __call__(self, prediction, input_data, origin_data):
        result = []
        preds = non_max_suppression(prediction, self.conf_thres, self.iou_thres, self.classes, self.agnostic, max_det=self.max_det)[0]
        preds[:, :4] = scale_coords(input_data.shape[2:], preds[:, :4], origin_data.shape).round()
        for index, value in enumerate(preds):
            det_result = DetResult()
            det_result.class_id = value[5]
            det_result.confidence = value[4]
            det_result.head_location = []
            det_result.object_location = value[0:4]
            result.append(det_result)
        return result
