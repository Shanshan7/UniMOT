#!/usr/bin/env python
# -*- coding:utf-8 -*-
import cv2
import os
import time
from abc import ABC
from common.data_struct import DetResultInfo, DetResult


class BaseDetector(ABC):

    def __init__(self, det_method, conf_thresh, iou_thresh, class_filter):
        super().__init__()

        self.det_method = det_method
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.class_filter = class_filter

        system_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        self.save_path = os.path.join("./detect_result", system_time+".txt")
        if not os.path.exists(self.save_path.split("/")[1]):
            os.mkdir(self.save_path.split("/")[1])

        self.det_result_info = None

    # def set_train_config(self, config_path=None):
    #     pass

    def set_model_param(self, **params):
        pass

    # @abc.abstractmethod
    # def postprocess(self):
    #     pass

    # @abc.abstractmethod
    def infer(self, origin_data):
        pass

    def save_result(self):
        image = self.det_result_info.image
        det_results = self.det_result_info.det_results_vector
        cv2.imwrite(os.path.join(".", self.save_path.split("/")[1], str(det_results[0].current_frame)+".jpg"), image)

        file_save = open(self.save_path, mode="a")

        for det_result in det_results:
            file_save.write("{} {} {} {} {} {} {} -1 -1 {}\n".format(det_result.current_frame,
                                                                det_result.track_id,
                                                                det_result.pedestrian_location[0],
                                                                det_result.pedestrian_location[1],
                                                                det_result.pedestrian_location[2],
                                                                det_result.pedestrian_location[3],
                                                                det_result.confidence,
                                                                det_result.class_id))
        file_save.close()

    def show_result(self):
        image = self.det_result_info.image
        det_results = self.det_result_info.det_results_vector
        for det_result in det_results:
            p1, p2 = (int(det_result.pedestrian_location[0]), int(det_result.pedestrian_location[1])), \
                     (int(det_result.pedestrian_location[2]), int(det_result.pedestrian_location[3]))
            cv2.rectangle(image, p1, p2, (0, 128, 128), thickness=2, lineType=cv2.LINE_AA)
        cv2.namedWindow("image", 0)
        cv2.imshow("image", image)
        if cv2.getWindowProperty('image', 1) < 0:   # 断窗口是否关闭
            return False
        if cv2.waitKey(0) & 0xff == ord('q'):  # 按q退出
            cv2.destroyAllWindows()
            return False
        else:
            return True


    # @property
    def device(self, GPU=True):
        return 'cuda:0' if GPU else 'cpu'
