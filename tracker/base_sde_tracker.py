import numpy as np
import time
import os
import cv2

from detection.det_infer import DetectInfer
from tracker.base_tracker import BaseTrack
from common.data_struct import DetResult
from tracker.deep_sort_pytorch.deep_sort import DeepSort
from tracker.deep_sort_pytorch.utils.parser import get_config
from detection.yolov5.utils.general import xyxy2xywh


class SDETracker(BaseTrack):
    def __init__(self, track_config_path, det_method, feature=None, **detect_params):
        super().__init__(track_config_path, det_method, feature, **detect_params)
        self.det_method = det_method
        self.det_result_info = None
        system_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        self.save_path = os.path.join("./track_result", system_time+".txt")
        if not os.path.exists(self.save_path.split("/")[1]):
            os.mkdir(self.save_path.split("/")[1])
        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file(track_config_path)
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

        self.detect_infer = DetectInfer(det_method, conf_thresh=0.5, iou_thresh=0.45, class_filter=0)
        if self.det_method == "yolov5Det":
            self.detect_infer.set_model_param(model_path=detect_params['model_path'],
                                         augment=False, agnostic_nms=False)
        elif self.det_method == "nanoDet":
            self.detect_infer.set_model_param(config_path=detect_params['config_path'],
                                         model_path=detect_params['model_path'])
        elif self.det_method == "offlineDet":
            self.detect_infer.set_model_param(detect_result_path=detect_params['detect_result_path'])

    def process(self, input_data, frame_idx=-1):
        self.detect_infer.infer(input_data, frame_idx)
        self.det_result_info = self.detect_infer.det_result_info

        det_results = self.det_result_info.det_results_vector
        if len(det_results) != 0:
            xywhs = xyxy2xywh(np.array([det_result.pedestrian_location for det_result in det_results]))
            confs = np.array([det_result.confidence for det_result in det_results])
            clss = np.array([det_result.class_id for det_result in det_results])

            # pass detections to deepsort
            outputs = self.deepsort.update(xywhs, confs, clss, input_data)
            self.det_result_info.det_results_vector = []
            if len(outputs) > 0:
                for j, output in enumerate(outputs):
                    det_result_d = DetResult()
                    det_result_d.current_frame = frame_idx
                    det_result_d.class_id = int(output[5])
                    det_result_d.track_id = output[4]
                    det_result_d.confidence = output[6]
                    det_result_d.pedestrian_location = output[0:4]
                    self.det_result_info.det_results_vector.append(det_result_d)
        else:
            self.deepsort.increment_ages()