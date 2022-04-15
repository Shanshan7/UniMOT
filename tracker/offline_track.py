# limit the number of cpus used by high performance libraries
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from detection.det_infer import DetectInfer
from common.data_struct import DetResult, DetResultInfo
from tracker.deep_sort_pytorch.utils.parser import get_config
from tracker.deep_sort_pytorch.deep_sort import DeepSort
from detection.yolov5.utils.general import xyxy2xywh

import numpy as np
import time


class OfflineTracker():
    def __init__(self, track_result_path):
        super().__init__()
        self.offline_track_results_dict = {}
        offline_track_file = open(track_result_path)
        offline_detect_results = offline_track_file.readlines()
        for offline_detect_result in offline_detect_results:
            detect_result_vec = offline_detect_result.replace("\n", "").split(" ")
            det_result = DetResult()
            det_result.current_frame = int(detect_result_vec[0])
            det_result.track_id = int(float(detect_result_vec[1]))
            det_result.class_id = int(float(detect_result_vec[9]))
            det_result.confidence = float(detect_result_vec[6])
            det_result.head_location = []
            det_result.pedestrian_location = [float(det_loc) for det_loc in detect_result_vec[2:6]]
            if det_result.current_frame in self.offline_track_results_dict:
                self.offline_track_results_dict[det_result.current_frame] = \
                    self.offline_track_results_dict[det_result.current_frame] + [det_result]
            else:
                self.offline_track_results_dict[det_result.current_frame] = \
                    [det_result]

    def process(self, input_data, frame_idx=-1):
        self.det_result_info = DetResultInfo(input_data)
        self.det_result_info.det_results_vector = self.offline_track_results_dict[frame_idx]

    def show_result(self):
        image = self.det_result_info.image
        det_results = self.det_result_info.det_results_vector
        for det_result in det_results:
            p1, p2 = (int(det_result.pedestrian_location[0]), int(det_result.pedestrian_location[1])), \
                     (int(det_result.pedestrian_location[2]), int(det_result.pedestrian_location[3]))
            cv2.rectangle(image, p1, p2, (0, 128, 128), thickness=2, lineType=cv2.LINE_AA)
            label = f'{det_result.track_id} {det_result.class_id} {det_result.confidence:.2f}'
            tf = max(2 - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=1 / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, (0, 128, 128), -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, 1 / 3, (255, 255, 255),
                        thickness=tf, lineType=cv2.LINE_AA)
        cv2.namedWindow("image", 0)
        cv2.imshow("image", image)
        cv2.waitKey()


if __name__ == '__main__':
    import cv2
    offline_tracker = OfflineTracker("./detect_result/2021_12_30_16_11_58.txt")
    for i in range(1, 201):
        input_data = cv2.imread("./detect_result/" + str(i) + ".jpg")
        offline_tracker.process(input_data, i)
        offline_tracker.show_result()
        # jde_tracker.save_result()
