# limit the number of cpus used by high performance libraries
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from det_infer import DetectInfer
from common.data_struct import DetResult, DetResultInfo
from tracker.deep_sort_pytorch.utils.parser import get_config
from tracker.deep_sort_pytorch.deep_sort import DeepSort
from detection.yolov5.utils.general import xyxy2xywh

import numpy as np
import time


class JDETracker():
    def __init__(self, track_config_path, det_method, feature=None, **detect_params):
        super().__init__()
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
            self.offline_detect_results_dict = {}
            offline_detect_file = open(detect_params['detect_result_path'])
            offline_detect_results = offline_detect_file.readlines()
            for offline_detect_result in offline_detect_results:
                detect_result_vec = offline_detect_result.replace("\n", "").split(" ")
                det_result = DetResult()
                det_result.current_frame = int(detect_result_vec[0])
                det_result.track_id = int(detect_result_vec[1])
                det_result.class_id = int(float(detect_result_vec[9]))
                det_result.confidence = float(detect_result_vec[6])
                det_result.head_location = []
                det_result.pedestrian_location = [float(det_loc) for det_loc in detect_result_vec[2:6]]
                if det_result.current_frame in self.offline_detect_results_dict:
                    self.offline_detect_results_dict[det_result.current_frame] = \
                        self.offline_detect_results_dict[det_result.current_frame] + [det_result]
                else:
                    self.offline_detect_results_dict[det_result.current_frame] = \
                        [det_result]

    def process(self, input_data, frame_idx=-1):
        if self.det_method == "offlineDet":
            self.det_result_info = DetResultInfo(input_data)
            self.det_result_info.det_results_vector = self.offline_detect_results_dict[frame_idx]
        else:
            self.detect_infer.infer(input_data, frame_idx)
            self.det_result_info = self.detect_infer.det_result_info

        det_results = self.det_result_info.det_results_vector
        if len(det_results) != 0:
            xywhs = xyxy2xywh(np.array([det_result.pedestrian_location for det_result in det_results]))
            confs = np.array([det_result.confidence for det_result in det_results])
            clss = np.array([det_result.class_id for det_result in det_results])

            # pass detections to deepsort
            outputs = self.deepsort.update(xywhs, confs, clss, input_data)
            if len(outputs) > 0:
                for j, (output, det_result) in enumerate(zip(outputs, self.det_result_info.det_results_vector)):
                    det_result.class_id = int(output[5])
                    det_result.track_id = output[4]
                    det_result.confidence = output[6]
                    det_result.pedestrian_location = output[0:4]
            else:
                for j, det_result in enumerate(self.det_result_info.det_results_vector):
                    det_result.current_frame = frame_idx
        else:
            self.deepsort.increment_ages()

    def save_result(self):
        image = self.det_result_info.image
        det_results = self.det_result_info.det_results_vector
        cv2.imwrite(os.path.join(".", self.save_path.split("/")[1], str(det_results[0].current_frame)+".jpg"), image)

        file_save = open(os.path.join(self.save_path), mode="a")

        for det_result in det_results:
            file_save.write("{} {} {} {} {} {} {} -1 -1\n".format(det_result.current_frame,
                                                                det_result.track_id,
                                                                det_result.pedestrian_location[0],
                                                                det_result.pedestrian_location[1],
                                                                det_result.pedestrian_location[2],
                                                                det_result.pedestrian_location[3],
                                                                det_result.confidence))
        file_save.close()

    def show_result(self):
        image = self.detect_infer.det_result_info.image
        det_results = self.detect_infer.det_result_info.det_results_vector
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
    jde_tracker = JDETracker("tracker/deep_sort_pytorch/configs/deep_sort.yaml", "offlineDet",
                             detect_result_path="./detect_result/2021_12_30_16_11_58.txt")
    for i in range(1, 201):
        input_data = cv2.imread("./detect_result/" + str(i) + ".jpg")
        jde_tracker.process(input_data, i)
        # jde_tracker.show_result()
        jde_tracker.save_result()
