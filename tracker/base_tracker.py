import numpy as np
import os
import cv2


class BaseTrack(object):
    def __init__(self, track_config_path, det_method, feature=None, **detect_params):
        super().__init__()
        self.det_method = det_method
        self.reid_method = True
        self.det_result_info = None
        self.save_path = ".easy_log/track_result"

    def process(self, input_data, frame_idx=-1):
        pass

    def save_result(self):
        image = self.det_result_info.image
        det_results = self.det_result_info.det_results_vector
        if len(det_results) > 0:
            cv2.imwrite(os.path.join(".", self.save_path.split("/")[1], str(det_results[0].current_frame)+".jpg"), image)

            file_save = open(os.path.join(self.save_path), mode="a")

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
