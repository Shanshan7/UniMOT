from detection.base_detector import BaseDetector
from common.data_struct import DetResultInfo, DetResult

import sys
sys.path.append('./detection/yolov5')
sys.path.append('./detection/nanodet')

# nanodet
from detection.nanodet.nanodet.model.arch import build_model
from detection.nanodet.nanodet.util import Logger, cfg, load_config, load_model_weight
from detection.nanodet.nanodet.util.path import mkdir
from detection.nanodet.nanodet.data.transform import Pipeline
from detection.nanodet.nanodet.data.batch_process import stack_batch_img
from detection.nanodet.nanodet.data.collate import naive_collate
from detection.nanodet.nanodet.model.backbone.repvgg import repvgg_det_model_convert

# yolov5
from detection.yolov5.models.experimental import attempt_load
from detection.yolov5.utils.augmentations import letterbox
from detection.yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh

import torch
import os
import cv2
import numpy as np
from pathlib import Path

class DetectInfer(BaseDetector):
    def __init__(self, det_method, conf_thresh, iou_thresh, class_filter):
        super().__init__(det_method, conf_thresh, iou_thresh, class_filter)

    def set_model_param(self, **params):
        if self.det_method == "yolov5Det":
            self.model = attempt_load(params['model_path'], map_location=self.device())  # load FP32 model
            self.stride = int(self.model.stride.max())  # model stride
            # imgsz = check_img_size(params['imgsz'], s=stride)  # check img_size
            # names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            self.augment, self.agnostic_nms = params['augment'], params['agnostic_nms']
        elif self.det_method == "nanoDet":
            load_config(cfg, params['config_path'])
            self.cfg = cfg
            model = build_model(self.cfg.model)
            ckpt = torch.load(params['model_path'], map_location=lambda storage, loc: storage)
            logger = Logger(0, use_tensorboard=False)
            load_model_weight(model, ckpt, logger)
            if self.cfg.model.arch.backbone.name == "RepVGG":
                deploy_config = self.cfg.model
                deploy_config.arch.backbone.update({"deploy": True})
                deploy_model = build_model(deploy_config)

                model = repvgg_det_model_convert(model, deploy_model)
            self.model = model.to(self.device()).eval()
            self.pipeline = Pipeline(self.cfg.data.val.pipeline, self.cfg.data.val.keep_ratio)
        elif self.det_method == "offlineDet":
            self.offline_detect_results_dict = {}
            offline_detect_file = open(params['detect_result_path'])
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

    def infer(self, origin_data, frame_idx=-1):
        self.det_result_info = DetResultInfo(origin_data)
        input_data = origin_data.copy()
        if self.det_method == "yolov5Det":
            input_data = letterbox(input_data, 640, stride=self.stride, auto=True)[0]

            # Convert
            input_data = input_data.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            input_data = np.ascontiguousarray(input_data)
            input_data = torch.from_numpy(input_data).to(self.device())
            input_data = input_data.float()  # uint8 to fp16/32
            input_data /= 255.0  # 0 - 255 to 0.0 - 1.0
            if input_data.ndimension() == 3:
                input_data = input_data.unsqueeze(0)
            pred = self.model(input_data, augment=self.augment)[0]

            # Apply NMS
            results = non_max_suppression(pred, self.conf_thresh, self.iou_thresh, classes=self.class_filter,
                                       agnostic=self.agnostic_nms)[0]
            results[:, :4] = scale_coords(input_data.shape[2:], results[:, :4], origin_data.shape).round()
            results_numpy = results.cpu().numpy()
            for res in results_numpy:
                det_result = DetResult()

                det_result.current_frame = frame_idx
                det_result.class_id = res[5]
                det_result.confidence = res[4]
                det_result.head_location = []
                det_result.pedestrian_location = res[0:4]
                self.det_result_info.det_results_vector.append(det_result)

        elif self.det_method == "nanoDet":
            img_info = {"id": 0}
            img_info["file_name"] = None
            height, width = input_data.shape[:2]
            img_info["height"] = height
            img_info["width"] = width
            meta = dict(img_info=img_info, raw_img=input_data, img=input_data)
            meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
            meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device())
            meta = naive_collate([meta])
            meta["img"] = stack_batch_img(meta["img"], divisible=32)
            with torch.no_grad():
                results = self.model.inference(meta)[0]

            for key, value in results.items():
                if self.class_filter != None and key != self.class_filter:
                    continue
                for val in value:
                    if val[4] > self.conf_thresh:
                        det_result = DetResult()

                        det_result.current_frame = frame_idx
                        det_result.class_id = key
                        det_result.confidence = val[4]
                        det_result.head_location = []
                        det_result.pedestrian_location = val[0:4]
                        self.det_result_info.image = input_data
                        self.det_result_info.det_results_vector.append(det_result)

        elif self.det_method == "offlineDet":
            self.det_result_info = DetResultInfo(input_data)
            self.det_result_info.det_results_vector = self.offline_detect_results_dict[frame_idx]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_method', type=str, default='yolov5Det',
                        help='detect method')
    parser.add_argument('--input_data', type=str, default=None,
                        help='input data')
    parser.add_argument('--show_vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save_vid', action='store_true', help='save video tracking results')

    args = parser.parse_args()

    detect_infer = DetectInfer(args.detect_method, conf_thresh=0.5, iou_thresh=0.45, class_filter=0)
    if args.detect_method == "yolov5Det":
        detect_infer.set_model_param(model_path="./detection/yolov5/weights/best.pt",
                                     augment=False, agnostic_nms=False)
    elif args.detect_method == "nanoDet":
        detect_infer.set_model_param(config_path="./detection/nanodet/config/nanodet-plus-m_416.yml",
                                     model_path="./detection/nanodet/workspace/nanodet-plus-m_416/model_best/nanodet_model_best.pth")
    else:
        print("Wrong detect method!!!")

    if Path(args.input_data).is_dir():
        image_list = os.listdir(args.input_data)
        image_num = len(image_list)
        _, image_suffix = os.path.splitext(image_list[0])

        for i in range(1, image_num):
            input_data = cv2.imread(os.path.join(args.input_data, str(i) + image_suffix))
            detect_infer.infer(input_data, i)
            # detect_infer.save_result()
            if not detect_infer.show_result():
                break

    # elif Path(args.input_data).suffix in ['.txt', '.text']:
    #     dataloader = TextDataLoader(input_path, image_size, data_channel,
    #                                 resize_type, normalize_type, mean, std,
    #                                 transform_func)
    else:
        video_capture = cv2.VideoCapture(args.input_data)
        success = True
        frame_id = 1
        if video_capture is not None:
            while True:
                if not success:
                    break
                success, frame = video_capture.read()
                detect_infer.infer(frame, frame_id)
                frame_id +=1
                # detect_infer.save_result()
                if not detect_infer.show_result():
                    break