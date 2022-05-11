import os
from optparse import OptionParser
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt
import cv2

from detection.post_process.nanodet_post_process import NanoDetPostProcess
from detection.nanodet.nanodet.util import cfg, load_config
from detection.nanodet.nanodet.data.transform import Pipeline
from detection.nanodet.nanodet.data.batch_process import stack_batch_img
from detection.nanodet.nanodet.data.collate import naive_collate
import torch


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program is onnx inference"

    parser.add_option("-i", "--input", dest="input_path",
                      metavar="PATH", type="string", default=None,
                      help="image path")

    parser.add_option("-o", "--onnx", dest="onnx_path",
                      metavar="PATH", type="string", default=None,
                      help="onnx path")

    (options, args) = parser.parse_args()

    if options.input_path:
        if not os.path.exists(options.input_path):
            parser.error("Could not find the input file")
        else:
            options.input_path = os.path.normpath(options.input_path)
    else:
        parser.error("'input' option is required to run this program")

    if options.onnx_path:
        if not os.path.exists(options.onnx_path):
            parser.error("Could not find the onnx file")
        else:
            options.onnx_path = os.path.normpath(options.onnx_path)
    else:
        parser.error("'onnx' option is required to run this program")

    return options


class OnnxInference():

    def __init__(self, onnx_path, config_path = "./detection/nanodet/config/nanodet-plus-m_416.yml"):
        self.onnx_path = onnx_path
        self.session = onnxruntime.InferenceSession(self.onnx_path, None)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name

        load_config(cfg, config_path)
        self.cfg = cfg
        self.pipeline = Pipeline(self.cfg.data.val.pipeline, self.cfg.data.val.keep_ratio)
        self.nanodet_post_process = NanoDetPostProcess()

    def infer(self, data_path):
        input_data = cv2.imread(data_path)
        image = input_data.copy()
        data = self.preprocess(input_data)
        raw_result = torch.tensor(self.session.run([self.output_name], {self.input_name: data}))
        print(raw_result.shape)
        result = self.nanodet_post_process(raw_result, data, image)
        self.show_result(result, image)

    def preprocess(self, input_data):
        img_info = {"id": 0}
        height, width = input_data.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=input_data, img=input_data)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1))
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        return meta["img"].numpy()

    def show_result(self, result, image):
        for det_result in result:
            p1, p2 = (int(det_result.object_location[0]), int(det_result.object_location[1])), \
                     (int(det_result.object_location[2]), int(det_result.object_location[3]))
            cv2.rectangle(image, p1, p2, (0, 128, 128), thickness=2, lineType=cv2.LINE_AA)
        cv2.namedWindow("image", 0)
        cv2.imshow("image", image)
        if cv2.getWindowProperty('image', 1) < 0:  # 断窗口是否关闭
            return False
        if cv2.waitKey(0) & 0xff == ord('q'):  # 按q退出
            cv2.destroyAllWindows()
            return False
        else:
            return True


def main():
    print("process start...")
    options = parse_arguments()
    inference = OnnxInference(options.onnx_path)
    inference.infer(options.input_path)
    print("process end!")


if __name__ == "__main__":
    main()
