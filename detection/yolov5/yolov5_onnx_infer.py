import os
from optparse import OptionParser
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt
import cv2

from detection.yolov5.utils.augmentations import letterbox
from detection.yolov5.utils.general import non_max_suppression, scale_coords
from detection.post_process.yolov5_post_process import YoloV5PostProcess
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

    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        self.session = onnxruntime.InferenceSession(self.onnx_path, None)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name

        self.yolov5_post_process = YoloV5PostProcess()

    def infer(self, data_path):
        input_data = cv2.imread(data_path)
        image = input_data.copy()
        data = self.preprocess(input_data)
        raw_result = torch.tensor(self.session.run([self.output_name], {self.input_name: data}))
        result = self.yolov5_post_process(raw_result, data, image)
        self.show_result(result, image)

    def preprocess(self, input_data):
        # Padded resize
        input_data = letterbox(input_data, self.input_shape[2:], stride=32, auto=False)[0]
        input_data = input_data.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        input_data = np.ascontiguousarray(input_data)
        result = np.array(input_data).astype('float32')
        result = result / 255.0  # 0 - 255 to 0.0 - 1.0
        result = np.expand_dims(result, axis=0)
        return result

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
