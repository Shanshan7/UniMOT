from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from optparse import OptionParser
import torch
import time
import os
import cv2

import numpy as np
import torch.nn.functional as F
from models.decode import mot_decode
from models.utils import _tranpose_and_gather_feat
from image import transform_preds

import onnxruntime

def parse_arguments():
    parser = OptionParser()
    parser.description = "This program is onnx inference"

    parser.add_option("-i", "--input", dest="input_path",
                      metavar="PATH", type="string", default=None,
                      help="image path")

    parser.add_option("-o", "--onnx", dest="onnx_path",
                      metavar="PATH", type="string", default=None,
                      help="onnx path")

    parser.add_option("--K",
                      type=int, default=4,
                      help="max number of output objects.")

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

def ctdet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(
          dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret

def preprocess(input_data):
    # Padded resize
    input_data = letterbox(input_data)[0]
    input_data = input_data.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    input_data = np.ascontiguousarray(input_data)
    result = np.array(input_data).astype('float32')
    result = result / 255.0  # 0 - 255 to 0.0 - 1.0
    result = np.expand_dims(result, axis=0)
    return result
def letterbox(img, height=1088, width=1920,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular127.5
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh

def post_process(dets, meta):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(dets.copy(), [meta['c']], [meta['s']],meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
    #print('dets',dets[0].keys())
    return dets[0]

def merge_outputs(detections):
    results = {}
    for j in range(1, num_classes + 1):
        results[j] = np.concatenate([detection[j] for detection in detections], axis=0).astype(np.float32)

    scores = np.hstack([results[j][:, 4] for j in range(1, num_classes + 1)])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results

class LoadImages:  # for inference
    def __init__(self, path, img_size=(1088, 608)):
        if os.path.isdir(path):
            image_format = ['.jpg', '.jpeg', '.png', '.tif']
            self.files = sorted(glob.glob('%s/*.*' % path))
            self.files = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, self.files))
        elif os.path.isfile(path):
            self.files = [path]

        self.nF = len(self.files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        assert self.nF > 0, 'No images found in ' + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration
        img_path = self.files[self.count]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return img_path, img, img0

    def __getitem__(self, idx):
        idx = idx % self.nF
        img_path = self.files[idx]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img0

    def __len__(self):
        return self.nF  # number of files    

options = parse_arguments()
max_per_image = 500
num_classes = 1
gpu = True
reid_dim = 128
arch = 'yolo'
ltrb = True
reg_offset = True
conf_thres = 0.3
Kt = options.K
heads = {'hm': num_classes, 'wh': 2 if not ltrb else 4, 'id': reid_dim, 'reg': 2}
head_conv = 256
down_ratio = 4
if gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

session = onnxruntime.InferenceSession(options.onnx_path, None)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
img_size = (input_shape[3],input_shape[2])
output_name = session.get_outputs()[0].name
print(options.onnx_path)

# Get dataloader
dataset = dataloader = LoadImages(options.input_path, img_size)
saveimg = True
savedir = '../output/'
os.makedirs(savedir, exist_ok=True)

for i, (path, img, img0) in enumerate(dataloader):
    person_count = 0
    im_blob = torch.from_numpy(img).cuda().unsqueeze(0)
    width = img0.shape[1]
    height = img0.shape[0]
    inp_height = im_blob.shape[2]
    inp_width = im_blob.shape[3]
    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
    meta = {'c': c, 's': s,'out_height': inp_height // down_ratio,'out_width': inp_width // down_ratio}

    ''' Step 1: Network forward, get detections & embeddings'''
    with torch.no_grad():
        im_blob = im_blob.cpu().numpy()
        raw_result = session.run(["hm", "wh", "id", "reg"], {input_name: im_blob})
        raw_result[0] = torch.tensor(raw_result[0])
        raw_result[1] = torch.tensor(raw_result[1])
        raw_result[2] = torch.tensor(raw_result[2])
        raw_result[3] = torch.tensor(raw_result[3])
        raw_result = {'hm':raw_result[0],'wh':raw_result[1],'id':raw_result[2],'reg':raw_result[3]}
        output = raw_result
        hm = output['hm'].sigmoid()
        wh = output['wh']
        id_feature = output['id']
        id_feature = F.normalize(id_feature, dim=1)

        reg = output['reg'] if reg_offset else None
        dets, inds = mot_decode(hm, wh, reg=reg, ltrb=ltrb, K=Kt)
        id_feature = _tranpose_and_gather_feat(id_feature, inds)
        id_feature = id_feature.squeeze(0)
        id_feature = id_feature.cpu().numpy()

    dets = post_process(dets, meta)
    dets = merge_outputs([dets])[1]
    remain_inds = dets[:, 4] > conf_thres
    dets = dets[remain_inds]
    id_feature = id_feature[remain_inds]

    # vis
    person_count+=len(dets)
    for i in range(0, dets.shape[0]):
        bbox = dets[i][0:4]
        cv2.rectangle(img0, (int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(0, 255, 0), 5)
    print(f'Img: {path} ++ Result: {person_count}')
    print('------------')
    if saveimg:
        cv2.imwrite(os.path.join(savedir,path.split('/')[-1]),img0)
        
    cv2.namedWindow('dets',0)
    cv2.imshow('dets', img0)
    k = cv2.waitKey(0)
    if k == ord('q'):
        break

cv2.destroyAllWindows()
