from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
import time
import os
import cv2
import numpy as np
import functional as f
import onnxruntime

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
        transforms = [
            f.Resize(target_size=(target_width, target_height)), 
            f.Normalize(mean=(0,0,0), std=(1,1,1))
        ]

        img_path = self.files[self.count]
        img0 = cv2.imread(str(img_path))
        normalized_img, _ = f.Compose(transforms)(img0)
        # add an new axis in front
        img_input = normalized_img[np.newaxis, :]
        # scale_factor is calculated as: im_shape / original_im_shape
        h_scale = target_height / img0.shape[0]
        w_scale = target_width / img0.shape[1]
        img = {"im_shape": [[target_height, target_width]], "image": img_input, "scale_factor": [[h_scale, w_scale]]}

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return img_path, img, img0

    def __getitem__(self, idx):
        idx = idx % self.nF
        img_path = self.files[idx]

        transforms = [
            f.Resize(target_size=(target_width, target_height)), 
            f.Normalize(mean=(0,0,0), std=(1,1,1))
        ]

        img_path = self.files[self.count]
        img0 = cv2.imread(str(img_path))
        normalized_img, _ = f.Compose(transforms)(img0)
        # add an new axis in front
        img_input = normalized_img[np.newaxis, :]
        # scale_factor is calculated as: im_shape / original_im_shape
        h_scale = target_height / img0.shape[0]
        w_scale = target_width / img0.shape[1]
        img = {"im_shape": [[target_height, target_width]], "image": img_input, "scale_factor": [[h_scale, w_scale]]}

        return img_path, img, img0

    def __len__(self):
        return self.nF  # number of files 
   
def prepare_input():
    transforms = [
        f.Resize(target_size=(target_width, target_height)), 
        f.Normalize(mean=(0,0,0), std=(1,1,1))
    ]

    img_file = "/home/wc/wc/dataset/images/street.jpeg"
    #img_path = self.files[self.count]
    img = cv2.imread(str(img_file))
    normalized_img, _ = f.Compose(transforms)(img)
    # add an new axis in front
    img_input = normalized_img[np.newaxis, :]
    # scale_factor is calculated as: im_shape / original_im_shape
    h_scale = target_height / img.shape[0]
    w_scale = target_width / img.shape[1]
    input = {"im_shape": [[target_height, target_width]], "image": img_input, "scale_factor": [[h_scale, w_scale]]}
    return input, img

#-----------------------------------------------------------------------------------------
target_height = 1088
target_width = 1088
Kt = 500
conf_thres = 0.3
gpu = True
num_classes = 1
max_per_image = 500
saveimg = True
savedir = '../output/'
os.makedirs(savedir, exist_ok=True)
#-----------------------------------------------------------------------------------------

if gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
#input, img = prepare_input()

session = onnxruntime.InferenceSession("/home/wc/wc/dataset/model/fairmot_1088x1088.onnx", providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
input_shape = session.get_inputs()[1].shape
img_size = (input_shape[3],input_shape[2])

dataloader = LoadImages('/home/wc/wc/dataset/images/street.jpeg', img_size)
for i, (path, img, img0) in enumerate(dataloader):
    person_count = 0
    with torch.no_grad():
        raw_result = session.run(["concat_8.tmp_0", "gather_5.tmp_0"], img)
        raw_result[0] = np.array(raw_result[0])
        raw_result[1] = np.array(raw_result[1])
        a = raw_result[0]
        a[:,[0,1,2,3,4,5]] = a[:,[2,3,4,5,1,0]]#切换列
        raw_result[0] = a
        raw_result[0] = torch.tensor(np.array([raw_result[0]]))
        raw_result[1] = torch.tensor(np.array([raw_result[1]]))
        dets = raw_result[0]#x0, y0, x1, y1, score, cls_id
        id_feature = raw_result[1]
        id_feature = id_feature.squeeze(0)
        id_feature = id_feature.cpu().numpy()
    dets = f.post_process(dets, num_classes)
    dets = f.merge_outputs([dets], num_classes, max_per_image)[1]
    remain_inds = dets[:, 4] > conf_thres
    dets = dets[remain_inds]
    id_feature = id_feature[remain_inds]

# vis
path = "/home/wc/wc/dataset/images/street.jpeg"
person_count+=len(dets)
for i in range(0, dets.shape[0]):
    bbox = dets[i][0:4]
    cv2.rectangle(img0, (int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)#
print('------------')
print(f'Result: {person_count}')
print('------------')
if saveimg:
    cv2.imwrite(os.path.join(savedir,path.split('/')[-1]),img0)
        
cv2.namedWindow('dets',0)
cv2.imshow('dets', img0)
k = cv2.waitKey(0)


cv2.destroyAllWindows()
while True:
    if k == ord('q'):
        break
