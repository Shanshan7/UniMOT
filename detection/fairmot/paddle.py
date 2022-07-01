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

def prepare_input():
    transforms = [
        f.Resize(target_size=(target_width, target_height)), 
        f.Normalize(mean=(0,0,0), std=(1,1,1))
    ]

    img_file = "/home/wc/wc/UniMOT/detection/fairmot/images/street.jpeg"
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
input, img = prepare_input()

session = onnxruntime.InferenceSession("/home/wc/wc/dataset/model/fairmot_1088x1088.onnx", None)
person_count = 0
with torch.no_grad():
    raw_result = session.run(["concat_8.tmp_0", "gather_5.tmp_0"], input)
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
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)#
print('------------')
print(f'Result: {person_count}')
print('------------')
if saveimg:
    cv2.imwrite(os.path.join(savedir,path.split('/')[-1]),img)
        
cv2.namedWindow('dets',0)
cv2.imshow('dets', img)
k = cv2.waitKey(0)


cv2.destroyAllWindows()
while True:
    if k == ord('q'):
        break
