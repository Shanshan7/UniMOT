# UniMOT下fairmot视频前向推理
## 将模型,预训练模型和视频放在指定目录下：
```
..
   |——————UniMOT
   |——————dataset
            └——————model
                        └——————hrnet18.pth
                        └——————..
            └——————pretrained_model
                        └——————hrnetv2_w18_imagenet_pretrained.pth
                        └——————..
            └——————videos
                        └——————MOT16-03.mp4
                        └——————..
```
## 修改文件路径：
1. 修改arch路径（../UniMOT/detection/fairmot/src/lib/opts下'--arch'）
2. 修改预模型路径（../UniMOT/detection/fairmot/src/lib/models/networks/pose_hrnet.py下cfg.MODEL.PRETRAINED（547行））
3. 修改视频路径（../UniMOT/detection/fairmot/src/lib/opts下'--input-video'）
4. 修改输出路径（../UniMOT/detection/fairmot/src/lib/opts下'--output-root'）
## pth视频前向推理：
```
#前向推理
python demo.py mot --load_model /home/wc/wc/dataset/model/hrnet18.pth --conf_thres 0.4
#追踪，获取模型各项性能指标
python track.py mot --test_mot17 True --load_model /home/wc/wc/dataset/model/hrnet18.pth --conf_thres 0.4
```
## onnx（pth转）视频前向推理：
```
#前向推理
python demo_onnx.py mot --load_model /home/wc/wc/dataset/model/model_last.onnx --conf_thres 0.6
#追踪，获取模型各项性能指标
python track_onnx.py mot --test_mot17 True --load_model /home/wc/wc/dataset/model/model_last.onnx --conf_thres 0.6
```

## onnx（paddle转）视频前向推理：
```
#前向推理
python demo_paddle.py mot --load_model /home/wc/wc/dataset/model/fairmot_1088x1088.onnx --conf_thres 0.5
#追踪，获取模型各项性能指标
python track_onnx.py mot --test_mot17 True --load_model /home/wc/wc/dataset/model/model_last.onnx --conf_thres 0.6
```

