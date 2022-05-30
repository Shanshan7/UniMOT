#python3 detection/fairmot/fairmot_onnx_infer.py -i /home/lpj/wc/image/0.jpg -o /home/lpj/wc/FairMOT-master/exp/mot/mot17_yolo_17/model_last.onnx 
#python3 detection/fairmot/torch_model.py --arch yolo --load_model '/home/lpj/wc/FairMOT-master/exp/mot/mot17_yolo_17/model_last.pth'
python3 detection/fairmot/onnxrun.py -i /home/lpj/wc/image/0.jpg -o /home/lpj/wc/FairMOT-master/exp/mot/mot17_yolo_17/model_last.onnx --K 6
