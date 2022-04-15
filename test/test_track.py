import argparse
import cv2

from tracker.base_sde_tracker import SDETracker
# from tracker.base_jde_tracker import JDETracker


parser = argparse.ArgumentParser()
parser.add_argument('--detect_method', type=str, default='yolov5Det',
                    help='detect method')

args = parser.parse_args()

sde_tracker = SDETracker("tracker/deep_sort_pytorch/configs/deep_sort.yaml", "yolov5Det",
                         model_path="./detection/yolov5/weights/best.pt")
# jde_tracker = JDETracker("tracker/deep_sort_pytorch/configs/deep_sort.yaml", "yolov5Det",
#                          model_path="./detection/yolov5/weights/best.pt")


for i in range(94, 1074):
    input_data = cv2.imread("/home/edge/data/VOCdevkit/Velocity_Calculation/20210121/rgb/20220121_180042_rgb/" + str(i) + ".jpg")
    sde_tracker.process(input_data, i)
    # sde_tracker.show_result()
    sde_tracker.save_result()
#     if cv2.waitKey(20) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()
