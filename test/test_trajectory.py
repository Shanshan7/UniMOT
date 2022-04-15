from tracker.base_sde_tracker import SDETracker
from tracker.offline_track import OfflineTracker
from trajectory.calculate_trajectory import CalculateTraj
from common.data_struct import DetResult, DetResultInfo
from visualization.trajectory_show import TrajectoryShow
import time

import cv2

def main(track_method, det_method, track_config_path,  calibration_file, feature=None, **params):
    calculate_traj = CalculateTraj(calibration_file)
    trajectory_show = TrajectoryShow()
    if track_method == "deepSort":
        if det_method == "yolov5Det":
            tracker = SDETracker(track_config_path, det_method,
                                     model_path=params['model_path'])
        elif det_method == "nanoDet":
            tracker = SDETracker(track_config_path, det_method,
                                     config_path=params['config_path'],
                                     model_path=params['model_path'])
        elif det_method == "offlineDet":
            tracker = SDETracker(track_config_path, det_method,
                                     config_path=params['detect_result_path'])
    elif track_method == "offlineTrack":
        tracker = OfflineTracker(params['track_result_path'])

    for i in range(1, 4000):
        input_data = cv2.imread("/home/edge/data_v2/MOT_PROJECT/白天路口/20211229/192.168.0.10_01_202112291344190918_0_part005/" + str(i) + ".jpg")
        # tof_data = cv2.imread("/home/edge/data/VOCdevkit/Velocity_Calculation/20210121/tof/20220121_175553_tof/" + str(i) + ".jpg")
        # tof_data = tof_data[552:607, 991:1136]

        start_time_track = time.time()
        tracker.process(input_data, i)
        print("[object_track] Time cost: {}".format(time.time() - start_time_track))

        # calculate_trajectory
        start_time_trajectory = time.time()
        calculate_traj.calculate_trajectory(tracker.det_result_info)
        print("[calculate_trajectory] Time cost: {}".format(time.time() - start_time_trajectory))
        trajectory_show.trajectory_show(tracker.det_result_info.image, calculate_traj.track_idx_map, calculate_traj.bird)

        # cv2.namedWindow("tof_show", 0)
        # cv2.resizeWindow(window_name, (int(image.shape[1]*0.7), int(image[0]*0.7)))
        # cv2.imshow("tof_show", tof_data)
        # key = cv2.waitKey()

if __name__ == '__main__':
    # main("offlineTrack", None, "tracker/deep_sort_pytorch/configs/deep_sort.yaml",
    #      "./calibration_211221.xml", feature=None, track_result_path="./track_result/2021_12_30_17_38_04.txt")
    main("deepSort", "yolov5Det", "tracker/deep_sort_pytorch/configs/deep_sort.yaml",
         "./calibration_211221.json", feature=None, model_path="./detection/yolov5/weights/best.pt")
