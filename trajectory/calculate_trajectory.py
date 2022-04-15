import cv2
import time
import numpy as np
from scipy import signal
from common.data_struct import TrajectoryParams
from trajectory.track_norm import norm_trajectory, norm_trajectory_array
from trajectory.bird_view import BirdView


class CalculateTraj():

    def __init__(self, camera_calibration_file):
        super().__init__()

        self.camera_calibration_file = camera_calibration_file
        self.bird_view = BirdView(camera_calibration_file)
        self.track_idx_map = {}
        # self.dTs = 1. / 20.  # the interval between frames, assuming that the camera acquisition frame rate is 25 frames
        # self.pixel2world_distance = 0.6 / 48. # 3. / 40.  # assume 40 pixels equal 3 meters

    def calculate_trajectory(self, det_result_info):
        self.image = det_result_info.image
        self.bird_view.bird_view_matrix_calculate()
        self.bird = self.bird_view.convrt2Bird(self.image)
        det_results = det_result_info.det_results_vector
        if len(det_results) > 0:
            for key in list(self.track_idx_map.keys()):
                if self.track_idx_map[key].latest_frame_id != -1 and \
                        (det_results[0].current_frame - self.track_idx_map[key].latest_frame_id) > 3:
                    del self.track_idx_map[key]
                    continue

        for i in range(0, len(det_results)):
            current_frame = det_results[i].current_frame
            track_id = det_results[i].track_id
            # if track_id = -1 detection has not match the track result, we don't need to put out
            if track_id == -1:
                continue
            confidence = det_results[i].confidence
            class_id = det_results[i].class_id
            head_loc = det_results[i].head_location
            pedestrian_location = det_results[i].pedestrian_location
            x_start = det_results[i].pedestrian_location[0]
            y_start = det_results[i].pedestrian_location[1]
            x_end = det_results[i].pedestrian_location[2]
            y_end = det_results[i].pedestrian_location[3]

            if track_id in self.track_idx_map and \
                len(self.track_idx_map[track_id].pedestrian_x_start) > 26:
                position_array = np.array([self.track_idx_map[track_id].pedestrian_x_start, self.track_idx_map[track_id].pedestrian_x_end,
                                  self.track_idx_map[track_id].pedestrian_y_start, self.track_idx_map[track_id].pedestrian_y_end])
                position_array_slide, position_array_output = norm_trajectory_array(position_array,
                                                                                self.track_idx_map[track_id].pedestrian_x_start,
                                                                                self.track_idx_map[track_id].pedestrian_y_start,
                                                                                self.track_idx_map[track_id].pedestrian_x_end,
                                                                                self.track_idx_map[track_id].pedestrian_y_end,
                                                                                self.image.shape[1], self.image.shape[0])
                position_array_slide = position_array_output * self.image.shape[1]
                trajectory_position_current = (int(position_array_slide[0,-1] + (position_array_slide[1,-1] - position_array_slide[0,-1]) / 2), \
                                               int(position_array_slide[3,-1]))
            else:
                trajectory_position_current = (int(x_start + (x_end - x_start) / 2), \
                                                   int(y_end))
            # have problem !!!!!
            trajectory_position_bird = self.bird_view.projection_on_bird((int(trajectory_position_current[0]), \
                                                               int(trajectory_position_current[1])))

            if track_id in self.track_idx_map:
                move_distance = ((self.track_idx_map[track_id].trajectory_bird_position[-1][1] - trajectory_position_bird[1]) \
                                 ** 2) ** 0.5

                if self.track_idx_map[track_id].trajectory_bird_position[-1][1] > trajectory_position_bird[1]:
                    trajectory_direction = 1  # backward
                else:
                    trajectory_direction = 0  # forward

                self.track_idx_map[track_id].trajectory_position.append(trajectory_position_current)
                self.track_idx_map[track_id].trajectory_bird_position.append(trajectory_position_bird)
                self.track_idx_map[track_id].draw_flag = 1
                self.track_idx_map[track_id].latest_frame_id = current_frame
                self.track_idx_map[track_id].pedestrian_direction = trajectory_direction
                self.track_idx_map[track_id].relative_distance = (self.bird_view.camera_position_pixel - trajectory_position_bird[1])\
                                                                 * self.bird_view.pixel2world_distance + self.bird_view.translation_position # det_result_info.image.shape[0]

                velocity_current = move_distance * self.bird_view.pixel2world_distance / self.bird_view.time_interval
                self.track_idx_map[track_id].pedestrian_x_start.append(x_start)
                self.track_idx_map[track_id].pedestrian_y_start.append(y_start)
                self.track_idx_map[track_id].pedestrian_x_end.append(x_end)
                self.track_idx_map[track_id].pedestrian_y_end.append(y_end)
                self.track_idx_map[track_id].pedestrian_location = pedestrian_location
                self.track_idx_map[track_id].head_location = head_loc
                self.track_idx_map[track_id].velocity_vector.append(velocity_current)
                knl_v_slide = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
                v_x_conv_0 = signal.convolve(self.track_idx_map[track_id].velocity_vector, knl_v_slide[::-1]) / sum(knl_v_slide)
                v_x_conv = v_x_conv_0[25:-24]

                # if len(self.track_idx_map[track_id].trajectory_bird_position) < 5:
                #     self.track_idx_map[track_id].mean_velocity = 1.38
                # else:
                #     self.track_idx_map[track_id].mean_velocity = sum(self.track_idx_map[track_id].velocity_vector[-5:]) / 5
                self.track_idx_map[track_id].mean_velocity = v_x_conv[-1]
                self.track_idx_map[track_id].confidence = confidence
                self.track_idx_map[track_id].class_id = class_id
            else:
                trajector_param = TrajectoryParams()
                trajector_param.draw_flag = 1
                trajector_param.latest_frame_id = current_frame
                trajector_param.trajectory_position.append(trajectory_position_current)
                trajector_param.trajectory_bird_position.append(trajectory_position_bird)
                trajector_param.pedestrian_direction = -1
                trajector_param.relative_distance = (self.bird_view.camera_position_pixel - trajectory_position_bird[1])\
                                                                 * self.bird_view.pixel2world_distance
                # 前26frame取第1frame的值
                trajector_param.pedestrian_x_start = [x_start] * 26
                trajector_param.pedestrian_y_start = [y_start] * 26
                trajector_param.pedestrian_x_end = [x_end] * 26
                trajector_param.pedestrian_y_end = [y_end] * 26
                trajector_param.pedestrian_location = pedestrian_location
                trajector_param.head_location = head_loc
                trajector_param.velocity_vector = [0.6] * 26
                trajector_param.mean_velocity = 0.6
                trajector_param.confidence = confidence
                trajector_param.class_id = class_id

                self.track_idx_map[track_id] = trajector_param