import numpy as np
import time
import cv2


class TrajectoryShow():
    def __init__(self):
        super().__init__()
        self.bird_draw_x_start = 0
        self.bird_draw_x_end = 2560
        self.bird_direction_line = [30, 270, 510]
        self.color = (0, 128, 128)
        self.txt_color = (255, 255, 255)

    def image_show(self, image, window_name):
        cv2.namedWindow(window_name, 0)
        # cv2.resizeWindow(window_name, (int(image.shape[1]*0.7), int(image[0]*0.7)))
        cv2.imshow(window_name, image)
        key = cv2.waitKey()

    def trajectory_show(self, image, track_idx_map, bird=None):
        bird_image = np.zeros((image.shape[0], self.bird_draw_x_end - self.bird_draw_x_start, 3))
        crowded_counted = len(track_idx_map)
        system_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        for key, value in track_idx_map.items():
            p1, p2 = (int(value.pedestrian_x_start[-1]), int(value.pedestrian_y_start[-1])), \
                     (int(value.pedestrian_x_end[-1]), int(value.pedestrian_y_end[-1]))
            # p1, p2 = (int(value.pedestrian_location[0]), int(value.pedestrian_location[1])), \
            #          (int(value.pedestrian_location[2]), int(value.pedestrian_location[3]))
            # p1_head, p2_head = (value.head_location[0], value.head_location[1]), \
            #                    (value.head_location[2], value.head_location[3])
            cv2.rectangle(image, p1, p2, self.color, thickness=1, lineType=cv2.LINE_AA)
            # cv2.rectangle(self.image, p1_head, p2_head, (255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
            label = f'{key} {value.pedestrian_direction} {value.mean_velocity:.2f} {value.relative_distance:.2f}'  # mean_velocity

            tf = max(2 - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=1 / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, self.color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, 1 / 3, self.txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)
            for point in value.trajectory_position:
                cv2.circle(image, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
            cv2.putText(image, f"Counted: {crowded_counted}", (20, 60), 0, 2 / 3, (255, 255, 255),
                        thickness=tf, lineType=cv2.LINE_AA)
            cv2.putText(image, f"{system_time}", (20, 30), 0, 2 / 3, (255, 255, 255),
                        thickness=tf, lineType=cv2.LINE_AA)

            # bird image show
            p1_bird, p2_bird = (int(value.trajectory_bird_position[-1][0] - self.bird_draw_x_start - 5), \
                                int(value.trajectory_bird_position[-1][1] - 5)), \
                               (int(value.trajectory_bird_position[-1][0] - self.bird_draw_x_start + 5), \
                                int(value.trajectory_bird_position[-1][1] + 5))
            cv2.rectangle(bird_image, p1_bird, p2_bird, self.color, thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(bird_image, label, (p1_bird[0], p1_bird[1] - 2 if outside else p1_bird[1] + h + 2), 0, 3 / 4,
                        self.txt_color, thickness=tf, lineType=cv2.LINE_AA)
            if value.pedestrian_direction == 0:
                cv2.line(bird_image, (int(p2_bird[0] - 5), int(p2_bird[1] - 2)), (int(p2_bird[0] - 5), \
                                                                                  int(p2_bird[1] + 2)), (0, 255, 0), 2)
            elif value.pedestrian_direction == 1:
                cv2.line(bird_image, (int(p1_bird[0] + 5), int(p1_bird[1] - 2)), (int(p1_bird[0] + 5), \
                                                                                  int(p1_bird[1] + 2)), (0, 255, 0), 2)
            for line_height_point in self.bird_direction_line:
                cv2.line(bird_image, (0, line_height_point), (bird_image.shape[1], line_height_point), (255, 0, 0), 1)
            for bird_point in value.trajectory_bird_position:
                cv2.circle(bird_image, ((bird_point[0] - self.bird_draw_x_start), bird_point[1]), 1, (0, 0, 255), -1)
            cv2.putText(bird_image, f"Counted: {crowded_counted}", (10, 30), 0, 1 / 3, (255, 255, 0),
                        thickness=tf, lineType=cv2.LINE_AA)
            cv2.putText(bird_image, f"{system_time}", (10, 15), 0, 1 / 3, (255, 255, 0),
                        thickness=tf, lineType=cv2.LINE_AA)

            for bird_point in value.trajectory_bird_position:
                cv2.circle(bird, ((bird_point[0] - self.bird_draw_x_start), bird_point[1]), 3, (0, 0, 255), -1)

        self.image_show(image, "traj_show")
        self.image_show(bird_image, "bird_map_show")
        self.image_show(bird, "bird_show")