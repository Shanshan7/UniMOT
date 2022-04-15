import numpy as np
import cv2
import os
import json
import codecs


class BirdView():

    def __init__(self, camera_calibration_file):
        super().__init__()
        self.camera_calibration_file = camera_calibration_file
        if not os.path.exists(self.camera_calibration_file):
            print("Camera Calibration File is not exists!")

    def bird_view_matrix_calculate(self, image=None, pts=None):
        self.image = image
        self.pts = pts
        if self.image is not None:
            self.c, self.r = self.image.shape[0:2]

        with codecs.open(self.camera_calibration_file, 'r', encoding='utf-8') as f:
            camera_calibration_paramaters = json.load(f)
        paramater = camera_calibration_paramaters['paramater']
        self.transferI2B = paramater['transfer_Image2Bird']
        self.transferI2B = np.array(self.transferI2B)
        self.transferB2I = paramater['transfer_Bird2Image']
        self.pixel2world_distance = paramater['pixel2world_distance']
        self.time_interval = paramater['time_interval']
        self.camera_position_pixel = paramater['camera_position_pixel']
        self.translation_position = paramater['translation_position']

        # self.image_resized = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)), interpolation=cv2.INTER_LINEAR)
        # self.c, self.r = self.image_resized.shape[0:2]
        # pst2 = np.float32([[180,162],[618,0],[552,540],[682,464]])
        # pst1 = np.float32([[0, 0], [self.r, 0], [0, self.c], [self.r, self.c]])
        # self.transferI2B = cv2.getPerspectiveTransform(pst1, pst2)
        # self.transferB2I = cv2.getPerspectiveTransform(pst2, pst1)

    def img2bird(self):
        self.bird = cv2.warpPerspective(self.image, self.transferI2B, (self.r, self.c))
        return self.bird

    def bird2img(self):
        self.image = cv2.warpPerspective(self.bird, self.transferB2I, (self.r, self.c))
        return self.image

    def convrt2Bird(self, img):
        c, r = img.shape[0:2]
        return cv2.warpPerspective(img, self.transferI2B, (r, c))

    def convrt2Image(self, bird):
        return cv2.warpPerspective(bird, self.transferB2I, (self.r, self.c))

    def projection_on_bird(self, p):
        M = self.transferI2B
        px = (M[0][0] * p[0] + M[0][1] * p[1] + M[0][2]) / (M[2][0] * p[0] + M[2][1] * p[1] + M[2][2])
        py = (M[1][0] * p[0] + M[1][1] * p[1] + M[1][2]) / (M[2][0] * p[0] + M[2][1] * p[1] + M[2][2])
        return (int(px), int(py))

    def projection_on_image(self, p):
        M = self.transferB2I
        px = (M[0][0] * p[0] + M[0][1] * p[1] + M[0][2]) / (M[2][0] * p[0] + M[2][1] * p[1] + M[2][2])
        py = (M[1][0] * p[0] + M[1][1] * p[1] + M[1][2]) / (M[2][0] * p[0] + M[2][1] * p[1] + M[2][2])
        return (int(px), int(py))

    def points_projection_on_image(self, center, radius):
        x, y = center
        points = self.midPointCircleDraw(x, y, radius)
        original = np.array([points], dtype=(np.float32))
        cvd = cv2.perspectiveTransform(original, self.transferB2I)
        return cvd[0]

    def midPointCircleDraw(self, x_centre, y_centre, r):
        points = []
        x = r
        y = 0
        points.append((x + x_centre, y + y_centre))
        if r > 0:
            points.append((x + x_centre, -y + y_centre))
            points.append((y + x_centre, x + y_centre))
            points.append((-y + x_centre, x + y_centre))
        P = 1 - r
        while x > y:
            y += 1
            if P <= 0:
                P = P + 2 * y + 1
            else:
                x -= 1
                P = P + 2 * y - 2 * x + 1
            if x < y:
                break
            points.append((x + x_centre, y + y_centre))
            points.append((-x + x_centre, y + y_centre))
            points.append((x + x_centre, -y + y_centre))
            points.append((-x + x_centre, -y + y_centre))
            if x != y:
                points.append((y + x_centre, x + y_centre))
                points.append((-y + x_centre, x + y_centre))
                points.append((y + x_centre, -x + y_centre))
                points.append((-y + x_centre, -x + y_centre))

        return points
