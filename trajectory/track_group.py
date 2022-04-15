# uncompyle6 version 3.7.4
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.6.12 |Anaconda, Inc.| (default, Sep  8 2020, 23:10:56)
# [GCC 7.3.0]
# Embedded file name: deepsocial.py
# Compiled at: 2021-03-06 05:54:51
# Size of source mod 2**32: 14036 bytes
import cv2, numpy as np
from itertools import combinations


def find_zone(centroid_dict, _greenZone, _redZone, criteria):
    redZone = []
    greenZone = []
    for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
        distance = Euclidean_distance(p1[0:2], p2[0:2])
        if distance < criteria:
            if id1 not in redZone:
                redZone.append(int(id1))
            if id2 not in redZone:
                redZone.append(int(id2))

    for idx, box in centroid_dict.items():
        if idx not in redZone:
            greenZone.append(idx)

    return (
     redZone, greenZone)


def find_relation(centroid_dict, criteria, redZone, _couples, _relation):
    pairs = list()
    memberships = dict()
    for p1 in redZone:
        for p2 in redZone:
            if p1 != p2:
                distanceX, distanceY = Euclidean_distance_seprate(centroid_dict[p1], centroid_dict[p2])
                if p1 < p2:
                    pair = (
                     p1, p2)
                else:
                    pair = (
                     p2, p1)
                if _couples.get(pair):
                    distanceX = distanceX * 0.6
                    distanceY = distanceY * 0.6
                if distanceX < criteria[0]:
                    if distanceY < criteria[1]:
                        if memberships.get(p1):
                            memberships[p1].append(p2)
                        else:
                            memberships[p1] = [
                             p2]
                    if pair not in pairs:
                        pairs.append(pair)

    relation = dict()
    for pair in pairs:
        if _relation.get(pair):
            _relation[pair] += 1
            relation[pair] = _relation[pair]
        else:
            _relation[pair] = 1

    obligation = {}
    for p in memberships:
        top_relation = 0
        for secP in memberships[p]:
            if p < secP:
                pair = (
                 p, secP)
            else:
                pair = (
                 secP, p)
            if relation.get(pair) and top_relation < relation[pair]:
                top_relation = relation[pair]
                obligation[p] = secP

    couple = dict()
    for m1 in memberships:
        for m2 in memberships:
            if m1 != m2 and obligation.get(m1) and obligation.get(m2) and obligation[m1] == m2:
                if obligation[m2] == m1:
                    if m1 < m2:
                        pair = (
                         m1, m2)
                    else:
                        pair = (
                         m2, m1)
                couple[pair] = relation[pair]

    return (
     _relation, couple)


def find_couples(img, centroid_dict, relation, criteria, _couples):
    couples = dict()
    coupleZone = list()
    for pair in relation:
        proxTime = relation[pair]
        if proxTime > criteria:
            coupleZone.append(pair[0])
            coupleZone.append(pair[1])
            couplesBox = center_of_2box(centroid_dict[pair[0]], centroid_dict[pair[1]])
            if _couples.get(pair):
                couplesID = _couples[pair]['id']
                _couples[pair]['box'] = couplesBox
            else:
                couplesID = len(_couples) + 1
                _couples[pair] = {'id':couplesID,  'box':couplesBox}
            couples[pair] = _couples[pair]

    return (
     _couples, couples, coupleZone)

def Euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def Euclidean_distance_seprate(p1, p2):
    dX = np.sqrt((p1[0] - p2[0]) ** 2)
    dY = np.sqrt((p1[1] - p2[1]) ** 2)
    return (dX, dY)


def checkupArea(img, leftRange, downRange, point, color='g', Draw=False):
    hmax, wmax = img.shape[0:2]
    hmin = hmax - int(hmax * downRange)
    wmin = int(wmax * leftRange)
    if Draw:
        if color == 'r':
            color = (0, 0, 255)
        if color == 'g':
            color = (0, 255, 0)
        if color == 'b':
            color = (255, 0, 0)
        if color == 'k':
            color = (0, 0, 0)
        cv2.line(img, (0, hmin), (wmax, hmin), color, 1)
        cv2.line(img, (wmin, 0), (wmin, hmax), color, 1)
    x, y = point
    if x < wmin:
        if y > hmin:
            return True
    return False


def center_of_2box(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1[4:8]
    xmin2, ymin2, xmax2, ymax2 = box2[4:8]
    if xmin1 < xmin2:
        xmin = xmin1
    else:
        xmin = xmin2
    if ymin1 < ymin2:
        ymin = ymin1
    else:
        ymin = ymin2
    if xmax1 > xmax2:
        xmax = xmax1
    else:
        xmax = xmax2
    if ymax1 > ymax2:
        ymax = ymax1
    else:
        ymax = ymax2
    ymax -= 5
    box = (xmin, ymin, xmax, ymax)
    w = xmax - xmin
    h = ymax - ymin
    x = xmin + w / 2
    y = ymax - h / 2
    return (
     int(x), int(y), xmin, ymin, xmax, ymax)

def centroid(detections, image, bird_view, _centroid_dict, CorrectionShift, HumanHeightLimit):
    centroid_dict = dict()
    now_present = list()
    if len(detections) > 0:
        for d in detections:
            p = int(d.track_id)
            now_present.append(p)
            xmin, ymin, xmax, ymax = d.pedestrian_location[0], \
                                     d.pedestrian_location[1], \
                                     d.pedestrian_location[2], \
                                     d.pedestrian_location[3],
            w = xmax - xmin
            h = ymax - ymin
            x = xmin + w/2
            y = ymax - h/2
            if h < HumanHeightLimit:
                overley = image
                bird_x, bird_y = bird_view.projection_on_bird((x, ymax))
                if CorrectionShift:
                    if checkupArea(overley, 1, 0.25, (x, ymin)):
                        continue
                center_bird_x, center_bird_y = bird_view.projection_on_bird((x, ymin))
                centroid_dict[p] = (
                            int(bird_x), int(bird_y),
                            int(x), int(ymax),
                            int(xmin), int(ymin), int(xmax), int(ymax),
                            int(center_bird_x), int(center_bird_y))

                _centroid_dict[p] = centroid_dict[p]
    return _centroid_dict, centroid_dict


if __name__ == '__main__':
    from tracker.base_sde_tracker import SDETracker
    from trajectory.bird_view import BirdView

    ######################## Units are Pixel
    ViolationDistForIndivisuals = 28
    ViolationDistForCouples = 31
    ######################## (0:OFF/ 1:ON)
    CorrectionShift = 1  # Ignore people in the margins of the video
    HumanHeightLimit = 200  # Ignore people with unusual heights
    MembershipDistForCouples = (16, 10)  # (Forward, Behind) per Pixel
    MembershipTimeForCouples = 35  # Time for considering as a couple (per Frame)

    _greenZone = list()
    _redZone = list()
    _yellowZone = list()
    _final_redZone = list()

    _centroid_dict = dict()
    _relation = dict()
    _couples = dict()

    bird_view = BirdView("calibration_211221.json")
    sde_tracker = SDETracker("./tracker/deep_sort_pytorch/configs/deep_sort.yaml", "yolov5Det",
                             model_path="./detection/yolov5/weights/best.pt")

    for i in range(1, 2000):
        input_data = cv2.imread(
            "/home/edge/data/VOCdevkit/videos/MOT11/" + str(i) + ".jpg")
        input_resized = cv2.resize(input_data, (int(input_data.shape[1] / 2), int(input_data.shape[0] / 2)), interpolation=cv2.INTER_LINEAR)
        sde_tracker.process(input_resized, i)
        sde_tracker.show_result()
        # sde_tracker.save_result()

        image = sde_tracker.det_result_info.image
        det_results = sde_tracker.det_result_info.det_results_vector
        bird_view.bird_view_matrix_calculate()

        _centroid_dict, centroid_dict = centroid(det_results, image, bird_view, _centroid_dict, CorrectionShift, HumanHeightLimit)
        redZone, greenZone = find_zone(centroid_dict, _greenZone, _redZone, criteria=ViolationDistForIndivisuals)

        _relation, relation = find_relation(centroid_dict, MembershipDistForCouples, redZone, _couples, _relation)
        _couples, couples, coupleZone = find_couples(image, _centroid_dict, relation, MembershipTimeForCouples, _couples)

        DTCShow = image.copy()
        for id, box in centroid_dict.items():
            center_bird = box[0], box[1]
            if not id in coupleZone:
                cv2.rectangle(DTCShow,(box[4], box[5]),(box[6], box[7]),(0,255,0),2)
                cv2.rectangle(DTCShow,(box[4], box[5]-13),(box[4]+len(str(id))*10, box[5]),(0,200,255),-1)
                cv2.putText(DTCShow,str(id),(box[4]+2, box[5]-2),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1,cv2.LINE_AA)
        for coupled in couples:
            p1 , p2 = coupled
            couplesID = couples[coupled]['id']
            couplesBox = couples[coupled]['box']
            cv2.rectangle(DTCShow, couplesBox[2:4], couplesBox[4:], (0,150,255), 4)
            loc = couplesBox[0] , couplesBox[3]
            offset = len(str(couplesID)*5)
            captionBox = (loc[0] - offset, loc[1]-13), (loc[0] + offset, loc[1])
            cv2.rectangle(DTCShow,captionBox[0],captionBox[1],(0,200,255),-1)
            wc = captionBox[1][0] - captionBox[0][0]
            hc = captionBox[1][1] - captionBox[0][1]
            cx = captionBox[0][0] + wc // 2
            cy = captionBox[0][1] + hc // 2
            textLoc = (cx - offset, cy + 4)
            cv2.putText(DTCShow, str(couplesID) ,(textLoc),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1,cv2.LINE_AA)

        cv2.imshow("result", DTCShow)
        cv2.waitKey(1)