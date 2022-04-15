# import cv2
#
#
# cap = cv2.VideoCapture("rtsp://admin:edge2021@192.168.13.224")
# ret, frame = cap.read()
# while ret:
#     ret, frame = cap.read()
#     cv2.namedWindow("frame", 0)
#     cv2.imshow("frame", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()
# cap.release()


from __future__ import division
import cv2
import time
import sys
import queue
import threading
from det_infer import DetectInfer
from track import JDETracker

q = queue.Queue()

def receive():
    print("start Receive")
    source = "rtsp://admin:edge2021@192.168.13.224"
    if len(sys.argv) > 1:
        source = sys.argv[1]
    cap = cv2.VideoCapture(source)
    hasFrame, frame = cap.read()
    q.put(frame)
    while (hasFrame):
        hasFrame, frame = cap.read()
        q.put(frame)


def display():
    print("start Display")
    frame_count = 0
    while (1):
        if q.empty() != True:
            frame = q.get()
            frame_count += 1
            # t = time.time()
            out_frame = frame.copy()
            # detect_infer.infer(out_frame, frame_count)
            # # detect_infer.save_result()
            # det_results = detect_infer.det_result_info.det_results_vector
            # for det_result in det_results:
            #     p1, p2 = (int(det_result.pedestrian_location[0]), int(det_result.pedestrian_location[1])), \
            #              (int(det_result.pedestrian_location[2]), int(det_result.pedestrian_location[3]))
            #     cv2.rectangle(out_frame, p1, p2, (0, 128, 128), thickness=2, lineType=cv2.LINE_AA)

            jde_tracker.process(out_frame, frame_count)
            det_results = jde_tracker.det_result_info.det_results_vector
            for det_result in det_results:
                p1, p2 = (int(det_result.pedestrian_location[0]), int(det_result.pedestrian_location[1])), \
                         (int(det_result.pedestrian_location[2]), int(det_result.pedestrian_location[3]))
                cv2.rectangle(out_frame, p1, p2, (0, 128, 128), thickness=2, lineType=cv2.LINE_AA)
                label = f'{det_result.track_id} {det_result.class_id} {det_result.confidence:.2f}'
                tf = max(2 - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=1 / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h - 3 >= 0  # label fits outside box
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(out_frame, p1, p2, (0, 128, 128), -1, cv2.LINE_AA)  # filled
                cv2.putText(out_frame, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, 1 / 3, (255, 255, 255),
                            thickness=tf, lineType=cv2.LINE_AA)

            cv2.namedWindow("Face Detection Comparison", 0)
            cv2.imshow("Face Detection Comparison", out_frame)
            # vid_writer.write(outOpencvDnn)
            key = cv2.waitKey(1)
            if key == 27:
                break


if __name__ == "__main__":
    source = "rtsp://admin:edge2021@192.168.13.224"
    if len(sys.argv) > 1:
        source = sys.argv[1]

    cap = cv2.VideoCapture(source)
    hasFrame, frame = cap.read()

    # vid_writer = cv2.VideoWriter('output-dnn-{}.avi'.format(str(source).split(".")[0]),
    #                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (frame.shape[1], frame.shape[0]))

    # detect_infer = DetectInfer("yolov5Det", conf_thresh=0.5, iou_thresh=0.45, class_filter=0)
    # detect_infer.set_model_param(model_path="./detection/yolov5/weights/best.pt",
    #                              augment=False, agnostic_nms=False)

    jde_tracker = JDETracker("tracker/deep_sort_pytorch/configs/deep_sort.yaml", "yolov5Det",
                             model_path="./detection/yolov5/weights/best.pt")

    p1 = threading.Thread(target=receive)
    p2 = threading.Thread(target=display)
    p1.start()
    p2.start()

    cv2.destroyAllWindows()
    # vid_writer.release()