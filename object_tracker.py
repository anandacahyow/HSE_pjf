from _collections import deque
from tools import generate_detections as gdet
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import nn_matching
from deep_sort import preprocessing
from yolov3_tf2.utils import convert_boxes
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.models import YoloV3
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
from datetime import datetime, time
from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)


def current_time():
    now = datetime.now().isoformat()
    return now


class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('./weights/yolov3.tf')

max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8

model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric(
    'cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

vid = cv2.VideoCapture('./data/video/HSE2.mp4')

codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps = int(vid.get(cv2.CAP_PROP_FPS))
vid_width, vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
    vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./data/video/results.mp4', codec,
                      vid_fps, (vid_width, vid_height))

pts = [deque(maxlen=30) for _ in range(1000)]

counter = []
count = 0
count_temp = 0
track_temp = 0

# WEBCAM
#vid = cv2.VideoCapture(0)

while True:
    _, img = vid.read()
    if img is None:
        print('Completed')
        break

    count += 1
    if count % 5 != 0:
        continue

    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in = tf.expand_dims(img_in, 0)
    img_in = transform_images(img_in, 416)
    font_scale = 1
    font = cv2.FONT_HERSHEY_PLAIN

    # t1 = time.time()

    # RED ZONE AREA
    area = [(256, 13), (351, 13), (351, 263), (256, 263)]  # AREA HSE 2
    # area = [(181, 45), (306, 45), (306, 232), (181, 232)]  # AREA HSE 1
    cv2.polylines(img, [np.array(area, np.int32)], True, (0, 0, 255), 6)
    cv2.rectangle(img, (256, 13), (351, 30), (0, 0, 255), -1)
    cv2.putText(img, "Red Zone", (265, 25), font,
                fontScale=font_scale,  color=(255, 255, 250), thickness=2)

    # PEOPLE DETECTION
    boxes, scores, classes, nums = yolo.predict(img_in)

    classes = classes[0]
    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    names = np.array(names)
    converted_boxes = convert_boxes(img, boxes[0])
    features = encoder(img, converted_boxes)

    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores[0], names, features)]

    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(
        boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]
    # print(detections)

    # PEOPLE TRACKING
    tracker.predict()
    tracker.update(detections)

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    current_count = int(0)

    # BOUNDING BOX
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        class_name = track.get_class()
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]

        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(
            len(class_name) + len(str(track.track_id)))*17, int(bbox[1])), color, -1)
        cv2.putText(img, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
                    (255, 255, 255), 2)

        # RED ZONE DETECTION
        cx = int(((bbox[0]) + (bbox[2]))/2)
        cy = int(((bbox[1])+(bbox[3]))/2)

        result = cv2.pointPolygonTest(
            np.array(area, np.int32), (cx, cy), False)

        if result == 1:
            if track.track_id != track_temp:
                print(
                    "\n===================================================================================================")
                print("[+] ALERT: RED ZONE OCCUPIED!!")
                print(
                    f"Time: {current_time()}:: {class_name} no {str(track.track_id)} is Detected in central coord (x,y): {cx,cy}")
                print(
                    "===================================================================================================\n")
                track_temp = int(track.track_id)
            else:
                print("[/] already reported")
                count_temp += 1
                if count_temp % 10 == 0:
                    track_temp = []
            print(
                f"NILAI ID TRACKED {track.track_id} // NILAI ID TEMP {track_temp}")
        else:
            print("[-] No Person Detected")
        #print(f"NILAI ID TRACKED {track.track_id} // NILAI ID TEMP {track_temp}")

        cv2.circle(img, (cx, cy),
                   radius=5, color=(0, 0, 255), thickness=5)

    # fps = 1./(time.time()-t1)
    # cv2.putText(img, f"Time:{current_time()}", (0, 30), 0, 1, (0, 0, 255), 2)
    # cv2.resizeWindow('output', 1024, 768)
    cv2.imshow('output', img)
    out.write(img)

    if cv2.waitKey(30) == ord('q'):
        break
vid.release()
out.release()
cv2.destroyAllWindows()
