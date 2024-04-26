import streamlit as st
import cv2
import numpy as np
from collections import deque
from tools import generate_detections as gdet
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import nn_matching
from deep_sort import preprocessing
#from yolov3_tf2.utils import convert_boxes
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.models import YoloV3
from datetime import datetime
from absl import flags
import sys

FLAGS = flags.FLAGS
FLAGS(sys.argv)


@st.cache(allow_output_mutation=True)
def load_model():
    class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
    yolo = YoloV3(classes=len(class_names))
    yolo.load_weights('./weights/yolov3.weights')

    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 0.8

    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric(
        'cosine', max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    return yolo, encoder, tracker, class_names


def current_time():
    now = datetime.now().isoformat()
    return now


def main():
    st.title("Object Tracking with Streamlit")

    yolo, encoder, tracker, class_names = load_model()

    vid = cv2.VideoCapture('./data/video/HSE2.mp4')
    codec = cv2.VideoWriter_fourcc(*'XVID')
    vid_fps = int(vid.get(cv2.CAP_PROP_FPS))
    vid_width, vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('./data/video/results.mp4', codec, vid_fps, (vid_width, vid_height))

    while True:
        _, img = vid.read()
        if img is None:
            st.write('Completed')
            break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, 416)

        boxes, scores, classes, nums = yolo.predict(img_in)

        # Rest of the code for tracking and drawing bounding boxes

        # Display the processed image
        st.image(img, channels="BGR", use_column_width=True)

        out.write(img)

        if cv2.waitKey(30) == ord('q'):
            break

    vid.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
