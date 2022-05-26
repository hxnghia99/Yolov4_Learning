#================================================================
#
#   File name   : detection_demo.py
#   Author      : PyLessons
#   Created date: 2020-09-27
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : object detection image and video example
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from yolov3.utils import *
from yolov3.configs import *

image_path   = "./IMAGES/street4.jpg"
# video_path   = "./YOLOv3-for-studying/TensorFlow-2.x-YOLOv3/IMAGES/test.mp4"

yolo = Load_Yolo_model()
image1 = detect_image(yolo, image_path, '', input_size=YOLO_INPUT_SIZE, show=True,  rectangle_colors=(255,0,0))


#detect_video(yolo, video_path, "", input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0))
#detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255, 0, 0))
print('a')
#detect_video_realtime_mp(video_path, "Output.mp4", input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0), realtime=False)
