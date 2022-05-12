#===============================================================#
#                                                               #
#   File name   : Test.py                                       #
#   Author      : hxnghia99                                     #
#   Created date: April 28th, 2022                              #
#   GitHub      : https://github.com/hxnghia99/Yolov4_Learning  #
#   Description : YOLOv3 image prediction testing               #
#                                                               #
#===============================================================#



import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from YOLOv3_utils import *
from YOLOv3_config import *
import cv2

# IMAGE_PATH = "./YOLOv3-for-studying/IMAGES/kite.jpg"
# # IMAGE_PATH = "./YOLOv3-for-studying/IMAGES/street.jpg"

# yolo = Load_YOLOv3_Model()
# detect_image(yolo, IMAGE_PATH, show=True, save=False, CLASS_FILE=YOLO_COCO_CLASS_DIR)


from tensorflow.keras.datasets import mnist
(train_X, train_y), (test_X, test_y) = mnist.load_data()
cv2.imshow('abc', train_X[0])