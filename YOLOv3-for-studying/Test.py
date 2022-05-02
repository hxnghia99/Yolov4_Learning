import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from YOLOv3_utils import *
from YOLOv3_config import *

# IMAGE_PATH = "./YOLOv3-for-studying/IMAGES/kite.jpg"
IMAGE_PATH = "./YOLOv3-for-studying/IMAGES/tensorboard.png"

yolo = Load_YOLOv3_Model()
detect_image(yolo, IMAGE_PATH, show=True, save=False, CLASS_FILE=YOLO_COCO_CLASS_DIR, rectangle_colors=(255,0,0))
