#===============================================================#
#                                                               #
#   File name   : YOLOv3_config.py                              #
#   Author      : hxnghia99                                     #
#   Created date: April 28th, 2022                              #
#   GitHub      : https://github.com/hxnghia99/Yolov4_Learning  #
#   Description : YOLOv3 configurations                         #
#                                                               #
#===============================================================#
import numpy as np

YOLO_SCALE_OFFSET           = [8, 16, 32]

YOLO_ANCHORS                = [[[10,  13], [16,   30], [33,   23]],
                               [[30,  61], [62,   45], [59,  119]],
                               [[116, 90], [156, 198], [373, 326]]]

YOLO_COCO_CLASS_DIR         = "YOLOv3-for-studying/coco/coco.names"
YOLO_V3_WEIGHTS             = "YOLOv3-for-studying/model_data/yolov3.weights"

USE_LOADED_WEIGHT           = True
YOLO_INPUT_SIZE             = 416
