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

YOLO_SCALE_OFFSET               = [8, 16, 32]

YOLO_ANCHORS                    =   [[[10,  13], [16,   30], [33,   23]],
                                    [[30,  61], [62,   45], [59,  119]],
                                    [[116, 90], [156, 198], [373, 326]]]

YOLO_COCO_CLASS_DIR             = "YOLOv3-for-studying/coco/coco.names"
YOLO_V3_COCO_WEIGHTS            = "YOLOv3-for-studying/model_data/yolov3.weights"
YOLO_V3_LG_WEIGHTS              = "YOLOv3-for-studying/checkpoints/YOLOv3_custom"

USE_LOADED_WEIGHT               = True
YOLO_INPUT_SIZE                 = 416

#Dataset configurations
LG_CLASS_NAMES_PATH             = "YOLOv3-for-studying/LG_DATASET/lg_class_names.txt"
TRAIN_ANNOTATION_PATH           = "YOLOv3-for-studying/LG_DATASET/train.txt"
TRAIN_INPUT_SIZE                = YOLO_INPUT_SIZE
TRAIN_BATCH_SIZE                = 6
TRAIN_DATA_AUG                  = True
TEST_ANNOTATION_PATH            = "YOLOv3-for-studying/LG_DATASET/test.txt"
TEST_INPUT_SIZE                 = YOLO_INPUT_SIZE
TEST_BATCH_SIZE                 = 6
TEST_DATA_AUG                   = False

RELATIVE_PATH                   = 'E:/dataset/TOTAL/'
PREFIX_PATH                     = '.\YOLOv3-for-studying\LG_DATASET'

YOLO_MAX_BBOX_PER_SCALE         = 100
ANCHORS_PER_GRID_CELL           = 3
ANCHOR_SELECTION_IOU_THRESHOLD  = 0.3

#Training settings
TRAIN_LOGDIR                    = 'YOLOv3-for-studying/log/'
TRAIN_CHECKPOINTS_FOLDER        = "YOLOv3-for-studying/checkpoints"
TRAIN_MODEL_NAME                = "YOLOv3_custom"
TRAIN_MODEL_WEIGHTS             = "YOLOv3-for-studying/checkpoints/YOLOv3_transfer.weights"
TRAIN_SAVE_BEST_ONLY            = True  # saves only best model according validation loss (True recommended)
TRAIN_SAVE_CHECKPOINT           = False # saves all best validated checkpoints in training process (may require a lot disk space) (False recommended)
TRAIN_TRANSFER                  = True
TRAIN_LOAD_IMAGES_TO_RAM        = False

TRAIN_WARMUP_EPOCHS             = 2
TRAIN_EPOCHS                    = 50
TRAIN_LR_END                    = 1e-6
TRAIN_LR_INIT                   = 1e-4
YOLO_LOSS_IOU_THRESHOLD         = 0.5

LAMBDA_COORD                    = 5
LAMBDA_NOOBJ                    = 0.5

#Evaluation settings
VALIDATE_SCORE_THRESHOLD        = 0.3
VALIDATE_IOU_THRESHOLD          = 0.5
USE_CUSTOM_WEIGHTS              = True
VALIDATE_GT_RESULTS_DIR         = 'YOLOv3-for-studying/mAP/ground-truth'
VALIDATE_MAP_RESULT_PATH        = "YOLOv3-for-studying/mAP/results.txt"

COCO_VALIDATATION               = False
if COCO_VALIDATATION:
    RELATIVE_PATH               = "./model_data/"  
    PREFIX_PATH                 = '.\YOLOv3-for-studying'
    TEST_ANNOTATION_PATH        = "YOLOv3-for-studying/coco/val2017.txt"
    VALIDATE_GT_RESULTS_DIR     = 'YOLOv3-for-studying/mAP/ground-truth-coco'
    VALIDATE_MAP_RESULT_PATH    = "YOLOv3-for-studying/mAP/results_coco.txt"