#===============================================================#
#                                                               #
#   File name   : YOLOv4_model.py                               #
#   Author      : hxnghia99                                     #
#   Created date: May 24th, 2022                                #
#   GitHub      : https://github.com/hxnghia99/Yolov4_Learning  #
#   Description : YOLOv4 configuration parameters               #
#                                                               #
#===============================================================#




#YOLO Model
YOLO_SCALE_OFFSET               = [8, 16, 32]
YOLO_ANCHORS                    = [[[12,  16], [19,   36], [40,   28]],
                                   [[36,  75], [76,   55], [72,  146]],
                                   [[142,110], [192, 243], [459, 401]]]
ANCHORS_PER_GRID_CELL           = 3

YOLO_COCO_CLASS_PATH            = "YOLOv4-for-studying/coco/coco.names"
YOLO_LG_CLASS_PATH              = "YOLOv4-for-studying/LG_DATASET/lg_class_names.txt"

YOLO_INPUT_SIZE                 = 416
USE_LOADED_WEIGHT               = True

#YOLO Load model
YOLO_V4_COCO_WEIGHTS            = "YOLOv4-for-studying/model_data/yolov4.weights"
YOLO_V4_LG_WEIGHTS              = "YOLOv4-for-studying/checkpoints/YOLOv4_custom"

#YOLO training  
YOLO_LOSS_IOU_THRESHOLD         = 0.5

#Dataset configurations
TRAIN_ANNOTATION_PATH           = "YOLOv4-for-studying/LG_DATASET/train_400samples.txt"
TRAIN_INPUT_SIZE                = YOLO_INPUT_SIZE
TRAIN_BATCH_SIZE                = 4
TRAIN_DATA_AUG                  = True
TEST_ANNOTATION_PATH            = "YOLOv4-for-studying/LG_DATASET/test_100samples.txt"
TEST_INPUT_SIZE                 = YOLO_INPUT_SIZE
TEST_BATCH_SIZE                 = 4
TEST_DATA_AUG                   = False

RELATIVE_PATH                   = 'E:/dataset/TOTAL/'
PREFIX_PATH                     = '.\YOLOv4-for-studying\LG_DATASET'

YOLO_MAX_BBOX_PER_SCALE         = 100
ANCHOR_SELECTION_IOU_THRESHOLD  = 0.3


#Training settings
TRAIN_LOGDIR                   = 'YOLOv4-for-studying/log/'
TRAIN_CHECKPOINTS_FOLDER        = "YOLOv4-for-studying/checkpoints"
TRAIN_MODEL_NAME                = "YOLOv4_custom"
TRAIN_MODEL_WEIGHTS             = "YOLOv4-for-studying/checkpoints/YOLOv4_transfer.weights"
TRAIN_SAVE_BEST_ONLY            = True  # saves only best model according validation loss (True recommended)
TRAIN_SAVE_CHECKPOINT           = False # saves all best validated checkpoints in training process (may require a lot disk space) (False recommended)
TRAIN_TRANSFER                  = True

TRAIN_WARMUP_EPOCHS             = 2
TRAIN_EPOCHS                    = 50
TRAIN_LR_END                    = 1e-6
TRAIN_LR_INIT                   = 1e-4
YOLO_LOSS_IOU_THRESHOLD         = 0.5

#Evaluation settings
VALIDATE_SCORE_THRESHOLD        = 0.05
VALIDATE_IOU_THRESHOLD          = 0.5
USE_CUSTOM_WEIGHTS              = True
VALIDATE_GT_RESULTS_DIR         = 'YOLOv4-for-studying/mAP/ground-truth'
VALIDATE_MAP_RESULT_PATH        = "YOLOv4-for-studying/mAP/results.txt"

COCO_VALIDATATION               = False
if COCO_VALIDATATION:
    RELATIVE_PATH               = "./model_data/"  
    PREFIX_PATH                 = '.\YOLOv4-for-studying'
    TEST_ANNOTATION_PATH        = "YOLOv4-for-studying/coco/val2017.txt"
    VALIDATE_GT_RESULTS_DIR     = 'YOLOv4-for-studying/mAP/ground-truth-coco'
    VALIDATE_MAP_RESULT_PATH    = "YOLOv4-for-studying/mAP/results_coco.txt"