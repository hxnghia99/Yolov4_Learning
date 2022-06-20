#===============================================================#
#                                                               #
#   File name   : YOLOv4_model.py                               #
#   Author      : hxnghia99                                     #
#   Created date: May 24th, 2022                                #
#   GitHub      : https://github.com/hxnghia99/Yolov4_Learning  #
#   Description : YOLOv4 configuration parameters               #
#                                                               #
#===============================================================#


""" ------ IMPORTANT SETTING ------ """
# ["COCO", "LG", "VISDRONE"]
TRAINING_DATASET_TYPE           = "VISDRONE"
TRAIN_TRANSFER                  = True

# ["COCO", "LG", "VISDRONE"]
MAKE_EVALUATION                 = False
EVALUATION_DATASET_TYPE         = "VISDRONE"
EVALUATE_TRANSFER               = TRAIN_TRANSFER
""" ---------------------------------"""

#Important initial settings
USE_CIOU_LOSS                   = False
EVALUATE_ORIGINAL_SIZE          = True
USE_NMS_CENTER_D                = False
USE_PRIMARY_EVALUATION_METRIC   = True         #calculate mAP0.5:0.95

#Slicing patch techniques setting
USE_SLICING_PATCH_TECHNIQUE     = True
SLICED_IMAGE_SIZE               = [416, 416]
OVERLAP_RATIO                   = [0.2, 0.2]
MIN_AREA_RATIO                  = 0.2
SLICE_BATCH_SIZE                = 2


TRAIN_BATCH_SIZE                = 2
TEST_BATCH_SIZE                 = 2

#overall settings
YOLO_COCO_CLASS_PATH            = "YOLOv4-for-studying/dataset/coco/coco.names"
YOLO_V4_COCO_WEIGHTS            = "YOLOv4-for-studying/model_data/yolov4.weights"
YOLO_INPUT_SIZE                 = [416, 416]
USE_LOADED_WEIGHT               = True

#Dataset configurations
TRAIN_INPUT_SIZE                = YOLO_INPUT_SIZE
TRAIN_DATA_AUG                  = True
TEST_INPUT_SIZE                 = YOLO_INPUT_SIZE
TEST_DATA_AUG                   = False

#Anchor box settings
YOLO_MAX_BBOX_PER_SCALE         = 1000
ANCHORS_PER_GRID_CELL           = 3
ANCHOR_SELECTION_IOU_THRESHOLD  = 0.3
YOLO_SCALE_OFFSET               = [8, 16, 32]
## COCO anchors
# YOLO_ANCHORS                    = [[[12,  16], [19,   36], [40,   28]],
#                                    [[36,  75], [76,   55], [72,  146]],
#                                    [[142,110], [192, 243], [459, 401]]]
## Visdrone anchors 992x992
# YOLO_ANCHORS                    = [[[6, 8], [10, 17], [18, 12]],
#                                    [[16, 27], [31, 19], [27, 39]],
#                                    [[54, 30], [47, 57], [96, 77]]]
# Visdrone anchors 992x640
YOLO_ANCHORS                    = [[[6, 8], [16, 11], [10, 18]],
                                   [[17, 27], [28, 17], [28, 40]],
                                   [[47, 27], [52, 52], [96, 75]]]


#Training settings
TRAIN_SAVE_BEST_ONLY            = True  # saves only best model according validation loss (True recommended)
TRAIN_SAVE_CHECKPOINT           = False # saves all best validated checkpoints in training process (may require a lot disk space) (False recommended)
TRAIN_LOAD_IMAGES_TO_RAM        = False
TRAIN_WARMUP_EPOCHS             = 2
TRAIN_EPOCHS                    = 50
TRAIN_LR_END                    = 1e-6
TRAIN_LR_INIT                   = 1e-4
YOLO_LOSS_IOU_THRESHOLD         = 0.5


VALIDATE_SCORE_THRESHOLD        = 0.35
VALIDATE_IOU_THRESHOLD          = 0.3


# COCO DATASET has only evaluation dataset
if TRAINING_DATASET_TYPE == "COCO":
    YOLO_CLASS_PATH             = YOLO_COCO_CLASS_PATH
    PREDICTION_WEIGHT_FILE      = YOLO_V4_COCO_WEIGHTS

# LG DATASET has trainset, validationset, testset
elif TRAINING_DATASET_TYPE == "LG":
    YOLO_CLASS_PATH             = "YOLOv4-for-studying/dataset/LG_DATASET/lg_class_names.txt"
    TRAIN_ANNOTATION_PATH       = "YOLOv4-for-studying/dataset/LG_DATASET/train.txt"
    TEST_ANNOTATION_PATH        = "YOLOv4-for-studying/dataset/LG_DATASET/test.txt"
    RELATIVE_PATH               = 'E:/dataset/TOTAL/'
    PREFIX_PATH                 = '.\YOLOv4-for-studying/dataset\LG_DATASET'
    
    if not TRAIN_TRANSFER:
        TRAIN_CHECKPOINTS_FOLDER    = f"YOLOv4-for-studying/checkpoints/{TRAINING_DATASET_TYPE.lower()}_dataset_from_scratch_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/"
        TRAIN_LOGDIR                = f"YOLOv4-for-studying/log/{TRAINING_DATASET_TYPE.lower()}_dataset_from_scratch_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/"
        TRAIN_MODEL_NAME            = f"yolov4_{TRAINING_DATASET_TYPE.lower()}_from_scratch"
    else:
        TRAIN_CHECKPOINTS_FOLDER    = f"YOLOv4-for-studying/checkpoints/{TRAINING_DATASET_TYPE.lower()}_dataset_transfer_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/"
        TRAIN_LOGDIR                = f"YOLOv4-for-studying/log/{TRAINING_DATASET_TYPE.lower()}_dataset_transfer_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/"
        TRAIN_MODEL_NAME            = f"yolov4_{TRAINING_DATASET_TYPE.lower()}_transfer"
    
    PREDICTION_WEIGHT_FILE      = TRAIN_CHECKPOINTS_FOLDER + TRAIN_MODEL_NAME

# VISDRONE DATASET has trainset, validationset, testset
elif TRAINING_DATASET_TYPE == "VISDRONE":
    YOLO_CLASS_PATH             = "YOLOv4-for-studying/dataset/Visdrone_DATASET/visdrone_class_names.txt"
    if not USE_SLICING_PATCH_TECHNIQUE:
        TRAIN_ANNOTATION_PATH       = f"YOLOv4-for-studying/dataset/Visdrone_DATASET/train.txt"
        TEST_ANNOTATION_PATH        = "YOLOv4-for-studying/dataset/Visdrone_DATASET/validation.txt"
    else:
        TRAIN_ANNOTATION_PATH       = f"YOLOv4-for-studying/dataset/Visdrone_DATASET/train_slice.txt"
        TEST_ANNOTATION_PATH        = "YOLOv4-for-studying/dataset/Visdrone_DATASET/validation_slice.txt"
    RELATIVE_PATH               = ""
    PREFIX_PATH                 = ""
    
    if not TRAIN_TRANSFER:
        TRAIN_CHECKPOINTS_FOLDER    = f"YOLOv4-for-studying/checkpoints/{TRAINING_DATASET_TYPE.lower()}_dataset_from_scratch_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/"
        TRAIN_LOGDIR                = f"YOLOv4-for-studying/log/{TRAINING_DATASET_TYPE.lower()}_dataset_from_scratch_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/"
        TRAIN_MODEL_NAME            = f"yolov4_{TRAINING_DATASET_TYPE.lower()}_from_scratch"
    else:
        TRAIN_CHECKPOINTS_FOLDER    = f"YOLOv4-for-studying/checkpoints/{TRAINING_DATASET_TYPE.lower()}_dataset_transfer_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/"
        TRAIN_LOGDIR                = f"YOLOv4-for-studying/log/{TRAINING_DATASET_TYPE.lower()}_dataset_transfer_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/"
        TRAIN_MODEL_NAME            = f"yolov4_{TRAINING_DATASET_TYPE.lower()}_transfer"
    
    PREDICTION_WEIGHT_FILE      = TRAIN_CHECKPOINTS_FOLDER + TRAIN_MODEL_NAME




#Evaluation settings
TEST_SCORE_THRESHOLD            = 0.05
TEST_IOU_THRESHOLD              = 0.5
USE_CUSTOM_WEIGHTS              = True

if MAKE_EVALUATION:
    if EVALUATION_DATASET_TYPE == "COCO":
        RELATIVE_PATH               = "./model_data/"  
        PREFIX_PATH                 = '.\YOLOv4-for-studying/dataset'
        YOLO_CLASS_PATH             = YOLO_COCO_CLASS_PATH
        TEST_ANNOTATION_PATH        = "YOLOv4-for-studying/dataset/coco/val2017.txt"
        EVALUATION_WEIGHT_FILE      = YOLO_V4_COCO_WEIGHTS

        VALIDATE_GT_RESULTS_DIR     = 'YOLOv4-for-studying/mAP/ground-truth-coco'
        VALIDATE_MAP_RESULT_PATH    = "YOLOv4-for-studying/mAP/results_coco.txt"

    elif EVALUATION_DATASET_TYPE == "LG":
        RELATIVE_PATH               = 'E:/dataset/TOTAL/'
        PREFIX_PATH                 = '.\YOLOv4-for-studying/dataset\LG_DATASET' 
        YOLO_CLASS_PATH             = "YOLOv4-for-studying/dataset/LG_DATASET/lg_class_names.txt"
        TEST_ANNOTATION_PATH        = "YOLOv4-for-studying/dataset/LG_DATASET/test.txt"  
        if EVALUATE_TRANSFER:
            EVALUATION_WEIGHT_FILE  = f"YOLOv4-for-studying/checkpoints/{EVALUATION_DATASET_TYPE.lower()}_dataset_transfer_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/yolov4_{EVALUATION_DATASET_TYPE.lower()}_transfer"
        else:
            EVALUATION_WEIGHT_FILE  = f"YOLOv4-for-studying/checkpoints/{EVALUATION_DATASET_TYPE.lower()}_dataset_from_scratch_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/yolov4_{EVALUATION_DATASET_TYPE.lower()}_from_scratch"
        
        VALIDATE_GT_RESULTS_DIR     = 'YOLOv4-for-studying/mAP/ground-truth-lg'
        VALIDATE_MAP_RESULT_PATH    = "YOLOv4-for-studying/mAP/results-lg.txt"

    elif EVALUATION_DATASET_TYPE == "VISDRONE":
        RELATIVE_PATH               = ""
        PREFIX_PATH                 = ""
        YOLO_CLASS_PATH             = "YOLOv4-for-studying/dataset/Visdrone_DATASET/visdrone_class_names.txt"
        TEST_ANNOTATION_PATH        = "YOLOv4-for-studying/dataset/Visdrone_DATASET/test.txt"
        if EVALUATE_TRANSFER:
            EVALUATION_WEIGHT_FILE  = f"YOLOv4-for-studying/checkpoints/{EVALUATION_DATASET_TYPE.lower()}_dataset_transfer_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/yolov4_{EVALUATION_DATASET_TYPE.lower()}_transfer"
        else:
            EVALUATION_WEIGHT_FILE  = f"YOLOv4-for-studying/checkpoints/{EVALUATION_DATASET_TYPE.lower()}_dataset_from_scratch_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/yolov4_{EVALUATION_DATASET_TYPE.lower()}_from_scratch"

        VALIDATE_GT_RESULTS_DIR     = 'YOLOv4-for-studying/mAP/ground-truth-visdrone'
        VALIDATE_MAP_RESULT_PATH    = "YOLOv4-for-studying/mAP/results-visdrone.txt"    

# EVALUATION_WEIGHT_FILE = "YOLOv4-for-studying/yolov4_992x992_origin/YOLOv4_custom"
# TEST_ANNOTATION_PATH        = "YOLOv4-for-studying/dataset/LG_DATASET/test.txt"  
# PREDICTION_WEIGHT_FILE      = "YOLOv4-for-studying/yolov4_992x992_origin/YOLOv4_custom"