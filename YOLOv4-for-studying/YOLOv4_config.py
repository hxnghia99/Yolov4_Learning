#===============================================================#
#                                                               #
#   File name   : YOLOv4_model.py                               #
#   Author      : hxnghia99                                     #
#   Created date: May 24th, 2022                                #
#   GitHub      : https://github.com/hxnghia99/Yolov4_Learning  #
#   Description : YOLOv4 configuration parameters               #
#                                                               #
#===============================================================#

# TRAINING INFORMATION
#FTTv2.2 + distillation-only-score + distillate-paper-19-L1-only-FTT-P2


import numpy as np

""" ------ IMPORTANT SETTING ------ """
# ["COCO", "LG", "VISDRONE"]
TRAINING_DATASET_TYPE           = "LG"
TRAIN_TRANSFER                  = True
TRAIN_FROM_CHECKPOINT           = False
MODEL_BRANCH_TYPE               = ["P2", "P5m"]
USE_FTT_P2                      = True
USE_FTT_P3                      = False
USE_FTT_P4                      = False
USE_FTT_DEVELOPING_VERSION      = False
LAMDA_FMAP_LOSS                 = 4.0
USE_SUPERVISION                 = True                         #when True, use at least 1 FTT module --> create teacher model
TEACHER_DILATION                = False                         #teacher uses dilation convolution or not
TRAINING_SHARING_WEIGHTS        = False or TEACHER_DILATION     #teacher uses weights from student or fixed pretrained weights


"""
MODEL_BRANCH_TYPE = [largest layer to be head, stop layer of backbone]
    - original    =             P3n         |           P5n
    - HR_P5       =             P0          |           P5
    - HR_P4       =             P0          |           P4
    - HR_P3       =             P0          |           P3
    - HR_P5_P(-1) =             P(-1)       |           P5
    - HR_P5_P2    =             P2          |           P5m
"""


FTT_P2_LAYERS_RANGE             = np.arange(462, 495)
TEACHER_LAYERS_RANGE            = np.arange(462)
STUDENT_LAYERS_RANGE            = np.arange(495)

# ["COCO", "LG", "VISDRONE"]
MAKE_EVALUATION                 = False
EVALUATION_DATASET_TYPE         = TRAINING_DATASET_TYPE
EVALUATE_TRANSFER               = TRAIN_TRANSFER
EVALUATION_SIZE                 = ["small"]#, "medium"]
"""
Evaluation size for different-sized objects in image size 640x480:
      Value         Details
    - small     : < 32x32 pixels
    - medium    : 32x32 < ... < 96x96 pixels
    - large     : > 96x96 pixels
"""
""" ---------------------------------"""

#Important initial settings
EVALUATE_ORIGINAL_SIZE          = False
USE_PRIMARY_EVALUATION_METRIC   = True         #calculate mAP0.5:0.95

#Slicing patch techniques setting: Only for Visdrone dataset
USE_SLICING_PATCH_TECHNIQUE     = True
SLICED_IMAGE_SIZE               = [416, 416]
OVERLAP_RATIO                   = [0.2, 0.2]
MIN_AREA_RATIO                  = 0.2


TRAIN_BATCH_SIZE                = 12
TEST_BATCH_SIZE                 = 12

#overall settings
YOLO_COCO_CLASS_PATH            = "YOLOv4-for-studying/dataset/coco/coco.names"
YOLO_V4_COCO_WEIGHTS            = "YOLOv4-for-studying/model_data/yolov4.weights"
YOLO_INPUT_SIZE                 = [224, 128]
USE_LOADED_WEIGHT               = True

#Dataset configurations
TRAIN_INPUT_SIZE                = YOLO_INPUT_SIZE
TRAIN_DATA_AUG                  = True
TEST_INPUT_SIZE                 = YOLO_INPUT_SIZE
TEST_DATA_AUG                   = False

#Anchor box settings
YOLO_MAX_BBOX_PER_SCALE         = 64
ANCHORS_PER_GRID_CELL           = 3
ANCHOR_SELECTION_IOU_THRESHOLD  = 0.3

if MODEL_BRANCH_TYPE[0] == "P(-1)":
    YOLO_SCALE_OFFSET           = [0.5, 1, 2]
elif MODEL_BRANCH_TYPE[0] == "P0":
    YOLO_SCALE_OFFSET           = [1, 2, 4]
elif MODEL_BRANCH_TYPE[0] == "P3n":
    YOLO_SCALE_OFFSET           = [8, 16, 32]
elif MODEL_BRANCH_TYPE[0] == "P2":
    YOLO_SCALE_OFFSET           = [4, 8, 16]


# # COCO anchors
# YOLO_ANCHORS                    = [[[12,  16], [19,   36], [40,   28]],
#                                    [[36,  75], [76,   55], [72,  146]],
#                                    [[142,110], [192, 243], [459, 401]]]
# # Visdrone anchors 992x992
# YOLO_ANCHORS                    = [[[6, 8], [10, 17], [18, 12]],
#                                    [[16, 27], [31, 19], [27, 39]],
#                                    [[54, 30], [47, 57], [96, 77]]]
# # Visdrone anchors 992x640
# YOLO_ANCHORS                    = [[[6, 8], [16, 11], [10, 18]],            #224x128
#                                    [[17, 27], [28, 17], [28, 40]],
#                                    [[47, 27], [52, 52], [96, 75]]]
# # Visdrone anchors 544x352 only for sliced images
# YOLO_ANCHORS                    = [[[5, 8], [8, 18], [14, 12]],
#                                    [[16, 28], [29, 18], [27, 43]],
#                                    [[53, 29], [60, 60], [123, 97]]]
# Visdrone anchors 416x416 only for sliced images
YOLO_ANCHORS                    = np.array([[[7, 9], [10, 18], [21, 14]],
                                   [[16, 28], [38, 23], [25, 42]],
                                   [[69, 38], [45, 67], [111, 95]]]) / 2





#Training settings
TRAIN_SAVE_BEST_ONLY            = True  # saves only best model according validation loss (True recommended)
TRAIN_SAVE_CHECKPOINT           = False # saves all best validated checkpoints in training process (may require a lot disk space) (False recommended)
TRAIN_LOAD_IMAGES_TO_RAM        = False
TRAIN_WARMUP_EPOCHS             = 2
TRAIN_EPOCHS                    = 50
TRAIN_LR_END                    = 1e-6
if MODEL_BRANCH_TYPE[0] == "P(-1)":
    TRAIN_LR_INIT               = 2e-3
elif MODEL_BRANCH_TYPE[0] == "P0":
    TRAIN_LR_INIT               = 1e-3
elif MODEL_BRANCH_TYPE[0] == "P3n" or MODEL_BRANCH_TYPE[0] == "P2":
    TRAIN_LR_INIT               = 1e-4
YOLO_LOSS_IOU_THRESHOLD         = 0.5


VALIDATE_SCORE_THRESHOLD        = 0.35
VALIDATE_IOU_THRESHOLD          = 0.5


# COCO DATASET has only evaluation dataset
if TRAINING_DATASET_TYPE == "COCO":
    YOLO_CLASS_PATH             = YOLO_COCO_CLASS_PATH
    PREDICTION_WEIGHT_FILE      = YOLO_V4_COCO_WEIGHTS

# LG DATASET has trainset, validationset, testset
elif TRAINING_DATASET_TYPE == "LG":
    YOLO_CLASS_PATH             = "YOLOv4-for-studying/dataset/LG_DATASET/lg_class_names.txt"
    # TRAIN_ANNOTATION_PATH       = "YOLOv4-for-studying/dataset/LG_DATASET/test_100samples.txt"
    # TEST_ANNOTATION_PATH        = "YOLOv4-for-studying/dataset/LG_DATASET/test_100samples.txt"
    # TRAIN_ANNOTATION_PATH       = "YOLOv4-for-studying/dataset/LG_DATASET/train_lg_total.txt"
    # TEST_ANNOTATION_PATH        = "YOLOv4-for-studying/dataset/LG_DATASET/test_lg_total.txt"
    TRAIN_ANNOTATION_PATH       = "YOLOv4-for-studying/dataset/LG_DATASET/train_5k.txt"
    TEST_ANNOTATION_PATH        = "YOLOv4-for-studying/dataset/LG_DATASET/validate_700.txt"
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
        # TRAIN_ANNOTATION_PATH       = f"YOLOv4-for-studying/dataset/Visdrone_DATASET/train2_slice.txt"
        # TEST_ANNOTATION_PATH        = "YOLOv4-for-studying/dataset/Visdrone_DATASET/train2_slice.txt"
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
VISDRONE_IGNORED_THRESHOLD      = 0.58
VISDRONE_OTHER_THRESHOLD        = 0.5
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
        # TEST_ANNOTATION_PATH        = "YOLOv4-for-studying/dataset/LG_DATASET/test_lg_total.txt"  
        TEST_ANNOTATION_PATH        = "YOLOv4-for-studying/dataset/LG_DATASET/evaluate_1300.txt" 
        if EVALUATE_TRANSFER:
            EVALUATION_WEIGHT_FILE  = f"YOLOv4-for-studying/checkpoints/{EVALUATION_DATASET_TYPE.lower()}_dataset_transfer_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/yolov4_{EVALUATION_DATASET_TYPE.lower()}_transfer"
            # EVALUATION_WEIGHT_FILE  = f"YOLOv4-for-studying/checkpoints/checkpoints_original_subset_224x128/{EVALUATION_DATASET_TYPE.lower()}_dataset_transfer_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/yolov4_{EVALUATION_DATASET_TYPE.lower()}_transfer"
            # EVALUATION_WEIGHT_FILE  = f"YOLOv4-for-studying/checkpoints/checkpoints_HR_P5_subset_224x128/{EVALUATION_DATASET_TYPE.lower()}_dataset_transfer_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/yolov4_{EVALUATION_DATASET_TYPE.lower()}_transfer"
        else:
            EVALUATION_WEIGHT_FILE  = f"YOLOv4-for-studying/checkpoints/{EVALUATION_DATASET_TYPE.lower()}_dataset_from_scratch_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/yolov4_{EVALUATION_DATASET_TYPE.lower()}_from_scratch"
            # EVALUATION_WEIGHT_FILE  = f"YOLOv4-for-studying/checkpoints/checkpoints_HR_P3_subset_224x128/{EVALUATION_DATASET_TYPE.lower()}_dataset_from_scratch_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/yolov4_{EVALUATION_DATASET_TYPE.lower()}_from_scratch"
        
        VALIDATE_GT_RESULTS_DIR     = 'YOLOv4-for-studying/mAP/ground-truth-lg'
        VALIDATE_MAP_RESULT_PATH    = f"YOLOv4-for-studying/mAP/results-lg.txt"

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
        VALIDATE_MAP_RESULT_PATH    = f"YOLOv4-for-studying/mAP/results-visdrone.txt"    

# EVALUATION_WEIGHT_FILE = "YOLOv4-for-studying/checkpoints/lg_dataset_transfer_224x128_Original"
# TEST_ANNOTATION_PATH        = "YOLOv4-for-studying/dataset/LG_DATASET/test.txt"  
# PREDICTION_WEIGHT_FILE      = "YOLOv4-for-studying/yolov4_992x992_origin/YOLOv4_custom"