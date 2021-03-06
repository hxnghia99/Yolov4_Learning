#===============================================================#
#                                                               #
#   File name   : YOLOv3_config.py                              #
#   Author      : hxnghia99                                     #
#   Created date: April 28th, 2022                              #
#   GitHub      : https://github.com/hxnghia99/Yolov4_Learning  #
#   Description : YOLOv3 configurations                         #
#                                                               #
#===============================================================#




""" ------ IMPORTANT SETTING ------ """
# ["COCO", "LG", "VISDRONE"]
TRAINING_DATASET_TYPE           = "LG"
TRAIN_TRANSFER                  = True

# ["COCO", "LG", "VISDRONE"]
MAKE_EVALUATION                 = True
EVALUATION_DATASET_TYPE         = "LG"
EVALUATE_TRANSFER               = TRAIN_TRANSFER
""" ---------------------------------"""



#Overall settings
YOLO_COCO_CLASS_PATH            = "YOLOv3-for-studying/dataset/coco/coco.names"
YOLO_V3_COCO_WEIGHTS            = "YOLOv3-for-studying/model_data/yolov3.weights"
USE_LOADED_WEIGHT               = True
YOLO_INPUT_SIZE                 = [416, 320]

#Dataset configurations
TRAIN_INPUT_SIZE                = YOLO_INPUT_SIZE
TRAIN_BATCH_SIZE                = 4
TRAIN_DATA_AUG                  = True
TEST_INPUT_SIZE                 = YOLO_INPUT_SIZE
TEST_BATCH_SIZE                 = 4
TEST_DATA_AUG                   = False

#Anchor box settings
YOLO_MAX_BBOX_PER_SCALE         = 1000
ANCHORS_PER_GRID_CELL           = 3
ANCHOR_SELECTION_IOU_THRESHOLD  = 0.3
YOLO_SCALE_OFFSET               = [8, 16, 32]
YOLO_ANCHORS                    =   [[[10,  13], [16,   30], [33,   23]],
                                    [[30,  61], [62,   45], [59,  119]],
                                    [[116, 90], [156, 198], [373, 326]]]

#Training settings
TRAIN_SAVE_BEST_ONLY            = True  # saves only best model according validation loss (True recommended)
TRAIN_SAVE_CHECKPOINT           = False # saves all best validated checkpoints in training process (may require a lot disk space) (False recommended)
TRAIN_LOAD_IMAGES_TO_RAM        = False

TRAIN_WARMUP_EPOCHS             = 2
TRAIN_EPOCHS                    = 50
TRAIN_LR_END                    = 1e-6
TRAIN_LR_INIT                   = 1e-4
YOLO_LOSS_IOU_THRESHOLD         = 0.5

LAMBDA_COORD                    = 5
LAMBDA_NOOBJ                    = 0.5


VALIDATE_SCORE_THRESHOLD        = 0.35
VALIDATE_IOU_THRESHOLD          = 0.5

# COCO DATASET has only evaluation dataset: only prediction
if TRAINING_DATASET_TYPE == "COCO":
    YOLO_CLASS_PATH             = YOLO_COCO_CLASS_PATH
    PREDICTION_WEIGHT_FILE      = YOLO_V3_COCO_WEIGHTS

# LG DATASET has trainset, validationset, testset
elif TRAINING_DATASET_TYPE == "LG":
    YOLO_CLASS_PATH              = "YOLOv3-for-studying/dataset/LG_DATASET/lg_class_names.txt"
    TRAIN_ANNOTATION_PATH       = "YOLOv3-for-studying/dataset/LG_DATASET/train_40samples.txt"
    TEST_ANNOTATION_PATH        = "YOLOv3-for-studying/dataset/LG_DATASET/test_10samples.txt"
    RELATIVE_PATH               = 'E:/dataset/TOTAL/'
    PREFIX_PATH                 = '.\YOLOv3-for-studying\dataset/LG_DATASET'
    
    if not TRAIN_TRANSFER:
        TRAIN_CHECKPOINTS_FOLDER    = f"YOLOv3-for-studying/checkpoints/{TRAINING_DATASET_TYPE.lower()}_dataset_from_scratch_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/"
        TRAIN_LOGDIR                = f"YOLOv3-for-studying/log/{TRAINING_DATASET_TYPE.lower()}_dataset_from_scratch_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/"
        TRAIN_MODEL_NAME            = f"yolov3_{TRAINING_DATASET_TYPE.lower()}_from_scratch"
    else:
        TRAIN_CHECKPOINTS_FOLDER    = f"YOLOv3-for-studying/checkpoints/{TRAINING_DATASET_TYPE.lower()}_dataset_transfer_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/"
        TRAIN_LOGDIR                = f"YOLOv3-for-studying/log/{TRAINING_DATASET_TYPE.lower()}_dataset_transfer_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/"
        TRAIN_MODEL_NAME            = f"yolov3_{TRAINING_DATASET_TYPE.lower()}_transfer"
    
    PREDICTION_WEIGHT_FILE      = TRAIN_CHECKPOINTS_FOLDER + TRAIN_MODEL_NAME
    

# VISDRONE DATASET has trainset, validationset, testset
elif TRAINING_DATASET_TYPE == "VISDRONE":
    YOLO_CLASS_PATH              = "YOLOv3-for-studying/dataset/Visdrone_DATASET/visdrone_class_names.txt"
    TRAIN_ANNOTATION_PATH       = "YOLOv3-for-studying/dataset/Visdrone_DATASET/train_20samples.txt"
    TEST_ANNOTATION_PATH        = "YOLOv3-for-studying/dataset/Visdrone_DATASET/test_10samples.txt"
    RELATIVE_PATH               = "./YOLOv4-for-studying/"
    PREFIX_PATH                 = "./YOLOv3-for-studying/dataset/"

    if not TRAIN_TRANSFER:
        TRAIN_CHECKPOINTS_FOLDER    = f"YOLOv3-for-studying/checkpoints/{TRAINING_DATASET_TYPE.lower()}_dataset_from_scratch_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/"
        TRAIN_LOGDIR                = f"YOLOv3-for-studying/log/{TRAINING_DATASET_TYPE.lower()}_dataset_from_scratch_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/"
        TRAIN_MODEL_NAME            = f"yolov3_{TRAINING_DATASET_TYPE.lower()}_from_scratch"
    else:
        TRAIN_CHECKPOINTS_FOLDER    = f"YOLOv3-for-studying/checkpoints/{TRAINING_DATASET_TYPE.lower()}_dataset_transfer_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/"
        TRAIN_LOGDIR                = f"YOLOv3-for-studying/log/{TRAINING_DATASET_TYPE.lower()}_dataset_transfer_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/"
        TRAIN_MODEL_NAME            = f"yolov3_{TRAINING_DATASET_TYPE.lower()}_transfer"
    
    PREDICTION_WEIGHT_FILE      = TRAIN_CHECKPOINTS_FOLDER + TRAIN_MODEL_NAME



#Evaluation settings
TEST_SCORE_THRESHOLD            = 0.05
TEST_IOU_THRESHOLD              = 0.5
USE_CUSTOM_WEIGHTS              = True

if MAKE_EVALUATION:
    if EVALUATION_DATASET_TYPE == "COCO":
        RELATIVE_PATH               = "./model_data/"  
        PREFIX_PATH                 = '.\YOLOv3-for-studying/dataset/'
        YOLO_CLASS_PATH              = YOLO_COCO_CLASS_PATH
        TEST_ANNOTATION_PATH        = "YOLOv3-for-studying/dataset/coco/val2017.txt"
        EVALUATION_WEIGHT_FILE      = YOLO_V3_COCO_WEIGHTS
        
        VALIDATE_GT_RESULTS_DIR     = 'YOLOv3-for-studying/mAP/ground-truth-coco'
        VALIDATE_MAP_RESULT_PATH    = "YOLOv3-for-studying/mAP/results_coco.txt"

    elif EVALUATION_DATASET_TYPE == "LG":
        RELATIVE_PATH               = 'E:/dataset/TOTAL/'
        PREFIX_PATH                 = '.\YOLOv3-for-studying\dataset/LG_DATASET' 
        YOLO_CLASS_PATH              = "YOLOv3-for-studying/dataset/LG_DATASET/lg_class_names.txt"
        TEST_ANNOTATION_PATH        = "YOLOv3-for-studying/dataset/LG_DATASET/test_10samples.txt"  
        if EVALUATE_TRANSFER:
            EVALUATION_WEIGHT_FILE  = f"YOLOv3-for-studying/checkpoints/{EVALUATION_DATASET_TYPE.lower()}_dataset_transfer_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/yolov3_{EVALUATION_DATASET_TYPE.lower()}_transfer"
        else:
            EVALUATION_WEIGHT_FILE  = f"YOLOv3-for-studying/checkpoints/{EVALUATION_DATASET_TYPE.lower()}_dataset_from_scratch_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/yolov3_{EVALUATION_DATASET_TYPE.lower()}_from_scratch"
        
        VALIDATE_GT_RESULTS_DIR     = 'YOLOv3-for-studying/mAP/ground-truth-lg'
        VALIDATE_MAP_RESULT_PATH    = "YOLOv3-for-studying/mAP/results-lg.txt"
        
    elif EVALUATION_DATASET_TYPE == "VISDRONE":
        RELATIVE_PATH               = "./YOLOv4-for-studying/"
        PREFIX_PATH                 = "./YOLOv3-for-studying/dataset/"
        YOLO_CLASS_PATH              = "YOLOv3-for-studying/dataset/Visdrone_DATASET/visdrone_class_names.txt"
        TEST_ANNOTATION_PATH        = "YOLOv3-for-studying/dataset/Visdrone_DATASET/validation.txt"
        if EVALUATE_TRANSFER:
            EVALUATION_WEIGHT_FILE  = f"YOLOv3-for-studying/checkpoints/{EVALUATION_DATASET_TYPE.lower()}_dataset_transfer_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/yolov3_{EVALUATION_DATASET_TYPE.lower()}_transfer"
        else:
            EVALUATION_WEIGHT_FILE  = f"YOLOv3-for-studying/checkpoints/{EVALUATION_DATASET_TYPE.lower()}_dataset_from_scratch_{YOLO_INPUT_SIZE[0]}x{YOLO_INPUT_SIZE[1]}/yolov3_{EVALUATION_DATASET_TYPE.lower()}_from_scratch"

        VALIDATE_GT_RESULTS_DIR     = 'YOLOv3-for-studying/mAP/ground-truth-visdrone'
        VALIDATE_MAP_RESULT_PATH    = "YOLOv3-for-studying/mAP/results-visdrone.txt"
    


# PREDICTION_WEIGHT_FILE      = "YOLOv3-for-studying/Best_weights/yolov3_custom"
