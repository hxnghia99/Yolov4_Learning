#===============================================================#
#                                                               #
#   File name   : generate_visdrone_text.py                     #
#   Author      : hxnghia99                                     #
#   Created date: May 30th, 2022                                #
#   GitHub      : https://github.com/hxnghia99/Yolov4_Learning  #
#   Description : From visdrone dataset, generateing text files # 
#  containing annotations with the same format as LG_DATASET    #
#                                                               #
#===============================================================#



import glob
from matplotlib.cbook import boxplot_stats
import numpy as np
PREFIX_PATH             = "./YOLOv4-for-studying/Visdrone_DATASET/"  

DATASET_TYPE = ["VisDrone2019-DET-train/", "VisDrone2019-DET-val/", "VisDrone2019-DET-test-dev/"]
TEXT_NAME    = ['train', 'validation', 'test']

for idx, _ in enumerate(DATASET_TYPE):
    IMAGES_FOLDER       = PREFIX_PATH + DATASET_TYPE[idx] + "images/"
    ANNOTATION_FOLDER   = PREFIX_PATH + DATASET_TYPE[idx] + "annotations/"
    TEXT_SAVE_PATH      = PREFIX_PATH + TEXT_NAME[idx] + '.txt'
    list_images = glob.glob(IMAGES_FOLDER + "*.jpg")
    with open(TEXT_SAVE_PATH, 'w') as f1:
        for image_path in list_images:
            annotation_name = (image_path.split("/")[-1]).split(".")[0]                                     #get annotation name for each pair (image-annotation)
            with open(ANNOTATION_FOLDER + annotation_name + ".txt", 'r') as f2:    
                bbox_annotations = f2.read().splitlines()
            
            bbox_annotations = [bbox[:len(bbox)-1] if bbox[-1] == ',' else bbox for bbox in bbox_annotations]

            bbox_annotations = [list(map(int, x.split(","))) for x in bbox_annotations]                 #split each line of annotation file by "," and cast those values into int32
            bbox_annotations = [np.concatenate([bbox[:2],
                                                np.add(bbox[0:1],bbox[2:3]),
                                                np.add(bbox[1:2],bbox[3:4]), 
                                                bbox[5:6]], axis=-1) for bbox in bbox_annotations if bbox[4] == 1]    #Select bbox with confidence score == 1 and store bbox coordinates with class
            bbox_annotations = [",".join(list(map(str, bbox))) for bbox in bbox_annotations]
            all_info_annotation = image_path + " " + " ".join(bbox_annotations)
            f1.write(all_info_annotation + "\n")

