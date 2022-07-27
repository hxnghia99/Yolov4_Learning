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
import numpy as np
PREFIX_PATH             = "./YOLOv4-for-studying/dataset/Visdrone_DATASET/"  

DATASET_TYPE = ["VisDrone2019-DET-train/", "VisDrone2019-DET-val/"]
TEXT_NAME    = ['train', 'validation']

for idx, _ in enumerate(DATASET_TYPE):
    IMAGES_FOLDER       = PREFIX_PATH + DATASET_TYPE[idx] + "images/"
    ANNOTATION_FOLDER   = PREFIX_PATH + DATASET_TYPE[idx] + "annotations/"
    TEXT_SAVE_PATH      = PREFIX_PATH + TEXT_NAME[idx] + '.txt'
    list_images = glob.glob(IMAGES_FOLDER + "*.jpg")
    with open(TEXT_SAVE_PATH, 'w') as f1:
        max = 0
        min = 1000
        cnt = 0
        for image_path in list_images:
            annotation_name = (image_path.split("/")[-1]).split(".")[0]                                     #get annotation name for each pair (image-annotation)
            with open(ANNOTATION_FOLDER + annotation_name + ".txt", 'r') as f2:    
                bbox_annotations = f2.read().splitlines()
            bbox_annotations = [bbox[:len(bbox)-1] if bbox[-1] == ',' else bbox for bbox in bbox_annotations]

            bbox_annotations = [list(map(int, x.split(","))) for x in bbox_annotations]                 #split each line of annotation file by "," and cast those values into int32
            bbox_annotations = [np.concatenate([np.array(bbox[:2]),
                                                np.add(bbox[0:1],bbox[2:3]) - 1,
                                                np.add(bbox[1:2],bbox[3:4]) - 1, 
                                                np.array(bbox[5:6]) - 1], axis=-1) for bbox in bbox_annotations]    #Select bbox with confidence score == 1 and store bbox coordinates with class
            if max < len(bbox_annotations):
                max = len(bbox_annotations)
                name_image = image_path
            if min > len(bbox_annotations):
                min = len(bbox_annotations)
                name_min_image = image_path
            if len(bbox_annotations) >= 100:
                cnt += 1
            
            if len(bbox_annotations) < 1000:
                bbox_annotations = [",".join(list(map(str, bbox))) for bbox in bbox_annotations]
                all_info_annotation = image_path + " " + " ".join(bbox_annotations)
                f1.write(all_info_annotation + "\n")
        print("\n ",max)
        print(name_image)
        print(min)
        print(name_min_image)
        print("Number of removed images = {} ~ {:.2f}% \n".format(cnt, cnt/len(list_images)*100))





# import glob
# import numpy as np
# PREFIX_PATH             = "./YOLOv4-for-studying/dataset/LG_DATASET/"  

# DATASET_TYPE = ["train/", "test/"]
# TEXT_NAME    = ['train2', 'test2']

# for idx, _ in enumerate(DATASET_TYPE):
#     IMAGES_FOLDER       = PREFIX_PATH + DATASET_TYPE[idx] + "images/"
#     ANNOTATION_FOLDER   = PREFIX_PATH + DATASET_TYPE[idx] + "annotations/"
#     TEXT_SAVE_PATH      = PREFIX_PATH + TEXT_NAME[idx] + '.txt'
#     list_images = glob.glob(IMAGES_FOLDER + "*.jpg")
#     with open(TEXT_SAVE_PATH, 'w') as f1:
#         max = 0
#         min = 1000
#         cnt = 0
#         name_image = ""
#         name_min_image = ""
#         for image_path in list_images:
#             temp = image_path.split("/")[-1].split("_")[0]
#             if temp !="":
#                 annotation_name = (image_path.split("/")[-1]).split(".")[0]                                     #get annotation name for each pair (image-annotation)
#                 with open(ANNOTATION_FOLDER + annotation_name + ".txt", 'r') as f2:    
#                     bbox_annotations = f2.read().splitlines()
#                 bbox_annotations = [bbox[:len(bbox)-1] if bbox[-1] == ',' else bbox for bbox in bbox_annotations]

#                 bbox_annotations = [list(map(int, x.split(","))) for x in bbox_annotations]                 #split each line of annotation file by "," and cast those values into int32
#                 bbox_annotations = [np.concatenate([np.array(bbox[:2]),
#                                                     np.add(bbox[0:1],bbox[2:3]) - 1,
#                                                     np.add(bbox[1:2],bbox[3:4]) - 1, 
#                                                     np.array(bbox[5:6]) - 1], axis=-1) for bbox in bbox_annotations]    #Select bbox with confidence score == 1 and store bbox coordinates with class
#                 if max < len(bbox_annotations):
#                     max = len(bbox_annotations)
#                     name_image = image_path
#                 if min > len(bbox_annotations):
#                     min = len(bbox_annotations)
#                     name_min_image = image_path
#                 if len(bbox_annotations) >= 100:
#                     cnt += 1
                
#                 if len(bbox_annotations) < 1000:
#                     bbox_annotations = [",".join(list(map(str, bbox))) for bbox in bbox_annotations]
#                     all_info_annotation = image_path + " " + " ".join(bbox_annotations)
#                     f1.write(all_info_annotation + "\n")
#         print("\n ",max)
#         print(name_image)
#         print(min)
#         print(name_min_image)
#         print("Number of removed images = {} ~ {:.2f}% \n".format(cnt, cnt/len(list_images)*100))





# import matplotlib.pyplot as plt

# PREFIX_PATH             = "./YOLOv4-for-studying/dataset/Visdrone_DATASET/"  

# DATASET_TYPE = ["VisDrone2019-DET-train/", "VisDrone2019-DET-val/"]
# TEXT_NAME    = ['train', 'validation']

# for idx, _ in enumerate(DATASET_TYPE):
#     IMAGES_FOLDER       = PREFIX_PATH + DATASET_TYPE[idx] + "images/"
#     ANNOTATION_FOLDER   = PREFIX_PATH + DATASET_TYPE[idx] + "annotations/"
#     TEXT_SAVE_PATH      = PREFIX_PATH + TEXT_NAME[idx] + '.txt'
#     list_images = glob.glob(IMAGES_FOLDER + "*.jpg")
    
    
#     max = 416*416 
#     max_num_bbox = []
#     max_coors = []
#     image_path_list = []

#     with open(TEXT_SAVE_PATH, 'w') as f1:
#         for image_path in list_images:
#             annotation_name = (image_path.split("/")[-1]).split(".")[0]                                     #get annotation name for each pair (image-annotation)
#             with open(ANNOTATION_FOLDER + annotation_name + ".txt", 'r') as f2:    
#                 bbox_annotations = f2.read().splitlines()
#             bbox_annotations = [bbox[:len(bbox)-1] if bbox[-1] == ',' else bbox for bbox in bbox_annotations]


            

#             bbox_annotations = np.array([list(map(int, x.split(",")))[2:4] for x in bbox_annotations])                 #split each line of annotation file by "," and cast those values into int32
#             bbox_area = np.multiply.reduce(bbox_annotations, axis=-1)
#             # bbox_dict = dict(zip(np.unique(bbox_area, return_counts=True)))
#             bbox_area_unique, count = np.unique(bbox_area, return_counts=True)
#             max_area = np.max(bbox_area)
#             max_bbox = bbox_annotations[np.argmax(bbox_area)]

#             if max < max_area:
#                 # max = max_area
#                 max_coors.append(max_bbox)
#                 max_num_bbox.append(len(bbox_annotations))
#                 image_path_list.append(image_path)
#     for i in range(len(max_coors)):
#         print(max_coors[i], " , ", max_num_bbox[i], ", ", image_path_list[i])
#     print(len(max_coors))

            
            

