#===============================================================#
#                                                               #
#   File name   : Test.py                                       #
#   Author      : hxnghia99                                     #
#   Created date: May 24th, 2022                                #
#   GitHub      : https://github.com/hxnghia99/Yolov4_Learning  #
#   Description : YOLOv4 image prediction testing               #
#                                                               #
#===============================================================#


##############################################################################################
"""
INFERENCE TIME
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from YOLOv4_utils import *
from YOLOv4_config import *

# IMAGE_PATH = "./YOLOv4-for-studying/IMAGES/kite.jpg"
IMAGE_PATH = "./YOLOv4-for-studying/IMAGES/lg_street.jpg"
USE_LG_WEIGHTS = False
if USE_LG_WEIGHTS:
    yolo = Load_YOLOv4_Model("LG_WEIGHTS")
    detect_image(yolo, IMAGE_PATH, show=True, save=False, CLASSES_PATH=YOLO_LG_CLASS_PATH)
else:
    yolo = Load_YOLOv4_Model("COCO_WEIGHTS")
    detect_image(yolo, IMAGE_PATH, show=True, save=False, CLASSES_PATH=YOLO_COCO_CLASS_PATH)
##############################################################################################

print('a')




# import os
# from YOLOv4_config import *
# from YOLOv4_utils import *
# # text_by_line = "E:/dataset/TOTAL/test\images\cyclist_223_171_53_30_11000_2020-10-27-15-17-46-000000.jpg 12,120,69,181,0 80,7,128,27,0 57,5,87,25,0 1069,208,1124,274,2"
# text_by_line = "E:/dataset/TOTAL/train\images\c1_2020-10-292020-10-29-12-16-28-000138.jpg 762,114,782,176,1 785,120,804,180,1 676,77,692,120,1 651,64,663,109,1 663,71,677,113,1 327,31,342,67,1 364,119,382,183,1 618,76,639,124,1 320,191,344,264,1 411,0,421,23,1 611,74,631,122,1 359,0,366,17,1 282,4,292,32,1 628,58,642,101,1 343,0,353,19,1 208,598,260,647,1 234,298,279,391,1 367,115,395,168,1 369,0,379,16,1 268,6,277,33,1 256,17,269,48,1 904,24,920,47,1 920,26,928,48,1 407,0,415,22,1"
# text = text_by_line.split()
# bboxes = []
# for t in text:
#     if not t.replace(',', '').isnumeric():
#         temp_path   = os.path.relpath(t, RELATIVE_PATH)
#         temp_path   = os.path.join(PREFIX_PATH, temp_path)
#         image_path  = temp_path.replace('\\','/')
#     else:
#         t = list(map(int, t.split(',')))
#         bboxes.append(t)
# bboxes = np.array(bboxes)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if len(gpus) > 0:
#     print(f'GPUs {gpus}')
#     try: tf.config.experimental.set_memory_growth(gpus[0], True)
#     except RuntimeError: pass
# yolo = Load_YOLOv4_Model("COCO_WEIGHTS")
# pred_image = detect_image(yolo, image_path, show=False, save=False, CLASS_FILE=YOLO_COCO_CLASS_PATH)

# image = cv2.imread(image_path)
# image = draw_bbox(image, bboxes, YOLO_COCO_CLASS_PATH)
# cv2.imshow('truth', image)
# cv2.imshow("prediction", pred_image)

# if cv2.waitKey() == 'q':
#     pass