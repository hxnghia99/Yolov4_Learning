import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, ZeroPadding2D, MaxPool2D, Input, UpSampling2D
from tensorflow.keras.regularizers import L2
import cv2
import colorsys
import random
import shutil
import sys

from YOLOv4_config import *
from YOLOv4_utils import *
import math
import matplotlib.pyplot as plt


file_path = "YOLOv4-for-studying/dataset/LG_DATASET/train_5k.txt"


num_obj         = [[0, 0, 0],[0, 0, 0],[0, 0, 0]]     # vehicle as 0, pedestrian as 1, cyclist as 2 --- small as 0, medium as 1, large as 2
num_size_obj    = [[], [], []]                        # vehicle as 0, pedestrian as 1, cyclist as 2 


def check_ground_truth(list_gt, path=None):
    global num_obj
    for gt_bbox in list_gt:
        width, height   = gt_bbox[2]-gt_bbox[0]+1, gt_bbox[3]-gt_bbox[1]+1
        class_num       = int(gt_bbox[4])
        size            = width * height
        num_size_obj[class_num].append(size)
        width, height   = width*640/224, height*480/128
        if width*height <= 32**2:
            num_obj[class_num][0] += 1
        elif width*height > 32**2 and width*height<=96**2:
            num_obj[class_num][1] += 1
        else:
            num_obj[class_num][2] += 1
            # if gt_bbox[4] == 1:
            #     print(path)
        


with open(file_path, "r") as f1:
    data = f1.read().splitlines()
    data = [text_by_line.strip() for text_by_line in data if len(text_by_line.strip().split()[1:]) != 0]

    for i, text_by_line in enumerate(data):
        image_path = text_by_line.strip().split()[0]
        image_path   = os.path.relpath(image_path, RELATIVE_PATH)
        image_path   = os.path.join(PREFIX_PATH, image_path)
        image_path  = image_path.replace('\\','/')
        original_image = cv2.imread(image_path)
        height, width, _ = original_image.shape

        ground_truth = text_by_line.strip().split()[1:]
        ground_truth = np.array([list(map(int, x.split(","))) for x in ground_truth], np.float32)

        _, ground_truth = image_preprocess(original_image, [224, 128], np.array(ground_truth))

        check_ground_truth(ground_truth, image_path)

        sys.stdout.write(f'\rLoaded images : {i}')


print("\nVehicle      [small, medium, large] : ", num_obj[0])
print("Pedestrian   [small, medium, large] : ", num_obj[1])
print("Cyclist      [small, medium, large] : ", num_obj[2])

fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.hist(num_size_obj[0], bins=100)
ax1.set_title("Vehicle")
# ax1.set_xlim(-10, 1210)
# ax1.set_ylim(0, 500)
ax2.hist(num_size_obj[1], bins=100)
ax2.set_title("Pedestrian")
# ax2.set_xlim(-10, 1010)
# ax2.set_ylim(0, 500)
ax3.hist(num_size_obj[2], bins=100)
ax3.set_title("Cyclist")
# ax3.set_xlim(-10, 1010)
# ax3.set_ylim(0, 500)
plt.show()

print("a")
