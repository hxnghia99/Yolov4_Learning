import os
import cv2
import numpy as np
import sys

type = "train"
# type = "test"


file_path = f"YOLOv4-for-studying/dataset/LG_DATASET/{type}_lg_total.txt"
RELATIVE_PATH               = 'E:/dataset/TOTAL/'
PREFIX_PATH                 = '.\YOLOv4-for-studying/dataset\LG_DATASET'
reso_x2 = [448,256]
reso_x1 = [224,128]

lr_file_path = file_path.split(type)[0] + "LR/"
hr_file_path = file_path.split(type)[0] + "HR/"

def image_preprocess(image, target_size):
    target_size_w, target_size_h = target_size
    image_h, image_w, _ = image.shape   
    resize_ratio = min(target_size_w/image_w, target_size_h/image_h)                      #resize ratio of the larger coordinate into 416
    new_image_w, new_image_h = int(resize_ratio*image_w), int(resize_ratio*image_h)
    
    image_resized = cv2.resize(image, (new_image_w, new_image_h))                     #the original image is resized into 416 x smaller coordinate

    image_padded = np.full(shape=[target_size_h, target_size_w, 3], fill_value=128.0, dtype=np.float32)
    dw, dh = (target_size_w - new_image_w) // 2, (target_size_h - new_image_h) // 2
    image_padded[dh:new_image_h+dh, dw:new_image_w+dw] = image_resized                #pad the resized image into image_padded

    return np.array(image_padded, np.uint8)




with open(file_path, "r") as f1:
    data = f1.read().splitlines()
    data = [text_by_line.strip() for text_by_line in data if len(text_by_line.strip().split()[1:]) != 0]

    for i, text_by_line in enumerate(data):
        image_path = text_by_line.strip().split()[0]
        image_path  = image_path.replace('\\','/')
        image_name = image_path.split("/")[-1].split(".")[0]

        image_path   = os.path.relpath(image_path, RELATIVE_PATH)
        image_path   = os.path.join(PREFIX_PATH, image_path)
        original_image = cv2.imread(image_path)
        height, width, _ = original_image.shape


        image_x2 = image_preprocess(np.copy(original_image), (reso_x2))
        image_x1 = image_preprocess(np.copy(original_image), (reso_x1))

        lr_file_path_saving = lr_file_path + image_name + ".png"
        hr_file_path_saving = hr_file_path + image_name + ".png"

        cv2.imwrite(lr_file_path_saving, image_x1)
        cv2.imwrite(hr_file_path_saving, image_x2)

print("\nFinished!")