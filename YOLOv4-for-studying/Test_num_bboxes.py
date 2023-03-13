from __future__ import annotations
from YOLOv4_config import *
from YOLOv4_dataset import *
import sys




data_set = Dataset("train")

num_bbox = [0,0,0]
num_bbox_4pixels = [0,0,0] 
num_bbox_9pixels = [0,0,0]
num_bbox_16pixels = [0,0,0]
num_bbox_25pixels = [0,0,0]


annotations = data_set.annotations

for i, (_, target) in enumerate(data_set):
    sys.stdout.write("\rLoaded image: {} --> {} <> {} <> {} <> {} <> {}".format(i, num_bbox, num_bbox_4pixels, num_bbox_9pixels, num_bbox_16pixels, num_bbox_25pixels))
    for i in range(3):
        for j in range(TRAIN_BATCH_SIZE):
            list_gt = target[i][1][j]
            flag = np.add.reduce(np.array(list_gt[:,2]*list_gt[:,3]>0, np.int32))
            flag_4 = np.add.reduce(np.array(list_gt[:,2]*list_gt[:,3]>=4, np.int32))
            flag_9 = np.add.reduce(np.array(list_gt[:,2]*list_gt[:,3]>=9, np.int32))
            flag_16 = np.add.reduce(np.array(list_gt[:,2]*list_gt[:,3]>=16, np.int32))
            flag_25 = np.add.reduce(np.array(list_gt[:,2]*list_gt[:,3]>=25, np.int32))
            if num_bbox[i] < flag:
                num_bbox[i] = flag
            if num_bbox_4pixels[i] < flag_4:
                num_bbox_4pixels[i] = flag_4
            if num_bbox_9pixels[i] < flag_9:
                num_bbox_9pixels[i] = flag_9
            if num_bbox_16pixels[i] < flag_16:
                num_bbox_16pixels[i] = flag_16
            if num_bbox_25pixels[i] < flag_25:
                num_bbox_25pixels[i] = flag_25
print("\n")
print(num_bbox, num_bbox_4pixels, num_bbox_9pixels, num_bbox_16pixels, num_bbox_25pixels)