
import numpy as np
from generate_anchor_func import kmeans, avg_iou
import os
import sys
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from YOLOv4_config import *

#Get image and target size to resize bboxes
def image_preporcess(image, target_size, gt_boxes=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    
    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2

    gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
    gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
    return gt_boxes



#Get annotation path and return list of [[image path, bboxes txt],...]
def load_annotations(annot_path):
    with open(annot_path, 'r') as f:
        txt = f.readlines()
        annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
    return annotations

#Get 1 annotation and returm [image, [bboxes] ]
def parse_annotations(annotation) :
    text_by_line = annotation.split()
    bboxes_annotations = []
    #At each annotations, divide into [image_path, [list of bboxes] ]
    for text in text_by_line:
        if not text.replace(',','').replace('-1','').isnumeric():
            temp_path   = os.path.relpath(text, RELATIVE_PATH)
            temp_path   = os.path.join(PREFIX_PATH, temp_path)
            image_path  = temp_path.replace('\\','/')
        else:
            bboxes_annotations.append(text)
    image = np.array(cv2.imread(image_path))
    bboxes = np.array([list(map(lambda x: float(x), box.split(','))) for box in bboxes_annotations], np.float32)
    return image, bboxes

#Extract ground truth bboxes for all data
def load_bbox(annotation, WIDTH, HEIGHT):
    save_bbox = [[],[],[]]
    flag = False    #for initializing
    for idx in range(len(annotation)) :
        sys.stdout.write("\rLoad image: {}".format(idx))
        image, bboxes = parse_annotations(annotation[idx])
        bboxes = image_preporcess(np.copy(image), [HEIGHT, WIDTH], np.copy(bboxes))
        wh = bboxes[:, 2:4] - bboxes[:,0:2] + 1
        #assign 1 if w or h < 1
        wh[:, 0]  = np.maximum(1, wh[:, 0])
        wh[:, 1]  = np.maximum(1, wh[:, 1])
        size = np.multiply.reduce(np.array(wh), axis=-1)

        for i in range(3):
            if i==0:
                size_threshold = (WIDTH*32/640) * (HEIGHT*32/480)
                bbox_wh = wh[size<=size_threshold]
                save_bbox[i].extend(bbox_wh.tolist())
            elif i==1:
                size_threshold_1 = (WIDTH*32/640) * (HEIGHT*32/480)
                size_threshold_2 = (WIDTH*96/640) * (HEIGHT*96/480)
                bbox_wh = wh[np.logical_and(size>size_threshold_1, size<=size_threshold_2)]
                save_bbox[i].extend(bbox_wh.tolist())
            else:
                size_threshold = (WIDTH*96/640) * (HEIGHT*96/480)
                bbox_wh = wh[size>size_threshold]
                save_bbox[i].extend(bbox_wh.tolist())
    return save_bbox    #[3, num bbox, 2]


def IoU_Estimate(path, size):
    annot_path = path
    WIDTH = size[0]
    HEIGHT = size[1]
    annotation = load_annotations(annot_path)       #List of [ [image_path, bboxes text], ...]
    data = load_bbox(annotation, WIDTH, HEIGHT)     #List of all gt_bboxes including (width, height)
    cnt = 0

    accuracy = 0

    clusters = []
    for i in range(3):
        out, cnt, arr = kmeans(np.array(data[i], np.float32), cnt, k=3)
        accuracy += avg_iou(np.array(data[i], np.float32), out) * 100
        clusters.extend(out)

    out = np.array(clusters, np.float32)

    # print("\nkmeans counter : {}".format(cnt))
    print("Accuracy: {:.2f}%".format(accuracy/3))
    #for i in range(len(arr)):
    #    print("Accuracy: {:.2f}%".format(avg_iou(data, arr[i]) * 100))
    print("Boxes: {}".format(out))

    anchor =    out
    # print(anchor)

    anchor_area = np.multiply.reduce(anchor, axis=-1)
    # print(anchor_area)

    anchor_n = []
    while len(anchor):
        i = np.argmin(anchor_area)
        anchor_n.append(anchor[i])
        anchor = np.delete(anchor, i, axis=0)
        anchor_area = np.delete(anchor_area, i, axis=0)
    anchor_n = np.round(np.array(anchor_n),2)
    anchor_n2 = np.round(np.array(anchor_n),0)
    print("\nSorted anchor:", anchor_n)
    print("\nSorted anchor:", anchor_n2)

    data_temp = []
    for i in range(3):
        data_temp.extend(data[i])
    data = np.array(data_temp, np.float32)
    
    colors = ['blue', 'yellow', 'red', 'green', 'cyan', 'magenta', 'white', 'gray', 'brown']
    for i in range(len(colors)):
        plt.scatter(data[:,0],data[:,1],s=5,c='black',label='scale')
    for i in range(len(colors)):
        plt.scatter(out[i, 0], out[i,1], s=300, c=colors[i], label=colors[i])
        #plt.scatter(kmeans.cluster_centers_[i, 0], kmeans.cluster_centers_[i,1], s=300, c=colors[i], label=colors[i])
    plt.xlabel('width')
    plt.ylabel('height')
    plt.ylim([0, 150])
    plt.xlim([0, 200])  
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # path = "YOLOv4-for-studying/dataset/Visdrone_DATASET/train.txt"
    path = "YOLOv4-for-studying/dataset/LG_DATASET/train_lg_total.txt"
    size = [448, 256]
    IoU_Estimate(path, size)