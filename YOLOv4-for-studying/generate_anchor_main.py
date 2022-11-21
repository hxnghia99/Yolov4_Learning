
import numpy as np
from generate_anchor_func import kmeans, avg_iou
import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


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
    line = annotation.split()
    image_path = line[0]
    if not os.path.exists(image_path):
        raise KeyError("%s does not exist ... " %image_path)
    image = np.array(cv2.imread(image_path))
    bboxes = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]])
    return image, bboxes

#Extract ground truth bboxes for all data
def load_bbox(annotation, WIDTH, HEIGHT):
    save_bbox = []
    flag = False
    for idx in range(len(annotation)) :
        image, bboxes = parse_annotations(annotation[idx])
        bboxes = image_preporcess(np.copy(image), [HEIGHT, WIDTH], np.copy(bboxes))
        xywh = bboxes[:,:4]
        xywh[:,[2,3]] = bboxes[:, [2,3]] - bboxes[:,[0,1]] + 1
        #assign 1 if w or h < 1
        xywh[:, 2]  = np.maximum(1, xywh[:, 2])
        xywh[:, 3]  = np.maximum(1, xywh[:, 3])
        if flag == False:
            save_bbox = xywh[:, 2:]
            flag = True
        else :
            save_bbox = np.concatenate((save_bbox,xywh[:, 2:]), axis = 0)
    return save_bbox


def IoU_Estimate(path):
    annot_path = path
    INPUT_SIZE = [416, 416]
    WIDTH = INPUT_SIZE[0]
    HEIGHT = INPUT_SIZE[1]
    annotation = load_annotations(annot_path)       #List of [ [image_path, bboxes text], ...]
    data = load_bbox(annotation, WIDTH, HEIGHT)     #List of all gt_bboxes including (width, height)
    cnt = 0

    out, cnt, arr = kmeans(data, cnt, k=9)
    #anchors = np.array([[10,13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90],
    #            [156, 198], [373, 326]])

    print("kmeans counter : {}".format(cnt))
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
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
    anchor_n = np.array(anchor_n)  
    print("\nSorted anchor:", anchor_n)


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
    path = "YOLOv4-for-studying/dataset/Visdrone_DATASET/train.txt"
    IoU_Estimate(path)