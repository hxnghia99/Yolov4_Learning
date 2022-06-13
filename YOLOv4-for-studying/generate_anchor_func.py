import numpy as np


def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y                                #areas of k intersection
    box_area = box[0] * box[1]                          #area of 1 box
    cluster_area = clusters[:, 0] * clusters[:, 1]      #areas of k clusters

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, cnt, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]               #number of bboxes

    distances = np.empty((rows, k))     #(num_bboxes, num_clusters)
    last_clusters = np.zeros((rows,))   #(num_bboxes, )

    np.random.seed(42)
    arr = []
    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]
    
    while True:
        #calculate distance from all bboxes to k clusters
        for row in range(rows):                             
            distances[row] = 1 - iou(boxes[row], clusters)  

        nearest_clusters = np.argmin(distances, axis=1) # index of nearest centeroid to each bbox, shape (num_bboxes, )

        if (last_clusters == nearest_clusters).all():
            break

        for idx in range(k):
            clusters[idx] = dist(boxes[nearest_clusters == idx], axis=0) #apply distance method for nearest bboxes 

        last_clusters = nearest_clusters
        #######################################
        cnt+=1
        arr.append(clusters)
    return clusters, cnt, arr
