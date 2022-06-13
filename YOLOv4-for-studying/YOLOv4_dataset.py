#===============================================================#
#                                                               #
#   File name   : YOLOv4_dataset.py                             #
#   Author      : hxnghia99                                     #
#   Created date: May 24th, 2022                                #
#   GitHub      : https://github.com/hxnghia99/Yolov4_Learning  #
#   Description : YOLOv4 dataset preparation                    #
#                                                               #
#===============================================================#


import os
import cv2
import numpy as np
import tensorflow as tf
from YOLOv4_utils import *
from YOLOv4_config import *
import random



class Dataset(object):
    def __init__(self, dataset_type, TRAIN_INPUT_SIZE=YOLO_INPUT_SIZE, TEST_INPUT_SIZE=YOLO_INPUT_SIZE):    #train and test data use only one size 416x416
        #settings of annotation path, input size, batch size
        self.annotation_path        = TRAIN_ANNOTATION_PATH if dataset_type == 'train' else TEST_ANNOTATION_PATH
        self.input_size             = TRAIN_INPUT_SIZE if dataset_type == 'train' else TEST_INPUT_SIZE
        self.batch_size             = TRAIN_BATCH_SIZE if dataset_type == 'train' else TEST_BATCH_SIZE
        self.data_aug               = TRAIN_DATA_AUG if dataset_type == 'train' else TEST_DATA_AUG
        #settings of classes
        self.class_names            = read_class_names(YOLO_CLASS_PATH)
        self.num_classes            = len(self.class_names)
        #settings of anchors in different scales
        self.strides                = np.array(YOLO_SCALE_OFFSET)
        self.anchors                = (np.array(YOLO_ANCHORS)/ self.strides[:, np.newaxis, np.newaxis])
        self.num_anchors_per_gcell  = ANCHORS_PER_GRID_CELL
        #settings of datasets
        self.annotations            = self.load_annotations()
        self.num_samples            = len(self.annotations)
        self.num_batchs             = int(np.ceil(self.num_samples / self.batch_size))
        self.batchs_count           = 0
        self.max_bbox_per_scale     = YOLO_MAX_BBOX_PER_SCALE
        #settings of output sizes, output levels
        self.num_output_levels      = len(self.strides)
        self.output_gcell_sizes_w   = self.input_size[0] // self.strides   #number of gridcells each scale
        self.output_gcell_sizes_h   = self.input_size[1] // self.strides

    #special method to give number of batchs in dataset
    def __len__(self):
        return self.num_batchs

    #Get the list of annotations for each line
    def load_annotations(self):
        with open(self.annotation_path, 'r') as f:
            all_texts_by_line = f.read().splitlines()
            #Ensure that image has objects
            annotations = [text_by_line.strip() for text_by_line in all_texts_by_line if len(text_by_line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        #Go through each annotation to process
        final_annotations = []
        for annotation in annotations:
            text_by_line = annotation.split()
            bboxes_annotations = []
            #At each annotations, divide into [image_path, [list of bboxes] ]
            for text in text_by_line:
                if not text.replace(',','').isnumeric():
                    temp_path   = os.path.relpath(text, RELATIVE_PATH)
                    temp_path   = os.path.join(PREFIX_PATH, temp_path)
                    image_path  = temp_path.replace('\\','/')
                else:
                    bboxes_annotations.append(text)
            if not os.path.exists(image_path):
                raise KeyError("%s does not exit !" %image_path)
            if TRAIN_LOAD_IMAGES_TO_RAM:
                image = cv2.imread(image_path)
                final_annotations.append([image_path, bboxes_annotations, image])
            else:
                final_annotations.append([image_path, bboxes_annotations])
        return final_annotations                                        #shape [num_samples, 2], item includes image_path + [list of bboxes]

    #Receive annotation, preprocess image and produce image+bboxes in size 416x416
    def parse_annotation(self, annotation, mAP=False):
        if TRAIN_LOAD_IMAGES_TO_RAM:
            image, bboxes_annotations = annotation[2], annotation[1]
        else:
            #Get data inside annotation
            image_path, bboxes_annotations = annotation
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = np.array([list(map(int, box.split(','))) for box in bboxes_annotations])
        """
        DATA AUGMENTATION if needed
        """
        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        if mAP:
            return image, bboxes
        #preprocess, bboxes as (xmin, ymin, xmax, ymax)
        image, bboxes = image_preprocess(np.copy(image), self.input_size, np.copy(bboxes))
        
        """
        VISDRONE ignored region and class "other" preprocessing
        """
        bbox_mask = np.logical_and(bboxes[:,4]>0.0, bboxes[:,4]<11.0)
        for bbox in bboxes:
            if bbox[4] == 0:     #class 0 : ignored region
                x_tl, y_tl, x_br, y_br = bbox[:4]
                image[y_tl:y_br, x_tl:x_br] = 128/255.0 #make ignored region into gray
        bboxes = bboxes[bbox_mask]

        return image, bboxes
        
    #Find the best anchors for each bbox at each scale
    def preprocess_true_bboxes(self, bboxes):
        #create label from true bboxes
        #shape [3, gcell, gcell, anchors, 5 + num_classes]
        label = [np.zeros((self.output_gcell_sizes_h[i], self.output_gcell_sizes_w[i], self.num_anchors_per_gcell, 5 + self.num_classes), dtype=np.float)
                            for i in range(self.num_output_levels)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(self.num_output_levels)]
        bboxes_idx = np.zeros((self.num_output_levels,), dtype=np.int32)
        #For each bbox, find the good anchors
        for bbox in bboxes:
            bbox_coordinates = bbox[:4]
            bbox_class_idx   = bbox[4]
            #class label smoothing
            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_idx] = 1.0
            delta = 0.01
            smooth_onehot = onehot * (1 - delta) + delta / self.num_classes    # label*(1-delta) + delta/K
            #coordinate processing to generate bbox in 3 scales
            xy = (bbox_coordinates[:2] + bbox_coordinates[2:])*0.5
            wh = bbox_coordinates[2:] - bbox_coordinates[:2]
            bbox_xywh = np.concatenate([xy, wh], axis=-1)                       #shape [4,]
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]         #shape [1, 4] / shape [3, 1] = shape [3,4]
            #At each scale of bbox, select the good anchors
            all_iou_scores = []
            has_positive_flag = False
            for i in range(self.num_output_levels):
                anchor_xywh = np.zeros((self.num_anchors_per_gcell, 4))                             #shape [3, 4]      
                anchor_xywh[:, :2] = np.floor(bbox_xywh_scaled[i, :2]).astype(np.int32) + 0.5       #xy of anchor is center of gridcell that has xy of bbox
                anchor_xywh[:, 2:] = self.anchors[i]
                #compare the iou between 3 anchors and 1 bbox within specific scale
                iou_scores = bboxes_iou_from_xywh(bbox_xywh_scaled[i][np.newaxis, :], anchor_xywh)  #shape [3,]
                all_iou_scores.append(iou_scores)                                                   #use this ious if do not have positive anchor
                iou_mask = iou_scores > ANCHOR_SELECTION_IOU_THRESHOLD
                if np.any(iou_mask):
                    column, row = np.floor(bbox_xywh_scaled[i, :2]).astype(np.int32)
                    label[i][row, column, iou_mask, :]  = 0             
                    label[i][row, column, iou_mask, :4] = bbox_xywh     #coordinates
                    label[i][row, column, iou_mask, 4]  = 1.0           #confidence score
                    label[i][row, column, iou_mask, 5:] = smooth_onehot #class probabilities
                    has_positive_flag = True
                    #store true bboxes at scale i
                    bboxes_xywh[i][bboxes_idx[i], :4]   = bbox_xywh
                    bboxes_idx[i] += 1
            #If not have positive anchor, select one with the best iou 
            if not has_positive_flag:
                best_anchor_in_all_idx = np.argmax(np.array(all_iou_scores).reshape(-1), axis=-1)
                best_scale_idx  = int(best_anchor_in_all_idx // self.num_anchors_per_gcell)
                best_anchor_idx = int(best_anchor_in_all_idx % self.num_anchors_per_gcell)
                column, row = np.floor(bbox_xywh_scaled[best_scale_idx, :2]).astype(np.int32)
                #assign the anchor with best iou to the label
                label[best_scale_idx][row, column, best_anchor_idx, :]  = 0
                label[best_scale_idx][row, column, best_anchor_idx, :4] = bbox_xywh
                label[best_scale_idx][row, column, best_anchor_idx, 4]  = 1.0
                label[best_scale_idx][row, column, best_anchor_idx, 5:] = smooth_onehot
                #store true bbox corresponding to the above label bbox
                bboxes_xywh[best_scale_idx][bboxes_idx[best_scale_idx], :4] = bbox_xywh
                bboxes_idx[best_scale_idx] += 1
        #Get label at each scale
        label_sbboxes, label_mbboxes, label_lbboxes = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbboxes, label_mbboxes, label_lbboxes, sbboxes, mbboxes, lbboxes
    
    def __iter__(self):
        return self

    def __next__(self):
        with tf.device('/cpu:0'):
            #Generate initial variables for batch: image, label small+medium+large bboxes
            batch_image = np.zeros((self.batch_size, self.input_size[1], self.input_size[0], 3), dtype=np.float32)
            batch_label_sbboxes = np.zeros((self.batch_size, self.output_gcell_sizes_h[0], self.output_gcell_sizes_w[0],
                                            self.num_anchors_per_gcell, 5 + self.num_classes), dtype=np.float32)
            batch_label_mbboxes = np.zeros((self.batch_size, self.output_gcell_sizes_h[1], self.output_gcell_sizes_w[1],
                                            self.num_anchors_per_gcell, 5 + self.num_classes), dtype=np.float32)
            batch_label_lbboxes = np.zeros((self.batch_size, self.output_gcell_sizes_h[2], self.output_gcell_sizes_w[2],
                                            self.num_anchors_per_gcell, 5 + self.num_classes), dtype=np.float32)
            batch_sbboxes       = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_mbboxes       = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_lbboxes       = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            #Read annotations, then read image and label, finally store them
            num_annotations = 0
            if self.batchs_count < self.num_batchs:
                while(num_annotations < self.batch_size):
                    annotation_idx = self.batchs_count * self.batch_size + num_annotations
                    #There is a posibility that number_samples is not completely divided by batch_size
                    if annotation_idx >= self.num_samples:
                        annotation_idx -= self.num_samples
                    annotation = self.annotations[annotation_idx]
                    #Read image and bboxes from annotation, then extract labels of 3 scales
                    image, bboxes = self.parse_annotation(annotation)
                    label_sbboxes, label_mbboxes, label_lbboxes, sbboxes, mbboxes, lbboxes = self.preprocess_true_bboxes(bboxes)   
                    #shape [output size, output size, 3, 85]
                    #Add image, labels to batchs
                    batch_image[num_annotations,:,:,:] = image
                    batch_label_sbboxes[num_annotations,:,:,:,:]    = label_sbboxes         #shape [batch, output size, output size, 3, 85]
                    batch_label_mbboxes[num_annotations,:,:,:,:]    = label_mbboxes
                    batch_label_lbboxes[num_annotations,:,:,:,:]    = label_lbboxes 
                    batch_sbboxes[num_annotations,:,:]              = sbboxes
                    batch_mbboxes[num_annotations,:,:]              = mbboxes
                    batch_lbboxes[num_annotations,:,:]              = lbboxes
                    #end while -> increase num_annotations by 1
                    num_annotations += 1
                #end if -> increase batchs_count by 1
                self.batchs_count += 1
                #concatenate output label
                batch_small_target = batch_label_sbboxes, batch_sbboxes
                batch_medium_target  = batch_label_mbboxes, batch_mbboxes
                batch_large_target  = batch_label_lbboxes, batch_lbboxes
                return batch_image, (batch_small_target, batch_medium_target, batch_large_target)
            else:
                self.batchs_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    #Data augmentation with 3 methods
    def random_horizontal_flip(self, image, bboxes):
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]         #change xmin, xmax after flip
        return image, bboxes
    def random_crop(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
        return image, bboxes
    def random_translate(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
        return image, bboxes


    #Function to test when reading annotation
    def test(self):
        image, bboxes = self.parse_annotation(self.annotations[0])
        image = cv2.cvtColor(np.array(image, np.float32), cv2.COLOR_BGR2RGB)
        image_test = draw_bbox(np.copy(image), np.copy(bboxes), CLASSES_PATH=YOLO_CLASS_PATH, show_label=False)
        
        label_sbboxes, label_mbboxes, label_lbboxes, sbboxes, mbboxes, lbboxes = self.preprocess_true_bboxes(bboxes)
        bbox_test = np.concatenate([sbboxes[:,:2] - sbboxes[:,2:]*0.5, sbboxes[:,:2]+sbboxes[:,2:]*0.5], axis=-1)
        bbox_test = np.concatenate([bbox_test, np.ones((YOLO_MAX_BBOX_PER_SCALE,1))], axis=-1)

        image_test2 = draw_bbox(np.copy(image), np.copy(bbox_test), CLASSES_PATH=YOLO_CLASS_PATH, show_label=False)
        cv2.imshow("Test label", image_test2)
        cv2.imshow("Test", image_test)
        if cv2.waitKey() == "q":
            pass
        print("Test")
        

if __name__ == '__main__':
    train_dataset = Dataset('train')
    train_dataset.test() 