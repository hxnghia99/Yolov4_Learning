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
import collections


class Dataset(object):
    def __init__(self, dataset_type, TRAIN_INPUT_SIZE=YOLO_INPUT_SIZE, TEST_INPUT_SIZE=YOLO_INPUT_SIZE, TESTING=None, TEST_LABEL_GT_PATH=None, EVAL_MODE=None):    #train and test data use only one size 416x416
        #settings of annotation path, input size, batch size
        self.annotation_path        = TRAIN_ANNOTATION_PATH if dataset_type == 'train' else VALID_ANNOTATION_PATH
        if EVAL_MODE:
            self.annotation_path    = TEST_ANNOTATION_PATH
        self.input_size             = TRAIN_INPUT_SIZE if dataset_type == 'train' else TEST_INPUT_SIZE
        if USE_SUPERVISION:
            self.input_size_x2 = np.array(self.input_size, dtype=np.int32) * 2
        self.batch_size             = TRAIN_BATCH_SIZE if dataset_type == 'train' else TEST_BATCH_SIZE
        self.data_aug               = TRAIN_DATA_AUG if dataset_type == 'train' else TEST_DATA_AUG
        
        self.test_label             = False
        if TEST_LABEL_GT_PATH != None:
            self.annotation_path    = TEST_LABEL_GT_PATH
            self.batch_size         = 1
            self.data_aug           = False
            self.test_label         = True
        
        #settings of classes
        self.class_names            = read_class_names(YOLO_CLASS_PATH)
        self.num_classes            = len(self.class_names)
        #settings of anchors in different scales
        self.strides                = np.array(YOLO_SCALE_OFFSET)
        self.anchors                = [YOLO_ANCHORS[i] / self.strides[:, np.newaxis, np.newaxis][i] for i in range(3)]
        self.num_anchors_per_gcell  = ANCHORS_PER_GRID_CELL
        if USE_5_ANCHORS_SMALL_SCALE:
            self.num_anchors_per_gcell_small = ANCHORS_PER_GRID_CELL_SMALL
        #settings of datasets
        self.annotations            = self.load_annotations()
        self.num_samples            = len(self.annotations)
        self.num_batchs             = int(np.ceil(self.num_samples / self.batch_size))
        self.batchs_count           = 0
        self.max_bbox_per_scale     = YOLO_MAX_BBOX_PER_SCALE
        #settings of output sizes, output levels
        self.num_output_levels      = len(self.strides)
        self.output_gcell_sizes_w   = np.array(self.input_size[0] // self.strides).astype(np.int32)   #number of gridcells each scale
        self.output_gcell_sizes_h   = np.array(self.input_size[1] // self.strides).astype(np.int32)

        #Testing
        self.testing = TESTING

        #Super-resolution input path
        self.sr_path = "YOLOv4-for-studying/dataset/LG_DATASET/SR/"

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
                if not text.replace(',','').replace('-1','').isnumeric():
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
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #Use super-resolution input image
            if USE_SUPER_RESOLUTION_INPUT:
                image_path = self.sr_path + image_path.split("/")[-1].split(".")[0] + '.npy'
                image_sr = np.load(image_path)
                # image_sr = cv2.cvtColor(image_sr, cv2.COLOR_BGR2RGB)
        #Transform and sort bboxes in ascending order of area
        bboxes = list(np.array([list(map(float, box.split(','))) for box in bboxes_annotations], np.float32))
        bboxes.sort(key=lambda x: (x[2]-x[0]+1)*(x[3]-x[1]+1))
        bboxes = np.array(bboxes, np.float32)
        
        #return raw image and bboxes
        if mAP:
            if USE_SUPER_RESOLUTION_INPUT:
                return image, bboxes, image_sr
            return image, bboxes
        
        """ DATA AUGMENTATION if needed """
        #bboxes [xy_min, xy_max]
        if self.data_aug:
            if not USE_SUPER_RESOLUTION_INPUT:
                image, bboxes, _ = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
                image, bboxes, _ = self.random_crop(np.copy(image), np.copy(bboxes))
                image, bboxes, _ = self.random_translate(np.copy(image), np.copy(bboxes))
            else:
                image, bboxes, image_sr = self.random_horizontal_flip(np.copy(image), np.copy(bboxes), np.copy(image_sr))
                image, bboxes, image_sr = self.random_crop(np.copy(image), np.copy(bboxes), np.copy(image_sr))
                image, bboxes, image_sr = self.random_translate(np.copy(image), np.copy(bboxes), np.copy(image_sr))

        if TRAINING_DATASET_TYPE == "VISDRONE":
            """
            VISDRONE ignored region and class "other" preprocessing
            """
            bbox_mask = np.logical_and(bboxes[:,4]>-0.5, bboxes[:,4]<9.5)
            for bbox in bboxes:
                if bbox[4] == -1:     #class 0 (-1 after transforming to coco format) : ignored region
                    x_tl, y_tl, x_br, y_br =list(map(int,bbox[:4]))
                    image[y_tl:y_br, x_tl:x_br] = 128.0 #make ignored region into gray
            bboxes = bboxes[bbox_mask]

        #image-x2 for teacher
        if USE_SUPERVISION:
            image_x2 = image_preprocess(np.copy(image), self.input_size_x2, sizex2_flag=True)       #flag to use BICUBIC Interpolation

        #image preprocessing
        image, bboxes = image_preprocess(np.copy(image), self.input_size, bboxes)
        if USE_SUPER_RESOLUTION_INPUT:
            image_sr = image_preprocess(np.copy(image_sr), self.input_size)


        if FILTER_GT_BBOX_SIZE:
            temp = []
            for bbox in bboxes:
                num_pixels = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if num_pixels >= MIN_NUM_PIXEL:
                    temp.append(bbox)
            bboxes = temp

        # image_test = draw_bbox(np.array(image*255.0, np.uint8), np.round(bboxes), YOLO_CLASS_PATH)
        # cv2.imshow("test", cv2.resize(image_test, (960,540)))
        # if cv2.waitKey() == 'q':
        #     pass
        # print("A")

        if USE_SUPERVISION:
            return image, bboxes, image_x2
        elif USE_SUPER_RESOLUTION_INPUT:
            return image_sr, bboxes
        return image, bboxes

    #Find the best anchors for each bbox at each scale
    def preprocess_true_bboxes(self, bboxes):       #bbox: [xy_min, xy_max]
        #create label from true bboxes
        #shape [3, gcell, gcell, anchors, 5 + num_classes]
        label = [np.zeros((self.output_gcell_sizes_h[i], self.output_gcell_sizes_w[i], self.num_anchors_per_gcell_small if (i==0 and USE_5_ANCHORS_SMALL_SCALE) else self.num_anchors_per_gcell, 5 + self.num_classes), dtype=np.float32)
                            for i in range(self.num_output_levels)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale[0], 4), dtype=np.float32), np.zeros((self.max_bbox_per_scale[1], 4), dtype=np.float32), np.zeros((self.max_bbox_per_scale[2], 4), dtype=np.float32)]
        bboxes_idx = np.zeros((self.num_output_levels,), dtype=np.int32)  
        if USE_5_ANCHORS_SMALL_SCALE:
            label_flag =  [np.ones((self.output_gcell_sizes_h[i], self.output_gcell_sizes_w[i], self.num_anchors_per_gcell_small if (i==0 and USE_5_ANCHORS_SMALL_SCALE) else self.num_anchors_per_gcell), dtype=np.bool_)
                            for i in range(self.num_output_levels)]  
        #For each bbox, find the good anchors
        for bbox in bboxes:
            bbox_coordinates = bbox[:4]
            bbox_class_idx   = int(bbox[4])
            #class label smoothing
            onehot = np.zeros(self.num_classes, dtype=np.float32)
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
                anchor_xywh = np.zeros((self.num_anchors_per_gcell_small if (i==0 and USE_5_ANCHORS_SMALL_SCALE) else self.num_anchors_per_gcell, 4))                             #shape [3, 4]      
                anchor_xywh[:3, :2] = np.floor(bbox_xywh_scaled[i, :2]).astype(np.int32) + 0.5       #xy of anchor is center of gridcell that has xy of bbox
                if i==0 and USE_5_ANCHORS_SMALL_SCALE:
                    anchor_xywh[3:4,:2] = np.floor(bbox_xywh_scaled[i, :2]).astype(np.int32) + 1/6
                    anchor_xywh[4:5,:2] = np.floor(bbox_xywh_scaled[i, :2]).astype(np.int32) + 5/6
                anchor_xywh[:, 2:] = self.anchors[i]
                #compare the iou between 3 anchors and 1 bbox within specific scale
                iou_scores = bboxes_iou_from_xywh_np(bbox_xywh_scaled[i][np.newaxis, :], anchor_xywh)  #shape [3,]
                column, row = np.floor(bbox_xywh_scaled[i, :2]).astype(np.int32)
                if USE_5_ANCHORS_SMALL_SCALE:
                    iou_scores = iou_scores * np.array(label_flag[i][row,column,:], np.float32)
                all_iou_scores.append(iou_scores)                                                   #use this ious if do not have positive anchor
                iou_mask = iou_scores > ANCHOR_SELECTION_IOU_THRESHOLD
                # if i == 0:
                #     test = np.floor(bbox_xywh_scaled[i, :2]).astype(np.int32)
                #     print(test)
                #     print("------")
                if np.any(iou_mask):
                    label[i][row, column, iou_mask, :]  = 0             
                    label[i][row, column, iou_mask, :4] = bbox_xywh     #coordinates
                    label[i][row, column, iou_mask, 4]  = 1.0           #confidence score
                    label[i][row, column, iou_mask, 5:] = smooth_onehot #class probabilities
                    has_positive_flag = True
                    best_idx = np.argmax(iou_scores)
                    if USE_5_ANCHORS_SMALL_SCALE:
                        label_flag[i][row,column,best_idx] = False
                    #store true bboxes at scale i
                    if TRAINING_DATASET_TYPE == "VISDRONE":
                        bboxes_id = int(bboxes_idx[i] % self.max_bbox_per_scale[i])
                        bboxes_xywh[i][bboxes_id, :4]   = bbox_xywh
                    else:
                        bboxes_xywh[i][bboxes_idx[i], :4]   = bbox_xywh
                    bboxes_idx[i] += 1
            #If not have positive anchor, select one with the best iou 
            if not has_positive_flag:
                best_anchor_in_all_idx = np.argmax(np.concatenate(all_iou_scores, axis=-1), axis=-1)
                if best_anchor_in_all_idx < (self.num_anchors_per_gcell_small if USE_5_ANCHORS_SMALL_SCALE else self.num_anchors_per_gcell):
                    best_scale_idx = 0
                    best_anchor_idx = best_anchor_in_all_idx
                else:
                    best_anchor_in_all_idx = best_anchor_in_all_idx - (self.num_anchors_per_gcell_small if USE_5_ANCHORS_SMALL_SCALE else self.num_anchors_per_gcell)
                    best_scale_idx  = int(best_anchor_in_all_idx // self.num_anchors_per_gcell) + 1
                    best_anchor_idx = int(best_anchor_in_all_idx % self.num_anchors_per_gcell)
                column, row = np.floor(bbox_xywh_scaled[best_scale_idx, :2]).astype(np.int32)
                #assign the anchor with best iou to the label
                label[best_scale_idx][row, column, best_anchor_idx, :]  = 0
                label[best_scale_idx][row, column, best_anchor_idx, :4] = bbox_xywh
                label[best_scale_idx][row, column, best_anchor_idx, 4]  = 1.0
                label[best_scale_idx][row, column, best_anchor_idx, 5:] = smooth_onehot
                #store true bbox corresponding to the above label bbox
                if TRAINING_DATASET_TYPE == "VISDRONE":
                    bboxes_id = int(bboxes_idx[best_scale_idx] % self.max_bbox_per_scale[best_scale_idx])
                    bboxes_xywh[best_scale_idx][bboxes_id, :4] = bbox_xywh
                else:
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
            if USE_SUPERVISION:
                batch_image_x2 = np.zeros((self.batch_size, self.input_size_x2[1], self.input_size_x2[0], 3), dtype=np.float32)
            batch_label_sbboxes = np.zeros((self.batch_size, self.output_gcell_sizes_h[0], self.output_gcell_sizes_w[0],
                                            self.num_anchors_per_gcell_small if USE_5_ANCHORS_SMALL_SCALE else self.num_anchors_per_gcell, 5 + self.num_classes), dtype=np.float32)
            batch_label_mbboxes = np.zeros((self.batch_size, self.output_gcell_sizes_h[1], self.output_gcell_sizes_w[1],
                                            self.num_anchors_per_gcell, 5 + self.num_classes), dtype=np.float32)
            batch_label_lbboxes = np.zeros((self.batch_size, self.output_gcell_sizes_h[2], self.output_gcell_sizes_w[2],
                                            self.num_anchors_per_gcell, 5 + self.num_classes), dtype=np.float32)
            batch_sbboxes       = np.zeros((self.batch_size, self.max_bbox_per_scale[0], 4), dtype=np.float32)
            batch_mbboxes       = np.zeros((self.batch_size, self.max_bbox_per_scale[1], 4), dtype=np.float32)
            batch_lbboxes       = np.zeros((self.batch_size, self.max_bbox_per_scale[2], 4), dtype=np.float32)
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
                    if USE_SUPERVISION:
                        image, bboxes, image_x2 = self.parse_annotation(annotation)
                    else:
                        image, bboxes = self.parse_annotation(annotation)
                    
                    label_sbboxes, label_mbboxes, label_lbboxes, sbboxes, mbboxes, lbboxes = self.preprocess_true_bboxes(bboxes)   
                    #shape [output size, output size, 3, 85]
                    #Add image, labels to batchs
                    batch_image[num_annotations,:,:,:] = image
                    if USE_SUPERVISION:
                        batch_image_x2[num_annotations,:,:,:] = image_x2

                    batch_label_sbboxes[num_annotations,:,:,:,:]    = label_sbboxes         #shape [batch, output size, output size, 3, 85]
                    batch_label_mbboxes[num_annotations,:,:,:,:]    = label_mbboxes
                    batch_label_lbboxes[num_annotations,:,:,:,:]    = label_lbboxes 
                    batch_sbboxes[num_annotations,:,:]              = sbboxes
                    batch_mbboxes[num_annotations,:,:]              = mbboxes
                    batch_lbboxes[num_annotations,:,:]              = lbboxes
                    #end while -> increase num_annotations by 1
                    num_annotations += 1

                    # #Testing
                    # if self.testing:
                    #     image = draw_bbox(image, bboxes, YOLO_CLASS_PATH, show_label=False)
                    #     cv2.imshow("Testing", image)
                    #     if cv2.waitKey() == 'q':
                    #         pass
                    #     cv2.destroyAllWindows()
                    #     continue

                #end if -> increase batchs_count by 1
                self.batchs_count += 1
                #concatenate output label
                batch_small_target = batch_label_sbboxes, batch_sbboxes
                batch_medium_target  = batch_label_mbboxes, batch_mbboxes
                batch_large_target  = batch_label_lbboxes, batch_lbboxes
                if USE_SUPERVISION:
                    return (batch_image, batch_image_x2) , (batch_small_target, batch_medium_target, batch_large_target)
                else:
                    return batch_image, (batch_small_target, batch_medium_target, batch_large_target)
            else:
                self.batchs_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    #Data augmentation with 3 methods
    def random_horizontal_flip(self, image, bboxes, image_sr=None):
        image = np.array(image)
        bboxes = np.array(bboxes)
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]         #change xmin, xmax after flip
            if image_sr==None:
                pass
            else:
                image_sr = image_sr[:, ::-1, :]
        return image, bboxes, image_sr
        
    def random_crop(self, image, bboxes, image_sr=None):
        image = np.array(image)
        bboxes = np.array(bboxes)
        if random.random() < 0.5:
            h, w, _ = image.shape
            #largest bbox covering all gt_bboxes: [xy_min, xy_max]
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            #max translation: left, up, right, down
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]
            #crop part
            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))
            #cropped image
            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]
            #new bbox coordinates
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
            #apply to SR-image
            if image_sr==None:
                pass
            else:
                h_sr, w_sr,_ = image_sr.shape
                crop_xmin_sr = round(crop_xmin*w_sr/w)
                crop_ymin_sr = round(crop_ymin*h_sr/h)
                crop_xmax_sr = round(crop_xmax*w_sr/w)
                crop_ymax_sr = round(crop_ymax*h_sr/h)
                image_sr = image_sr[crop_ymin_sr : crop_ymax_sr, crop_xmin_sr : crop_xmax_sr]
        return image, bboxes, image_sr
    
    def random_translate(self, image, bboxes, image_sr=None):
        image = np.array(image)
        bboxes = np.array(bboxes)
        if random.random() < 0.5:
            h, w, _ = image.shape
            #largest bbox covering all gt_bboxes: [xy_min, xy_max]
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            #max translation: left, up, right, down
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]
            #max translation according to x-axis, y-axis
            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))
            #code
            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))
            #bbox translation
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
            #apply to SR-image
            if image_sr==None:
                pass
            else:
                h_sr, w_sr,_ = image_sr.shape
                tx_sr = tx*w_sr/w
                ty_sr = ty*h_sr/h
                M_sr = np.array([[1, 0, tx_sr], [0, 1, ty_sr]])
                image_sr = cv2.warpAffine(image_sr, M_sr, (w_sr, h_sr))
        return image, bboxes, image_sr


    #Function to test when reading annotation
    def test(self):
        self.data_aug = False
        if not USE_SUPERVISION:
            image, bboxes = self.parse_annotation(['./YOLOv4-for-studying/dataset/LG_DATASET/train/images/c1_2020-10-292020-10-29-12-15-58-000137.jpg',['1087,343,1143,438,1','421,260,461,354,1','640,149,672,216,1','1034,151,1062,202,1','505,88,522,144,1','749,103,767,159,1','385,105,406,156,1','579,97,595,146,1','411,98,433,154,1','471,84,494,139,1','286,75,307,121,1','408,29,425,64,1','579,48,589,85,1','425,93,444,149,1','619,37,633,77,1','907,27,916,59,1','259,77,278,120,1','681,89,703,145,1','666,103,683,160,1','400,0,414,23,1','710,100,723,156,1','352,0,365,15,1','599,1,607,30,1','513,2,522,27,1','276,65,292,112,1','405,32,422,65,1','714,107,732,160,1','713,114,726,174,1','724,121,738,177,1','734,135,749,185,1','747,0,757,27,1','276,2,286,31,1','284,0,291,21,1','291,0,300,28,1','316,0,327,22,1','331,0,338,17,1','259,63,271,106,1']])
        else:
            image, bboxes, image_x2 = self.parse_annotation(['./YOLOv4-for-studying/dataset/LG_DATASET/train/images/c1_2020-10-292020-10-29-12-15-58-000137.jpg',['1087,343,1143,438,1','421,260,461,354,1','640,149,672,216,1','1034,151,1062,202,1','505,88,522,144,1','749,103,767,159,1','385,105,406,156,1','579,97,595,146,1','411,98,433,154,1','471,84,494,139,1','286,75,307,121,1','408,29,425,64,1','579,48,589,85,1','425,93,444,149,1','619,37,633,77,1','907,27,916,59,1','259,77,278,120,1','681,89,703,145,1','666,103,683,160,1','400,0,414,23,1','710,100,723,156,1','352,0,365,15,1','599,1,607,30,1','513,2,522,27,1','276,65,292,112,1','405,32,422,65,1','714,107,732,160,1','713,114,726,174,1','724,121,738,177,1','734,135,749,185,1','747,0,757,27,1','276,2,286,31,1','284,0,291,21,1','291,0,300,28,1','316,0,327,22,1','331,0,338,17,1','259,63,271,106,1']])
            # image_x2 = cv2.cvtColor(np.array(image_x2, np.float32), cv2.COLOR_BGR2RGB)
        # image = cv2.cvtColor(np.array(image, np.float32), cv2.COLOR_BGR2RGB)
        image_test = draw_bbox(np.copy(image), np.copy(bboxes), CLASSES_PATH=YOLO_CLASS_PATH, show_label=False)
        
        label_sbboxes, label_mbboxes, label_lbboxes, sbboxes, mbboxes, lbboxes = self.preprocess_true_bboxes(bboxes)
        bbox_test = np.concatenate([sbboxes[:,:2] - sbboxes[:,2:]*0.5, sbboxes[:,:2]+sbboxes[:,2:]*0.5], axis=-1)
        bbox_test = np.concatenate([bbox_test, np.ones((YOLO_MAX_BBOX_PER_SCALE,1))], axis=-1)

        image_test2 = draw_bbox(np.copy(image), np.copy(bbox_test), CLASSES_PATH=YOLO_CLASS_PATH, show_label=False)
        cv2.imshow("Test label for small bboxes", cv2.resize(image_test2, (1280, 720)))
        cv2.imshow("Test after parse_annotation()", cv2.resize(image_test, (1280, 720)))
        if USE_SUPERVISION:
            cv2.imshow("Test image x2", cv2.resize(image_x2, (1280, 720)))
        if cv2.waitKey() == "q":
            pass
        cv2.destroyAllWindows()
        print("Test")


    def convert_into_original_size(self, array_bboxes, original_size):
        input_size                  = self.input_size
        org_image_h, org_image_w    = original_size
        if len(array_bboxes)!=0:
            pred_xywh = array_bboxes[:,0:4]
            # (x, y, w, h) --> (xmin, ymin, xmax, ymax) : size 416 x 416
            pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                        pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
            # prediction (xmin, ymin, xmax, ymax) -> prediction (xmin_org, ymin_org, xmax_org, ymax_org)
            resize_ratio = min(input_size[0] / org_image_w, input_size[1] / org_image_h)
            dw = (input_size[0] - resize_ratio * org_image_w) / 2                      #pixel position recalculation
            dh = (input_size[1] - resize_ratio * org_image_h) / 2
            pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio     #(pixel_pos - dw)/resize_ratio
            pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio
            array_orig_bboxes = np.concatenate([pred_coor, array_bboxes[:,4:5]], axis=-1)
            return array_orig_bboxes
        else:
            return np.array([])

    def test_label_gt(self):
        annotation = self.annotations[0]
        original_image = cv2.imread(annotation[0])
        bboxes = np.array([list(map(int, box.split(','))) for box in annotation[1]], np.float32)
        
        if TRAINING_DATASET_TYPE=="VISDRONE":
            """
            VISDRONE ignored region and class "other" preprocessing
            """
            bbox_mask = np.logical_and(bboxes[:,4]>-0.5, bboxes[:,4]<9.5)
            for bbox in bboxes:
                if bbox[4] == -1:     #class 0 (-1 after transforming to coco format) : ignored region
                    x_tl, y_tl, x_br, y_br = list(map(int,bbox[:4]))
                    original_image[y_tl:y_br, x_tl:x_br] = 128.0 #make ignored region into gray
            bboxes = bboxes[bbox_mask]
        #resized image and bboxes
        try:
            image_data, resized_bboxes = self.parse_annotation(annotation)
        except:
            image_data, resized_bboxes, _ = self.parse_annotation(annotation)
        image_data = image_data[np.newaxis,...]
        #label based on resized image and bboxes
        resized_label_sbboxes, resized_label_mbboxes, resized_label_lbboxes, resized_sbboxes, resized_mbboxes, resized_lbboxes = self.preprocess_true_bboxes(np.copy(resized_bboxes))

        sbboxes = [np.concatenate([bbox, np.array([0], dtype=np.float32)]) for bbox in resized_sbboxes if np.prod(bbox)!=0]
        sbboxes = self.convert_into_original_size(np.array(sbboxes), np.shape(original_image)[0:2])
        mbboxes = [np.concatenate([bbox, np.array([0], dtype=np.float32)]) for bbox in resized_mbboxes if np.prod(bbox)!=0]
        mbboxes = self.convert_into_original_size(np.array(mbboxes), np.shape(original_image)[0:2])
        lbboxes = [np.concatenate([bbox, np.array([0], dtype=np.float32)]) for bbox in resized_lbboxes if np.prod(bbox)!=0]
        lbboxes = self.convert_into_original_size(np.array(lbboxes), np.shape(original_image)[0:2])
        
        label_sbboxes = [list(bbox) for bbox in np.reshape(resized_label_sbboxes, (-1,5+self.num_classes)) if np.prod(bbox[0:4])!=0]
        label_sbboxes = [np.concatenate([np.array(y, dtype=np.float32)[0:4], np.array([np.argmax(np.array(y)[5:8])], dtype=np.float32)]) for y in list(collections.OrderedDict((tuple(x), x) for x in label_sbboxes).values())]
        label_sbboxes = self.convert_into_original_size(np.array(label_sbboxes), np.shape(original_image)[0:2])
        label_mbboxes = [list(bbox) for bbox in np.reshape(resized_label_mbboxes, (-1,5+self.num_classes)) if np.prod(bbox[0:4])!=0]
        label_mbboxes = [np.concatenate([np.array(y, dtype=np.float32)[0:4], np.array([np.argmax(np.array(y)[5:8])], dtype=np.float32)]) for y in list(collections.OrderedDict((tuple(x), x) for x in label_mbboxes).values())]
        label_mbboxes = self.convert_into_original_size(np.array(label_mbboxes), np.shape(original_image)[0:2])
        label_lbboxes = [list(bbox) for bbox in np.reshape(resized_label_lbboxes, (-1,5+self.num_classes)) if np.prod(bbox[0:4])!=0]
        label_lbboxes = [np.concatenate([np.array(y, dtype=np.float32)[0:4], np.array([np.argmax(np.array(y)[5:8])], dtype=np.float32)]) for y in list(collections.OrderedDict((tuple(x), x) for x in label_lbboxes).values())]
        label_lbboxes = self.convert_into_original_size(np.array(label_lbboxes), np.shape(original_image)[0:2])

        return original_image, image_data, bboxes, [label_sbboxes, label_mbboxes, label_lbboxes, sbboxes, mbboxes, lbboxes], [resized_label_sbboxes, resized_label_mbboxes, resized_label_lbboxes, resized_sbboxes, resized_mbboxes, resized_lbboxes]


if __name__ == '__main__':
    train_dataset = Dataset('train', TESTING=True)
    test_way = 1
    i = 0
    if test_way:
        for image, bboxes in train_dataset:
            i += 1
            print(i)       
    else:
        train_dataset.test()
    


