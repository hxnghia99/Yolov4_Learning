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
    def __init__(self, dataset_type, TRAIN_INPUT_SIZE=YOLO_INPUT_SIZE, TEST_INPUT_SIZE=YOLO_INPUT_SIZE, TESTING=None):    #train and test data use only one size 416x416
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

        #Testing
        self.testing = TESTING

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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = np.array([list(map(int, box.split(','))) for box in bboxes_annotations])
        
        #return raw image and bboxes
        if mAP:
            return image, bboxes

        """ DATA AUGMENTATION if needed """
        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        if TRAINING_DATASET_TYPE == "VISDRONE":
            """
            VISDRONE ignored region and class "other" preprocessing
            """
            bbox_mask = np.logical_and(bboxes[:,4]>-0.5, bboxes[:,4]<9.5)
            for bbox in bboxes:
                if bbox[4] == -1:     #class 0 (-1 after transforming to coco format) : ignored region
                    x_tl, y_tl, x_br, y_br = bbox[:4]
                    image[y_tl:y_br, x_tl:x_br] = 128.0 #make ignored region into gray
            bboxes = bboxes[bbox_mask]

        #preprocess, bboxes as (xmin, ymin, xmax, ymax)
        image, bboxes = image_preprocess(image, self.input_size, bboxes)
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
                return batch_image, (batch_small_target, batch_medium_target, batch_large_target)
            else:
                self.batchs_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    #Data augmentation with 3 methods
    def random_horizontal_flip(self, image, bboxes):
        image = np.array(image)
        bboxes = np.array(bboxes)
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]         #change xmin, xmax after flip
        return image, bboxes
    def random_crop(self, image, bboxes):
        image = np.array(image)
        bboxes = np.array(bboxes)
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
        image = np.array(image)
        bboxes = np.array(bboxes)
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
        self.data_aug = False
        image, bboxes = self.parse_annotation(['YOLOv4-for-studying/dataset/Visdrone_DATASET/VisDrone2019-DET-train/images/0000293_01001_d_0000927.jpg', ['393,133,503,173,-1', '500,154,626,172,-1', '459,184,492,224,-1','422,214,456,234,-1', '774,234,864,292,-1', '712,203,739,240,-1','698,192,716,214,-1', '798,269,868,313,-1', '356,202,388,223,-1','895,368,947,409,-1', '609,176,632,197,8', '659,224,673,235,3','615,309,643,333,3', '724,241,729,255,0', '746,234,780,257,-1','672,200,684,210,3', '677,207,689,217,3', '684,208,711,245,8','720,275,742,291,3', '741,283,770,309,3', '864,349,875,371,2','852,351,864,368,2', '804,355,850,389,3', '829,384,880,422,3','869,376,881,409,0', '874,383,881,414,0', '879,386,890,414,0','949,370,961,391,2', '1049,279,1060,299,0', '965,337,974,363,0','1076,297,1083,319,0', '1063,296,1071,318,0','1064,293,1072,315,0', '1034,318,1044,345,0','1018,318,1027,341,0', '995,316,1003,338,0', '1009,317,1017,341,0','1004,315,1012,339,0', '1057,323,1065,347,0','1056,320,1068,345,0', '980,354,993,386,0', '1074,346,1085,377,0','1063,348,1074,378,0', '1067,372,1078,408,0','1019,367,1030,393,0', '1040,361,1051,395,0','1034,354,1043,382,0', '1041,352,1052,381,0','1021,392,1036,430,0', '1065,405,1077,440,0', '959,387,973,423,0','974,388,988,420,0', '927,411,939,445,0', '913,406,926,434,2','1308,388,1333,461,0', '1205,414,1219,461,0','1222,400,1239,451,0', '1235,408,1248,458,0','1208,435,1238,510,0', '1223,434,1258,520,0','1132,478,1164,561,0', '1130,619,1166,679,0','1056,602,1081,671,0', '1106,592,1118,659,0','1193,694,1255,740,2', '1194,670,1234,721,2', '853,416,901,461,3','735,464,766,557,0', '911,482,962,561,0', '1020,444,1056,524,0','992,433,1029,512,0', '946,442,983,525,0', '882,421,916,498,0','868,421,907,492,0', '821,566,904,663,3', '539,450,564,540,0','554,441,577,530,0', '475,463,499,552,0', '481,462,506,546,0','424,442,463,535,0', '430,422,480,517,0', '330,414,378,508,0','109,416,148,510,0', '206,395,234,479,0', '221,438,255,519,0','44,389,104,476,0', '7,393,41,479,0', '273,600,370,708,3','76,650,218,764,3', '201,720,310,764,3', '0,562,157,763,8','234,354,259,402,-1', '391,257,416,271,-1', '157,381,169,419,0','223,324,232,353,0', '173,361,191,397,0', '125,370,137,406,0','33,382,46,423,0', '133,333,149,358,2', '137,324,148,352,1','102,339,112,358,1', '171,323,183,352,0', '183,319,193,351,0','183,304,194,332,0', '179,278,187,306,0', '144,263,159,276,2','100,269,109,285,0', '107,263,112,281,0', '111,271,117,281,0','70,295,84,316,1', '76,291,85,316,0', '90,289,97,315,0','95,294,103,323,0', '101,298,111,324,0', '190,264,198,285,0','197,275,204,295,0', '203,272,211,294,0', '209,265,215,285,0','202,267,209,287,0', '299,288,308,310,0', '326,286,334,304,0','511,390,553,436,3', '566,223,583,239,3', '577,215,591,228,3','572,206,588,220,3', '576,201,588,213,3', '573,198,585,209,3','572,193,584,203,3', '568,190,579,200,3', '540,189,551,197,3','553,195,564,204,3', '553,199,565,207,3', '551,203,565,213,3','552,209,565,221,3', '546,212,561,224,3', '543,220,560,234,3','537,223,553,241,4', '534,234,553,251,3', '528,242,548,258,3','527,251,546,267,3', '521,257,543,276,3', '527,269,550,288,3','518,278,543,300,3', '491,303,522,330,3', '479,317,514,348,3','500,279,527,313,4', '467,334,505,368,3', '455,355,493,391,3','442,382,488,422,3', '519,187,530,195,3', '518,192,528,200,3','496,195,519,221,8', '533,193,543,202,3', '533,199,545,209,3','531,203,546,215,3', '525,209,538,223,3', '522,215,536,229,3','518,221,534,235,3', '513,226,531,242,3', '502,235,521,249,3','492,225,509,238,3', '498,218,513,229,3', '472,221,490,236,3','471,231,493,245,3', '466,242,486,257,3', '497,244,517,259,3','487,252,510,272,3', '475,260,502,283,3', '448,254,472,275,3','477,278,502,298,3', '465,287,498,312,3', '447,274,475,297,3','432,285,465,311,3', '443,299,474,327,3', '427,316,464,346,3','410,337,452,372,3', '386,368,434,409,3', '357,400,410,440,3','371,325,409,358,3', '374,304,408,333,3', '390,290,423,321,3','403,280,430,303,3', '415,270,443,291,3', '261,298,377,426,8','423,240,430,259,0', '345,254,351,273,0', '339,254,344,271,0','314,244,321,261,0', '304,242,310,260,0', '309,238,315,254,0','295,225,298,242,0', '300,225,304,241,0', '305,226,310,244,0','375,235,381,253,0', '347,231,351,248,0', '350,231,356,247,0','410,237,414,252,0', '413,235,418,250,0', '419,235,426,252,0','398,237,401,253,0', '390,236,395,253,0', '375,226,379,242,0','379,228,384,243,0', '384,229,388,245,0', '393,196,413,214,4']])
        
        image = cv2.cvtColor(np.array(image, np.float32), cv2.COLOR_BGR2RGB)
        image_test = draw_bbox(np.copy(image), np.copy(bboxes), CLASSES_PATH=YOLO_CLASS_PATH, show_label=False)
        
        label_sbboxes, label_mbboxes, label_lbboxes, sbboxes, mbboxes, lbboxes = self.preprocess_true_bboxes(bboxes)
        bbox_test = np.concatenate([sbboxes[:,:2] - sbboxes[:,2:]*0.5, sbboxes[:,:2]+sbboxes[:,2:]*0.5], axis=-1)
        bbox_test = np.concatenate([bbox_test, np.ones((YOLO_MAX_BBOX_PER_SCALE,1))], axis=-1)

        image_test2 = draw_bbox(np.copy(image), np.copy(bbox_test), CLASSES_PATH=YOLO_CLASS_PATH, show_label=False)
        cv2.imshow("Test label for small bboxes", image_test2)
        cv2.imshow("Test after parse_annotation()", image_test)
        if cv2.waitKey() == "q":
            pass
        cv2.destroyAllWindows()
        print("Test")
       
if __name__ == '__main__':
    train_dataset = Dataset('test', TESTING=True)
    test_way = 1
    i = 0
    if test_way:
        for image, bboxes in train_dataset:
            i += 1
            print(i)       
    else:
        train_dataset.test()
    


