#===============================================================#
#                                                               #
#   File name   : YOLOv4_mAP.py                                 #
#   Author      : hxnghia99                                     #
#   Created date: May 24th, 2022                                #
#   GitHub      : https://github.com/hxnghia99/Yolov4_Learning  #
#   Description : YOLOv4 mAP calculation for evaluation         #
#                                                               #
#===============================================================#


import sys
import numpy as np
from YOLOv4_dataset import Dataset
from YOLOv4_model import *
from YOLOv4_utils import *
from YOLOv4_config import *
from YOLOv4_slicing import *
import shutil
import json
import time
import glob
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: 
        print("RuntimeError in tf.config.experimental.list_physical_devices('GPU')")



#Calculate AP using interpolation from all points, refer AP matlab code VOC2012
def all_points_interpolation_AP(prec, rec):
    #insert first and last elements for precision and recall lists
    prec.insert(0, 0.0)
    prec.append(0.0)
    interpolated_prec = prec[:]
    rec.insert(0, 0.0)
    rec.append(1.0)
    interpolated_rec = rec[:]
    #calculate the interpolation of precision
    for i in range(len(interpolated_prec)-2, -1, -1):
        interpolated_prec[i] = max(interpolated_prec[i], interpolated_prec[i+1])
    #calculate the list of indexes where the recall changes
    i_list = []
    for i in range(1, len(interpolated_rec)):
        if interpolated_rec[i] != interpolated_rec[i-1]:
            i_list.append(i)
    #calculate AP
    ap = 0.0
    for i in i_list:
        ap += ((interpolated_rec[i] - interpolated_rec[i-1]) * interpolated_prec[i])
    return ap



#Calculate AP for each class, mAP of the model
def get_mAP(Yolo, dataset, score_threshold=VALIDATE_SCORE_THRESHOLD, iou_threshold=VALIDATE_IOU_THRESHOLD, TEST_INPUT_SIZE=TEST_INPUT_SIZE, 
            CLASSES_PATH=YOLO_COCO_CLASS_PATH, GT_DIR=VALIDATE_GT_RESULTS_DIR, mAP_PATH=VALIDATE_MAP_RESULT_PATH):
    if USE_PRIMARY_EVALUATION_METRIC:
        MIN_OVERLAP_RANGE = np.array([50+i*5 for i in range(10)], dtype=np.int32)
        print(f"\n Calculating primary mAP (0.5:0.95)... \n")
    else:
        MIN_OVERLAP_RANGE = np.array([50])   #value to define true/false positive
        print(f"\n Calculating mAP50... \n")

    size_ranges = np.array([[0, 32**2],
                            [32**2, 96**2],
                            [96**2, 1e+10]], np.float32) / (640*480) 
    
    CLASS_NAMES = read_class_names(CLASSES_PATH)
    #Check and create folder to store ground truth and mAP result
    ground_truth_dir_path = GT_DIR
    if not os.path.exists("YOLOv4-for-studying/mAP"):
        os.mkdir("YOLOv4-for-studying/mAP")
    if os.path.exists(ground_truth_dir_path):
        shutil.rmtree(ground_truth_dir_path)
    os.mkdir(ground_truth_dir_path)

    #Count the total of ground truth objects for each class
    gt_counter_per_class = {}
    gt_counter_per_class_small = {}
    gt_counter_per_class_medium = {}
    gt_counter_per_class_large = {}
    for index in range(dataset.num_samples):
        annotation = dataset.annotations[index]
        if USE_SUPER_RESOLUTION_INPUT:
            original_image, gt_bboxes, _  = dataset.parse_annotation(annotation, mAP=True)
        else:
            original_image, gt_bboxes  = dataset.parse_annotation(annotation, mAP=True)
        original_h, original_w, _ = original_image.shape

        #eliminate ignored region class and "other" class
        if EVALUATION_DATASET_TYPE == "VISDRONE":
            bbox_mask = np.logical_and(gt_bboxes[:,4]>-0.5, gt_bboxes[:,4]<9.5)
            gt_bboxes = gt_bboxes[bbox_mask]

        if FILTER_GT_BBOX_SIZE:
            _, gt_bboxes_resized = image_preprocess(np.copy(original_image), TEST_INPUT_SIZE, np.copy(gt_bboxes))
            gt_bboxes_flag = np.multiply.reduce(np.array(gt_bboxes_resized)[:,2:4] - np.array(gt_bboxes_resized)[:,0:2], axis=-1) >= MIN_NUM_PIXEL
            gt_bboxes = np.array(gt_bboxes)[gt_bboxes_flag]


        num_gt_bboxes = len(gt_bboxes)
        if len(gt_bboxes) == 0:
            gt_coordinates  = []
            gt_classes      = []
        else:
            gt_coordinates  = gt_bboxes[:,:4]
            gt_classes      = gt_bboxes[:, 4]
        
        #Create data of one ground truth bbox to save
        gt_bboxes_json_data = []
        for i in range(num_gt_bboxes):
            class_name = CLASS_NAMES[gt_classes[i]]
            xmin, ymin, xmax, ymax = list(map(str, gt_coordinates[i]))
            bbox = xmin + " " + ymin + " " + xmax + " " + ymax
            used_list = [False] * 10
            gt_bboxes_json_data.append({"class_name": class_name, "bbox": bbox, "used": used_list})

            #increase count for specific class
            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                gt_counter_per_class[class_name] = 1
        #saving grouth truth bboxes for each image
        with open(f'{ground_truth_dir_path}/{str(index)}_ground_truth.json', 'w') as outfile:
            json.dump(gt_bboxes_json_data, outfile)
    

        gt_counter_per_class_according_to_size = []
        for size in EVALUATION_SIZE:
            #Create gt_bboxes for small objects
            if size == "small":
                temp = []
                for gt_bbox in gt_bboxes:
                    width = (gt_bbox[2] - gt_bbox[0] + 1) * 640 / original_w    #transform into width and height in image size 640x480
                    height = (gt_bbox[3] - gt_bbox[1] + 1) * 480 / original_h
                    if (width * height <= 32**2):
                        temp.append(gt_bbox)
                gt_bboxes_small = np.array(temp)
            
                num_gt_bboxes = len(gt_bboxes_small)
                if len(gt_bboxes_small) == 0:
                    gt_coordinates  = []
                    gt_classes      = []
                else:
                    gt_coordinates  = gt_bboxes_small[:,:4]
                    gt_classes      = gt_bboxes_small[:, 4]
                
                #Create data of one ground truth bbox to save
                gt_bboxes_json_data = []
                for i in range(num_gt_bboxes):
                    class_name = CLASS_NAMES[gt_classes[i]]
                    xmin, ymin, xmax, ymax = list(map(str, gt_coordinates[i]))
                    bbox = xmin + " " + ymin + " " + xmax + " " + ymax
                    used_list = [False] * 10
                    gt_bboxes_json_data.append({"class_name": class_name, "bbox": bbox, "used": used_list})

                    #increase count for specific class
                    if class_name in gt_counter_per_class_small:
                        gt_counter_per_class_small[class_name] += 1
                    else:
                        gt_counter_per_class_small[class_name] = 1
                #saving grouth truth bboxes for each image
                with open(f'{ground_truth_dir_path}/{str(index)}_ground_truth_{size}.json', 'w') as outfile:
                    json.dump(gt_bboxes_json_data, outfile)
            

            #Create gt_bboxes for medium objects
            if size == "medium":
                temp = []
                for gt_bbox in gt_bboxes:
                    width = (gt_bbox[2] - gt_bbox[0] + 1) * 640 / original_w    #transform into width and height in image size 640x480
                    height = (gt_bbox[3] - gt_bbox[1] + 1) * 480 / original_h
                    if (width * height > 32**2 and width * height <=96**2):
                        temp.append(gt_bbox)
                gt_bboxes_medium = np.array(temp)
            
                num_gt_bboxes = len(gt_bboxes_medium)
                if len(gt_bboxes_medium) == 0:
                    gt_coordinates  = []
                    gt_classes      = []
                else:
                    gt_coordinates  = gt_bboxes_medium[:,:4]
                    gt_classes      = gt_bboxes_medium[:, 4]
                
                #Create data of one ground truth bbox to save
                gt_bboxes_json_data = []
                for i in range(num_gt_bboxes):
                    class_name = CLASS_NAMES[gt_classes[i]]
                    xmin, ymin, xmax, ymax = list(map(str, gt_coordinates[i]))
                    bbox = xmin + " " + ymin + " " + xmax + " " + ymax
                    used_list = [False] * 10
                    gt_bboxes_json_data.append({"class_name": class_name, "bbox": bbox, "used": used_list})

                    #increase count for specific class
                    if class_name in gt_counter_per_class_medium:
                        gt_counter_per_class_medium[class_name] += 1
                    else:
                        gt_counter_per_class_medium[class_name] = 1
                #saving grouth truth bboxes for each image
                with open(f'{ground_truth_dir_path}/{str(index)}_ground_truth_{size}.json', 'w') as outfile:
                    json.dump(gt_bboxes_json_data, outfile)


            #Create gt_bboxes for large objects
            if size == "large":
                temp = []
                for gt_bbox in gt_bboxes:
                    width = (gt_bbox[2] - gt_bbox[0] + 1) * 640 / original_w    #transform into width and height in image size 640x480
                    height = (gt_bbox[3] - gt_bbox[1] + 1) * 480 / original_h
                    if (width * height > 96**2 ):
                        temp.append(gt_bbox)
                gt_bboxes_large = np.array(temp)

                num_gt_bboxes = len(gt_bboxes_large)
                if len(gt_bboxes_large) == 0:
                    gt_coordinates  = []
                    gt_classes      = []
                else:
                    gt_coordinates  = gt_bboxes_large[:,:4]
                    gt_classes      = gt_bboxes_large[:, 4]
                
                #Create data of one ground truth bbox to save
                gt_bboxes_json_data = []
                for i in range(num_gt_bboxes):
                    class_name = CLASS_NAMES[gt_classes[i]]
                    xmin, ymin, xmax, ymax = list(map(str, gt_coordinates[i]))
                    bbox = xmin + " " + ymin + " " + xmax + " " + ymax
                    used_list = [False] * 10
                    gt_bboxes_json_data.append({"class_name": class_name, "bbox": bbox, "used": used_list})

                    #increase count for specific class
                    if class_name in gt_counter_per_class_large:
                        gt_counter_per_class_large[class_name] += 1
                    else:
                        gt_counter_per_class_large[class_name] = 1
                #saving grouth truth bboxes for each image
                with open(f'{ground_truth_dir_path}/{str(index)}_ground_truth_{size}.json', 'w') as outfile:
                    json.dump(gt_bboxes_json_data, outfile)

    
    for size in EVALUATION_SIZE:
        gt_counter_per_class_according_to_size.append(locals()["gt_counter_per_class_"+size])

    #Get class names in total of testset and the number of classes
    gt_class_names = list(gt_counter_per_class.keys())
    gt_class_names = sorted(gt_class_names)
    num_gt_classes = len(gt_class_names)
    


    #Calculate average FPS and store prediction bboxes to a list of specific classes
    times = []
    json_pred = [[] for _ in range(num_gt_classes)]
    for index in range(dataset.num_samples):
        annotation = dataset.annotations[index]
        if USE_SUPER_RESOLUTION_INPUT:
            original_image, bboxes, image_sr = dataset.parse_annotation(annotation, True)     #including cv2.cvtColor
        else:
            original_image, bboxes = dataset.parse_annotation(annotation, True)     #including cv2.cvtColor
            
        original_h, original_w, _ = original_image.shape
        # Create a new model using image original size scaling to 32
        if EVALUATE_ORIGINAL_SIZE:
            TEST_INPUT_SIZE = [int(np.ceil(original_w/32))*32, int(np.ceil(original_h/32))*32]
        
        if USE_SUPER_RESOLUTION_INPUT:
            image = image_preprocess(np.copy(image_sr), TEST_INPUT_SIZE)
        else:
            if TEST_INPUT_SIZE[0] == 448 and not USE_SUPERVISION:
                image = image_preprocess(np.copy(original_image), TEST_INPUT_SIZE)
            else:   
                image = image_preprocess(np.copy(original_image), np.int32(np.array(TEST_INPUT_SIZE)*2))
                image = image_preprocess(np.copy(image)*255.0, TEST_INPUT_SIZE)
        image_data = image[np.newaxis, ...].astype(np.float32)
        
        #measure time to make prediction
        t1 = time.time()
        pred_bboxes = Yolo(image_data, training=False)
        t2 = time.time()
        times.append(t2-t1)

        # pred_bboxes = pred_bboxes[1:6:2]
        #post process for prediction bboxes
        pred_bboxes = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bboxes]
        pred_bboxes = tf.concat(pred_bboxes, axis=0)                                #shape [total_bboxes, 5 + NUM_CLASS]
        pred_bboxes = postprocess_boxes(pred_bboxes, original_image, TEST_INPUT_SIZE, score_threshold)  #remove invalid and low score bboxes
        pred_bboxes = tf.convert_to_tensor(nms(pred_bboxes, method='nms', iou_threshold=iou_threshold))                 #remove bboxes for same object in specific class 

        if EVALUATION_DATASET_TYPE == "VISDRONE":
            bboxes = tf.cast(bboxes, dtype=tf.float64)
            ignored_bbox_mask   = bboxes[:,4]>-0.5
            ignored_bboxes      = tf.expand_dims(bboxes, axis=0)[tf.expand_dims(tf.math.logical_not(ignored_bbox_mask), axis=0)]
            other_bbox_mask     = bboxes[:,4]<9.5
            other_bboxes        = tf.expand_dims(bboxes, axis=0)[tf.expand_dims(tf.math.logical_not(other_bbox_mask), axis=0)]

            #getting mask of bboxes in ignored region
            removed_ignored_mask = tf.convert_to_tensor(np.zeros(np.array(tf.shape(pred_bboxes)[0])), dtype=tf.bool)
            if tf.shape(ignored_bboxes)[0] != 0:
                pred_bboxes_temp = tf.expand_dims(pred_bboxes[:, :4], axis=1)       #shape [total_bboxes, 1, 4]
                ignored_bboxes = tf.expand_dims(ignored_bboxes[:, :4], axis=0)      #shape [1, num_bboxes, 4]
                intersect_tf = tf.maximum(pred_bboxes_temp[..., :2], ignored_bboxes[..., :2])
                intersect_br = tf.minimum(pred_bboxes_temp[..., 2:], ignored_bboxes[..., 2:])
                intersection = tf.maximum(intersect_br - intersect_tf, 0.0)
                intersection_area = tf.math.reduce_sum(tf.math.reduce_prod(intersection, axis=-1), axis=-1, keepdims=True)      #shape [num_pred_bboxes, 2]
                pred_bboxes_area = tf.math.reduce_prod(tf.maximum(pred_bboxes_temp[...,2:] - pred_bboxes_temp[...,:2], 0.0), axis=-1) #shape [num_pred_bboxes, 1]
                removed_ignored_mask = tf.reduce_max(intersection_area / pred_bboxes_area , axis=-1) > VISDRONE_IGNORED_THRESHOLD

            #getting mask of bboxes that overlap "other" class
            removed_other_mask = tf.convert_to_tensor(np.zeros(np.array(tf.shape(pred_bboxes)[0])), dtype=tf.bool)
            if tf.shape(other_bboxes)[0] != 0:
                pred_bboxes_temp = tf.expand_dims(pred_bboxes[:, :4], axis=1)      #shape [total_bboxes, 1, 4]
                other_bboxes = tf.expand_dims(other_bboxes[:, :4], axis=0)         #shape [1, num_bboxes, 4]
                ious = bboxes_iou_from_minmax(pred_bboxes_temp, other_bboxes)   #shape [total_bboxes, num_bboxes]
                max_ious = tf.reduce_max(ious, axis=-1)
                removed_other_mask = max_ious > VISDRONE_OTHER_THRESHOLD
            #getting mask of removed bboxes
            removed_bbox_mask = tf.math.logical_or(removed_ignored_mask, removed_other_mask)
            pred_bboxes = tf.expand_dims(pred_bboxes, axis=0)[tf.expand_dims(tf.math.logical_not(removed_bbox_mask), axis=0)]

            if FILTER_GT_BBOX_SIZE:
                _, pred_bboxes_resized = image_preprocess(np.copy(original_image), TEST_INPUT_SIZE, np.copy(pred_bboxes))
                pred_bboxes_flag = np.multiply.reduce(np.array(pred_bboxes_resized)[:,2:4] - np.array(pred_bboxes_resized)[:,0:2], axis=-1) >= MIN_NUM_PIXEL
                pred_bboxes = np.array(pred_bboxes)[pred_bboxes_flag]


        # test_image = draw_bbox(cv2.cvtColor(np.copy(original_image), cv2.COLOR_BGR2RGB), np.copy(bboxes), "YOLOv4-for-studying/dataset/LG_DATASET/lg_class_names.txt", show_label=True)
        # cv2.imshow("Ground truth", cv2.resize(test_image, (1280, 720)))
        # test_image = draw_bbox(cv2.cvtColor(np.copy(original_image), cv2.COLOR_BGR2RGB), np.copy(pred_bboxes), YOLO_CLASS_PATH, show_label=True)
        # cv2.imshow("Prediction after slicing", cv2.resize(test_image, (1280, 720)))
        # if cv2.waitKey() == "q":
        #     pass
        # cv2.destroyAllWindows()
        # continue

        sys.stdout.write("\rLoaded image: {}".format(index))
       
        #Save each prediction bbox to list for specific class
        for pred_bbox in pred_bboxes:
            pred_coordinates    = np.array(pred_bbox[:4], dtype=np.int32)
            pred_conf           = pred_bbox[4]
            pred_class_idx      = int(pred_bbox[5])
            pred_class_name     = CLASS_NAMES[pred_class_idx]
            pred_conf           = '%.4f' % pred_conf
            xmin, ymin, xmax, ymax = list(map(str, pred_coordinates))
            bbox = xmin + " " + ymin + " " + xmax + " " + ymax
            json_pred[gt_class_names.index(pred_class_name)].append({"confidence": str(pred_conf), "file_id": str(index), "bbox": str(bbox), "original_width": str(original_w), "original_height": str(original_h)})
    
    
    
    times_ms = sum(times)/len(times) * 1000
    fps = 1000 / times_ms
    print("\nFinished extracting ground truth bboxes and prediction bboxes... \n")

    #save list of predictions for specific class with descending order of confidence score
    for class_name in gt_class_names:
        json_pred[gt_class_names.index(class_name)].sort(key=lambda x: float(x['confidence']), reverse=True)
        with open(f'{ground_truth_dir_path}/{class_name}_predictions.json', 'w') as outfile:
            json.dump(json_pred[gt_class_names.index(class_name)], outfile)



    #Calculate each AP and mAP of the model, then print out result
    AP_dictionary = {}
    with open(mAP_PATH, 'w') as results_file:
        results_file.write("#   EVALUATION RESULTS   # \n\n")
        sum_mAP = 0.0
        for index, MIN_OVERLAP_100 in enumerate(MIN_OVERLAP_RANGE):
            MIN_OVERLAP = MIN_OVERLAP_100/100
            sum_AP = 0.0
            results_file.write("# AP and precision/recall per class: IoU threshold = {:.2f} \n".format(MIN_OVERLAP))
            # count_true_positives = {}
            #Calculate AP of specific class and print out result
            for class_name in gt_class_names:
                # count_true_positives[class_name] = 0
                #Load predictions
                predictions_file = f'{ground_truth_dir_path}/{class_name}_predictions.json'
                predictions_data = json.load(open(predictions_file))
                num_predictions = len(predictions_data)
                true_positive = [0] * num_predictions
                false_positive = [0] * num_predictions
                #With each prediction, read all gt_bboxes in prediction's image and select the object with maximum overlap
                for idx, prediction in enumerate(predictions_data):
                    pred_coordinates = np.array([float(x) for x in prediction['bbox'].split()])
                    overlap_max = -1
                    gt_match    = -1
                    #Load ground truth file of the prediction
                    file_id = prediction['file_id']
                    gt_file = f'{ground_truth_dir_path}/{str(file_id)}_ground_truth.json'
                    ground_truth_data = json.load(open(gt_file))                            #all gt_bboxes in an image
                    #Go through all gt_bboxes and select the best overlap with prediction regarding to same class
                    for obj in ground_truth_data:
                        if obj['class_name'] == class_name:
                            gt_coordinates = np.array([float(x) for x in obj['bbox'].split()])
                            intersection = np.array([np.max((gt_coordinates[0], pred_coordinates[0])),
                                                    np.max((gt_coordinates[1], pred_coordinates[1])),
                                                    np.min((gt_coordinates[2], pred_coordinates[2])),
                                                    np.min((gt_coordinates[3], pred_coordinates[3]))])
                            intersect_w, intersect_h = intersection[2:] - intersection[:2]
                            # intersect_w, intersect_h = intersection[2:] - intersection[:2] + 1
                            if intersect_w > 0 and intersect_h > 0:
                                overlap = bboxes_iou_from_minmax(gt_coordinates[np.newaxis,:], pred_coordinates[np.newaxis,:])
                                # # compute overlap (IoU) = area of intersection / area of union
                                # union_area = (pred_coordinates[2] - pred_coordinates[0] + 1) * (pred_coordinates[3] - pred_coordinates[1] + 1) + \
                                # (gt_coordinates[2] - gt_coordinates[0]+ 1) * (gt_coordinates[3] - gt_coordinates[1] + 1) - intersect_w * intersect_h
                                # overlap = intersect_w * intersect_h / union_area
                                if overlap > overlap_max:
                                    overlap_max = overlap
                                    gt_match = obj
                    #assign prediction as true positive/false positive
                    if overlap_max > MIN_OVERLAP:
                        #true positive
                        if not bool(gt_match['used'][index]):
                            true_positive[idx] = 1
                            gt_match['used'][index] = True
                            # count_true_positives[class_name] += 1
                            #update the 'used' state for the gt bbox
                            with open(gt_file, 'w') as f:
                                f.write(json.dumps(ground_truth_data))
                        #false positive
                        else:
                            false_positive[idx] = 1
                    #false positive
                    else:
                        false_positive[idx] = 1
                
                #Calculate accumulated TP and FP for each class
                accumulated_value = 0
                for idx, value in enumerate(true_positive):
                    true_positive[idx] += accumulated_value
                    accumulated_value += value
                accumulated_value = 0
                for idx, value in enumerate(false_positive):
                    false_positive[idx] += accumulated_value
                    accumulated_value += value
                #Calculate precision and precall for each class
                prec = [0] * num_predictions
                for idx in range(len(prec)):
                    prec[idx] = float(true_positive[idx]) / (false_positive[idx] + true_positive[idx])
                rec = [0] * num_predictions
                for idx in range(len(rec)):
                    rec[idx] = float(true_positive[idx]) / gt_counter_per_class[class_name]
                
                #calculate AP
                ap = all_points_interpolation_AP(prec, rec)
                sum_AP += ap
                
                # print("'{}' AP = {:0.4f}\n".format(class_name, ap))
                #print result of class AP into result file
                text = "{0:.3f}%".format(ap * 100) + " = " + class_name + " AP" + str(MIN_OVERLAP_100) + " \n" 
                # rounded_prec = ['%.3f' % x for x in prec]
                # rounded_rec = ['%.3f' % x for x in rec]
                # results_file.write(text + "\n Precision: " + str(rounded_prec)
                #                         + "\n Recall   : " + str(rounded_rec) + "\n\n")
                results_file.write(text)
                if class_name not in AP_dictionary:
                    AP_dictionary[class_name] = [ap]
                else:
                    AP_dictionary[class_name].append(ap)

                if MIN_OVERLAP_100 == 50 or MIN_OVERLAP_100 == 75:
                    print(text)

            #Calculate mAP and print result
            results_file.write(f'\n# mAP{MIN_OVERLAP_100} of all classes\n')
            mAP = sum_AP / num_gt_classes
            text = "mAP{} = {:.2f}%  \n".format(MIN_OVERLAP_100, mAP*100)
            results_file.write(text + "\n")
            print(text)
            sum_mAP += mAP
        if USE_PRIMARY_EVALUATION_METRIC:
            results_file.write(f'\n#  mAP50:95 of all classes\n')    
            mAP = sum_mAP / len(MIN_OVERLAP_RANGE)
            text = "mAP50:95 = {:.3f}% , {:.2f} FPS \n".format(mAP*100, fps)
            results_file.write(text + "\n")
            print(text)
            for class_name in gt_class_names:
                AP = sum(AP_dictionary[class_name]) / len(MIN_OVERLAP_RANGE)
                text = "{0:.2f}%".format(AP * 100) + " = " + class_name + " AP" 
                results_file.write(text + "\n")
                print(text)
            print("\n\n")

    for size_idx, size in enumerate(EVALUATION_SIZE):
        print(f'{size.upper()} EVALUATION')
        #Calculate each AP and mAP of the model, then print out result
        AP_dictionary = {}
        with open(mAP_PATH, 'a') as results_file:
            results_file.write(f'\n\n# {size.upper()}-OBJECT EVALUATION RESULTS   # \n\n')
            sum_mAP = 0.0
            for index, MIN_OVERLAP_100 in enumerate(MIN_OVERLAP_RANGE):
                MIN_OVERLAP = MIN_OVERLAP_100/100
                sum_AP = 0.0
                results_file.write("# AP and precision/recall per class: IoU threshold = {:.2f} \n".format(MIN_OVERLAP))
                # count_true_positives = {}
                #Calculate AP of specific class and print out result
                for class_name in gt_class_names:
                    # count_true_positives[class_name] = 0
                    #Load predictions
                    predictions_file = f'{ground_truth_dir_path}/{class_name}_predictions.json'
                    predictions_data = json.load(open(predictions_file))
                    predictions_data_new = []
                    for idx, prediction in enumerate(predictions_data):
                        #Filter out the different size-based predictions
                        pred_coordinates = np.array([float(x) for x in prediction['bbox'].split()])
                        pred_size = (pred_coordinates[2]-pred_coordinates[0]) * (pred_coordinates[3]-pred_coordinates[1])
                        original_size = float(prediction["original_height"]) * float(prediction["original_width"])
                        if pred_size > size_ranges[size_idx][0]*original_size and pred_size <= size_ranges[size_idx][1]*original_size:
                            predictions_data_new.append(prediction)
                    predictions_data = predictions_data_new
                    num_predictions = len(predictions_data)
                    true_positive = [0] * num_predictions
                    false_positive = [0] * num_predictions
                    #With each prediction, read all gt_bboxes in prediction's image and select the object with maximum overlap
                    for idx, prediction in enumerate(predictions_data):
                        pred_coordinates = np.array([float(x) for x in prediction['bbox'].split()])
                        overlap_max = -1
                        gt_match    = -1
                        #Load ground truth file of the prediction
                        file_id = prediction['file_id']
                        gt_file = f'{ground_truth_dir_path}/{str(file_id)}_ground_truth_{size}.json'
                        ground_truth_data = json.load(open(gt_file))                            #all gt_bboxes in an image
                        #Go through all gt_bboxes and select the best overlap with prediction regarding to same class
                        for obj in ground_truth_data:
                            if obj['class_name'] == class_name:
                                gt_coordinates = np.array([float(x) for x in obj['bbox'].split()])
                                intersection = np.array([np.max((gt_coordinates[0], pred_coordinates[0])),
                                                        np.max((gt_coordinates[1], pred_coordinates[1])),
                                                        np.min((gt_coordinates[2], pred_coordinates[2])),
                                                        np.min((gt_coordinates[3], pred_coordinates[3]))])
                                intersect_w, intersect_h = intersection[2:] - intersection[:2]
                                if intersect_w > 0 and intersect_h > 0:
                                    overlap = bboxes_iou_from_minmax(gt_coordinates[np.newaxis,:], pred_coordinates[np.newaxis,:])
                                    if overlap > overlap_max:
                                        overlap_max = overlap
                                        gt_match = obj
                        #assign prediction as true positive/false positive
                        if overlap_max > MIN_OVERLAP:
                            #true positive
                            if not bool(gt_match['used'][index]):
                                true_positive[idx] = 1
                                gt_match['used'][index] = True
                                # count_true_positives[class_name] += 1
                                #update the 'used' state for the gt bbox
                                with open(gt_file, 'w') as f:
                                    f.write(json.dumps(ground_truth_data))
                            #false positive
                            else:
                                false_positive[idx] = 1
                        #false positive
                        else:
                            false_positive[idx] = 1
                    
                    #Calculate accumulated TP and FP for each class
                    accumulated_value = 0
                    for idx, value in enumerate(true_positive):
                        true_positive[idx] += accumulated_value
                        accumulated_value += value
                    accumulated_value = 0
                    for idx, value in enumerate(false_positive):
                        false_positive[idx] += accumulated_value
                        accumulated_value += value
                    #Calculate precision and precall for each class
                    prec = [0] * num_predictions
                    for idx in range(len(prec)):
                        prec[idx] = float(true_positive[idx]) / (false_positive[idx] + true_positive[idx])
                    rec = [0] * num_predictions
                    for idx in range(len(rec)):
                        rec[idx] = float(true_positive[idx]) / gt_counter_per_class_according_to_size[size_idx][class_name]
                    
                    #calculate AP
                    ap = all_points_interpolation_AP(prec, rec)
                    sum_AP += ap
                    
                    # print("'{}' AP = {:0.4f}\n".format(class_name, ap))
                    #print result of class AP into result file
                    text = "{0:.3f}%".format(ap * 100) + " = " + class_name + " AP" + str(MIN_OVERLAP_100) + " \n" 
                    # rounded_prec = ['%.3f' % x for x in prec]
                    # rounded_rec = ['%.3f' % x for x in rec]
                    # results_file.write(text + "\n Precision: " + str(rounded_prec)
                    #                         + "\n Recall   : " + str(rounded_rec) + "\n\n")
                    results_file.write(text)
                    if class_name not in AP_dictionary:
                        AP_dictionary[class_name] = [ap]
                    else:
                        AP_dictionary[class_name].append(ap)

                    if MIN_OVERLAP_100 == 50 or MIN_OVERLAP_100 == 75:
                        print(text)

                #Calculate mAP and print result
                results_file.write(f'\n# mAP{MIN_OVERLAP_100} of all classes\n')
                mAP = sum_AP / num_gt_classes
                text = "mAP{} = {:.2f}%  \n".format(MIN_OVERLAP_100, mAP*100)
                results_file.write(text + "\n")
                print(text)
                sum_mAP += mAP
            if USE_PRIMARY_EVALUATION_METRIC:
                results_file.write(f'\n#  mAP50:95 of all classes\n')    
                mAP = sum_mAP / len(MIN_OVERLAP_RANGE)
                text = "mAP50:95 = {:.3f}% , {:.2f} FPS \n".format(mAP*100, fps)
                results_file.write(text + "\n")
                print(text)
                for class_name in gt_class_names:
                    AP = sum(AP_dictionary[class_name]) / len(MIN_OVERLAP_RANGE)
                    text = "{0:.2f}%".format(AP * 100) + " = " + class_name + " AP" 
                    results_file.write(text + "\n")
                    print(text)
                print("\n\n")

        

def load_weights_old_new(weights_file):
    new_model = YOLOv4_Model(CLASSES_PATH=YOLO_CLASS_PATH, Modified_model=False, training=True)
    old_model = YOLOv4_Model(CLASSES_PATH=YOLO_CLASS_PATH, Modified_model=False, training=False)

    new_model.load_weights(weights_file)

    for i in range(535):
        if i==517 or i==518 or i==519 or i==523 or i==527 or i==531:
            continue
        j = i
        if i>=520:
            j=i-3
        if i>=524:
            j=i-4
        if i>=528:
            j=i-5
        if i>=532:
            j=i-6
        if new_model.layers[i].get_weights() != []:
            old_model.layers[j].set_weights(new_model.layers[i].get_weights())
    return old_model

if __name__ == '__main__':
    # weights_file = "YOLOv4-for-studying/checkpoints/lg_dataset_transfer_224x128_P5_nFTT_P2/yolov4_lg_transfer"
    # weights_file = "YOLOv4-for-studying/checkpoints/epoch-41_valid-loss-9.82/yolov4_lg_transfer"
    weights_file = "YOLOv4-for-studying/checkpoints/Num-229_lg_dataset_transfer_448x256/epoch-87_valid-loss-5.41/yolov4_lg_transfer"
    # weights_file = "YOLOv4-for-studying/checkpoints/Num-209_visdrone_dataset_transfer_480x288_origin/epoch-51_valid-loss-267.88/yolov4_visdrone_transfer"
    # weights_file = EVALUATION_WEIGHT_FILE
    yolo = YOLOv4_Model(CLASSES_PATH=YOLO_CLASS_PATH, Modified_model=False)
    # yolo = create_YOLOv4_backbone(CLASSES_PATH=YOLO_CLASS_PATH)

    # yolo = load_weights_old_new(weights_file)

    testset = Dataset('test', TEST_INPUT_SIZE=YOLO_INPUT_SIZE, EVAL_MODE=True)
    
    
    # if USE_CUSTOM_WEIGHTS:
    #     if EVALUATION_DATASET_TYPE == "COCO":
    #         load_yolov4_weights(yolo, weights_file)
    #     else:
    #         yolo.load_weights(weights_file) # use custom weights   
    # # yolot = YOLOv4_Model(CLASSES_PATH=YOLO_CLASS_PATH, dilation_bb=True)
    # # temp = len(yolo.weights)
    # # for i in range(temp):
    # #     yolot.weights[i].assign(yolo.weights[i])
    # get_mAP(yolo, testset, score_threshold=TEST_SCORE_THRESHOLD, iou_threshold=TEST_IOU_THRESHOLD, TEST_INPUT_SIZE=YOLO_INPUT_SIZE,
    #         CLASSES_PATH=YOLO_CLASS_PATH, GT_DIR=VALIDATE_GT_RESULTS_DIR, mAP_PATH=VALIDATE_MAP_RESULT_PATH)


    PATH = "YOLOv4-for-studying/checkpoints/lg_dataset_transfer_224x128_v2.4/*"
    list_path = glob.glob(PATH)
    for i,path in enumerate(list_path):
        VALIDATE_MAP_RESULT_PATH = f"YOLOv4-for-studying/mAP/results-lg_{i+51}.txt"
        yolo.load_weights(path + "/yolov4_lg_transfer")

        get_mAP(yolo, testset, score_threshold=TEST_SCORE_THRESHOLD, iou_threshold=TEST_IOU_THRESHOLD, TEST_INPUT_SIZE=YOLO_INPUT_SIZE,
            CLASSES_PATH=YOLO_CLASS_PATH, GT_DIR=VALIDATE_GT_RESULTS_DIR, mAP_PATH=VALIDATE_MAP_RESULT_PATH)