#===============================================================#
#                                                               #
#   File name   : YOLOv3_mAP.py                                 #
#   Author      : hxnghia99                                     #
#   Created date: May 19th, 2022                                #
#   GitHub      : https://github.com/hxnghia99/Yolov4_Learning  #
#   Description : YOLOv3 mAP calculation for evaluation         #
#                                                               #
#===============================================================#


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import tensorflow as tf
from YOLOv3_dataset import Dataset
from YOLOv3_model import YOLOv3_Model
from YOLOv3_utils import *
from YOLOv3_config import *
import shutil
import json
import time


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
    MIN_OVERLAP = 0.5   #value to define true/false positive
    
    CLASS_NAMES = read_class_names(CLASSES_PATH)
    #Check and create folder to store ground truth and mAP result
    ground_truth_dir_path = GT_DIR
    if not os.path.exists("YOLOv3-for-studying/mAP"):
        os.mkdir("YOLOv3-for-studying/mAP")
    if os.path.exists(ground_truth_dir_path):
        shutil.rmtree(ground_truth_dir_path)
    os.mkdir(ground_truth_dir_path)

    print(f"\n Calculating mAP{int(iou_threshold*100)}... \n")

    #Count the total of ground truth objects for each class
    gt_counter_per_class = {}
    for index in range(dataset.num_samples):
        annotation = dataset.annotations[index]
        _, gt_bboxes = dataset.parse_annotation(annotation, True)
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
            gt_bboxes_json_data.append({"class_name": class_name, "bbox": bbox, "used": False})

            #increase count for specific class
            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                gt_counter_per_class[class_name] = 1
        #saving grouth truth bboxes for each image
        with open(f'{ground_truth_dir_path}/{str(index)}_ground_truth.json', 'w') as outfile:
            json.dump(gt_bboxes_json_data, outfile)
    #Get class names in total of testset and the number of classes
    gt_class_names = list(gt_counter_per_class.keys())
    gt_class_names = sorted(gt_class_names)
    num_gt_classes = len(gt_class_names)
    
    #Calculate average FPS and store prediction bboxes to a list of specific classes
    times = []
    json_pred = [[] for _ in range(num_gt_classes)]
    for index in range(dataset.num_samples):
        annotation = dataset.annotations[index]
        original_image, _ = dataset.parse_annotation(annotation, True)
        image = image_preprocess(np.copy(original_image), TEST_INPUT_SIZE)
        image_data = image[np.newaxis, ...].astype(np.float32)

        #measure time to make prediction
        t1 = time.time()
        pred_bboxes = Yolo(image_data, training=False)
        t2 = time.time()
        times.append(t2-t1)

        #post process for prediction bboxes
        pred_bboxes = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bboxes]
        pred_bboxes = tf.concat(pred_bboxes, axis=0)                                #shape [total_bboxes, 5 + NUM_CLASS]
        pred_bboxes = postprocess_boxes(pred_bboxes, original_image, TEST_INPUT_SIZE, score_threshold)  #remove invalid and low score bboxes
        pred_bboxes = nms(pred_bboxes, iou_threshold, method='nms')                 #remove bboxes for same object in specific class
        #Save each prediction bbox to list for specific class
        for pred_bbox in pred_bboxes:
            pred_coordinates    = np.array(pred_bbox[:4], dtype=np.int32)
            pred_conf           = pred_bbox[4]
            pred_class_idx      = int(pred_bbox[5])
            pred_class_name     = CLASS_NAMES[pred_class_idx]
            pred_conf           = '%.4f' % pred_conf
            xmin, ymin, xmax, ymax = list(map(str, pred_coordinates))
            bbox = xmin + " " + ymin + " " + xmax + " " + ymax
            json_pred[gt_class_names.index(pred_class_name)].append({"confidence": str(pred_conf), "file_id": str(index), "bbox": str(bbox)})
    times_ms = sum(times)/len(times) * 1000
    fps = 1000 / times_ms
    print("\nFinished extracting ground truth bboxes from dataset... \n")

    #save list of predictions for specific class with descending order of confidence score
    for class_name in gt_class_names:
        json_pred[gt_class_names.index(class_name)].sort(key=lambda x: float(x['confidence']), reverse=True)
        with open(f'{ground_truth_dir_path}/{class_name}_predictions.json', 'w') as outfile:
            json.dump(json_pred[gt_class_names.index(class_name)], outfile)

    #Calculate each AP and mAP of the model, then print out result
    sum_AP = 0.0
    AP_dictionary = {}
    with open(mAP_PATH, 'w') as results_file:
        results_file.write("# AP and precision/recall per class \n")
        count_true_positives = {}
        #Calculate AP of specific class and print out result
        for class_name in gt_class_names:
            count_true_positives[class_name] = 0
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
                        if intersect_w > 0 and intersect_h > 0:
                            overlap = bboxes_iou_from_minmax(gt_coordinates[np.newaxis,:], pred_coordinates[np.newaxis,:])
                            if overlap > overlap_max:
                                overlap_max = overlap
                                gt_match = obj
                #assign prediction as true positive/false positive
                if overlap_max > MIN_OVERLAP:
                    #true positive
                    if not bool(gt_match['used']):
                        true_positive[idx] = 1
                        gt_match['used'] = True
                        count_true_positives[class_name] += 1
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
            print("'{}' AP = {:0.4f}\n".format(class_name, ap))

            #print result of class AP into result file
            text = "{0:.3f}%".format(ap * 100) + " = " + class_name + " AP "
            rounded_prec = ['%.3f' % x for x in prec]
            rounded_rec = ['%.3f' % x for x in rec]
            results_file.write(text + "\n Precision: " + str(rounded_prec)
                                    + "\n Recall   : " + str(rounded_rec) + "\n\n")
            AP_dictionary[class_name] = ap
        
        #Calculate mAP and print result
        results_file.write('\n# mAP of all classes\n')
        mAP = sum_AP / num_gt_classes
        text = "mAP = {:.3f}% , {:.2f} FPS".format(mAP*100, fps)
        results_file.write(text + "\n")
        print(text)

        return mAP*100




if __name__ == '__main__':
    weights_file = EVALUATION_WEIGHT_FILE
    yolo = YOLOv3_Model(input_size=YOLO_INPUT_SIZE, CLASSES_PATH=YOLO_CLASS_PATH)
    testset = Dataset('test', TEST_INPUT_SIZE=YOLO_INPUT_SIZE)
    if USE_CUSTOM_WEIGHTS:
        if EVALUATION_DATASET_TYPE == "COCO":
            load_yolov3_weights(yolo, weights_file)
        else:
            yolo.load_weights(weights_file) # use custom weights   
    get_mAP(yolo, testset, score_threshold=TEST_SCORE_THRESHOLD, iou_threshold=TEST_IOU_THRESHOLD, TEST_INPUT_SIZE=YOLO_INPUT_SIZE,
            CLASSES_PATH=YOLO_CLASS_PATH, GT_DIR=VALIDATE_GT_RESULTS_DIR, mAP_PATH=VALIDATE_MAP_RESULT_PATH)
  



