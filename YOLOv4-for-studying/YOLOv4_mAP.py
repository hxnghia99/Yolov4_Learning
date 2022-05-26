#===============================================================#
#                                                               #
#   File name   : YOLOv4_mAP.py                                 #
#   Author      : hxnghia99                                     #
#   Created date: May 24th, 2022                                #
#   GitHub      : https://github.com/hxnghia99/Yolov4_Learning  #
#   Description : YOLOv4 mAP calculation for evaluation         #
#                                                               #
#===============================================================#


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import tensorflow as tf
from YOLOv4_dataset import Dataset
from YOLOv4_model import YOLOv4_Model
from YOLOv4_utils import *
from YOLOv4_config import *
import shutil
import json
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: 
        print("RuntimeError in tf.config.experimental.list_physical_devices('GPU')")


# def voc_ap(rec, prec):
    
#     rec.insert(0, 0.0) # insert 0.0 at begining of list
#     rec.append(1.0) # insert 1.0 at end of list
#     mrec = rec[:]
#     prec.insert(0, 0.0) # insert 0.0 at begining of list
#     prec.append(0.0) # insert 0.0 at end of list
#     mpre = prec[:]
#     """
#      This part makes the precision monotonically decreasing
#         (goes from the end to the beginning)
#         matlab:  for i=numel(mpre)-1:-1:1
#                                 mpre(i)=max(mpre(i),mpre(i+1));
#     """
#     # matlab indexes start in 1 but python in 0, so I have to do:
#     #   range(start=(len(mpre) - 2), end=0, step=-1)
#     # also the python function range excludes the end, resulting in:
#     #   range(start=(len(mpre) - 2), end=-1, step=-1)
#     for i in range(len(mpre)-2, -1, -1):
#         mpre[i] = max(mpre[i], mpre[i+1])
#     """
#      This part creates a list of indexes where the recall changes
#         matlab:  i=find(mrec(2:end)~=mrec(1:end-1))+1;
#     """
#     i_list = []
#     for i in range(1, len(mrec)):
#         if mrec[i] != mrec[i-1]:
#             i_list.append(i) # if it was matlab would be i + 1
#     """
#      The Average Precision (AP) is the area under the curve
#         (numerical integration)
#         matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
#     """
#     ap = 0.0
#     for i in i_list:
#         ap += ((mrec[i]-mrec[i-1])*mpre[i])
#     return ap, mrec, mpre


# def get_mAP(Yolo, dataset, score_threshold=VALIDATE_SCORE_THRESHOLD, iou_threshold=VALIDATE_IOU_THRESHOLD, TEST_INPUT_SIZE=TEST_INPUT_SIZE, 
#             CLASSES_PATH=LG_CLASS_NAMES_PATH, GT_DIR=VALIDATE_GT_RESULTS_DIR, mAP_PATH=VALIDATE_MAP_RESULT_PATH):
#     MINOVERLAP = 0.5 # default value (defined in the PASCAL VOC2012 challenge)

#     CLASS_NAMES = read_class_names(CLASSES_PATH)
#     #check directory storing validation results
#     ground_truth_dir_path = GT_DIR
#     if not os.path.exists('YOLOv3-for-studying/mAP'): 
#         os.mkdir('YOLOv3-for-studying/mAP')
#     if os.path.exists(ground_truth_dir_path): 
#         shutil.rmtree(ground_truth_dir_path)
#     os.mkdir(ground_truth_dir_path)

#     print(f'\nCalculating mAP{int(iou_threshold*100)}...\n')

#     #Reading each annotation (image + gt_bboxes) and get num_gt for each class
#     gt_counter_per_class = {}
#     for index in range(dataset.num_samples):
#         annotation = dataset.annotations[index]
#         _, gt_bboxes = dataset.parse_annotation(annotation, mAP=True)   #do not use image_preprocess()
#         #Get annotation of coordinates and classes
#         if len(gt_bboxes) == 0:
#             gt_coors = []
#             gt_classes = []
#         else:
#             gt_coors, gt_classes = gt_bboxes[:, :4], gt_bboxes[:, 4]    #shape [num_bboxes, data_dimension]
#         #Parse annotation of coordinates and class for each gt_bbox
#         bounding_boxes = []
#         for i in range(len(gt_bboxes)):
#             class_name = CLASS_NAMES[gt_classes[i]]
#             xmin, ymin, xmax, ymax = list(map(str, gt_coors[i]))
#             bbox = xmin + " " + ymin + " " + xmax + " " +ymax
#             bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
#             # count class-specific object
#             if class_name not in gt_counter_per_class:
#                 gt_counter_per_class[class_name] = 1
#             else:
#                 gt_counter_per_class[class_name] += 1
#         #store gt_bboxes into json file
#         with open(f'{ground_truth_dir_path}/{str(index)}_ground_truth.json', 'w') as outfile:
#             json.dump(bounding_boxes, outfile)
#     #name of all classes in validation dataset
#     name_classes = list(gt_counter_per_class.keys())
#     #sort the classes alphabetically
#     name_classes = sorted(name_classes)
#     num_classes = len(name_classes)
#     #For each annotation, do prediction
#     times = []
#     json_pred = [[] for _ in range(num_classes)]        #create list of empty shape [num_classes]
#     for index in range(dataset.num_samples):
#         annotation = dataset.annotations[index]
#         original_image, _ = dataset.parse_annotation(annotation, mAP=True) #do not use image_preprocess()
#         image = image_preprocess(np.copy(original_image), TEST_INPUT_SIZE)
#         image_data = image[np.newaxis, ...].astype(np.float32)  #add shape of batch
#         #get inference time
#         t1 = time.time()
#         pred_bbox = Yolo(image_data, training=False)        #[3 bbox prediction scales]
#         t2 = time.time()
#         times.append(t2-t1)
#         #flatten prediction into shape [num_bboxes, 5 + num_classes]
#         pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
#         pred_bbox = tf.concat(pred_bbox, axis=0)
#         #postprocess and non-maximum suppression for all predicted bboxes
#         bboxes = postprocess_boxes(pred_bbox, original_image, TEST_INPUT_SIZE, score_threshold)
#         bboxes = nms(bboxes, iou_threshold, method='nms')
#         #store each predicted bbox to a list of same class
#         for bbox in bboxes:
#             coor = np.array(bbox[:4], dtype=np.int32)
#             score = bbox[4]
#             class_ind = int(bbox[5])
#             class_name = CLASS_NAMES[class_ind]
#             score = '%.4f' % score
#             xmin, ymin, xmax, ymax = list(map(str, coor))
#             bbox = xmin + " " + ymin + " " + xmax + " " +ymax
#             json_pred[name_classes.index(class_name)].append({"confidence": str(score), "file_id": str(index), "bbox": str(bbox)})  #shape [num_classes, num_bboxes_for_specific_class]
#     #calculate FPS
#     average_ms = sum(times)/len(times)*1000
#     fps = 1000 / average_ms
#     #sort the predicted bboxes following the descending order of confidence score
#     for class_name in name_classes:
#         print("saving {}...".format(class_name))
#         json_pred[name_classes.index(class_name)].sort(key=lambda x:float(x['confidence']), reverse=True)
#         with open(f'{ground_truth_dir_path}/{class_name}_predictions.json', 'w') as outfile:
#             json.dump(json_pred[name_classes.index(class_name)], outfile)

#     #Calculate the AP for each class
#     sum_AP = 0.0
#     ap_dictionary = {}
#     # open file to store the results
#     with open(mAP_PATH, 'w') as results_file:
#         results_file.write("# AP and precision/recall per class\n")
#         count_true_positives = {}
#         for class_index, class_name in enumerate(name_classes):
#             count_true_positives[class_name] = 0
#             # Load predictions of that class
#             predictions_file = f'{ground_truth_dir_path}/{class_name}_predictions.json'
#             predictions_data = json.load(open(predictions_file))

#             # Assign predictions to ground truth objects
#             nd = len(predictions_data)
#             tp = [0] * nd # creates an array of zeros of size nd
#             fp = [0] * nd
#             for idx, prediction in enumerate(predictions_data):
#                 file_id = prediction["file_id"]
#                 # assign prediction to ground truth object if any
#                 #   open ground-truth with that file_id
#                 gt_file = f'{ground_truth_dir_path}/{str(file_id)}_ground_truth.json'
#                 ground_truth_data = json.load(open(gt_file))
#                 ovmax = -1
#                 gt_match = -1
#                 # load prediction bounding-box
#                 bb = [ float(x) for x in prediction["bbox"].split() ] # bounding box of prediction
#                 for obj in ground_truth_data:
#                     # look for a class_name match
#                     if obj["class_name"] == class_name:
#                         bbgt = [ float(x) for x in obj["bbox"].split() ] # bounding box of ground truth
#                         bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
#                         iw = bi[2] - bi[0] + 1
#                         ih = bi[3] - bi[1] + 1
#                         if iw > 0 and ih > 0:
#                             # compute overlap (IoU) = area of intersection / area of union
#                             ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
#                                             + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
#                             ov = iw * ih / ua
#                             if ov > ovmax:
#                                 ovmax = ov
#                                 gt_match = obj

#                 # assign prediction as true positive/don't care/false positive
#                 if ovmax >= MINOVERLAP:# if ovmax > minimum overlap
#                     if not bool(gt_match["used"]):
#                         # true positive
#                         tp[idx] = 1
#                         gt_match["used"] = True
#                         count_true_positives[class_name] += 1
#                         # update the ".json" file
#                         with open(gt_file, 'w') as f:
#                             f.write(json.dumps(ground_truth_data))
#                     else:
#                         # false positive (multiple detection)
#                         fp[idx] = 1
#                 else:
#                     # false positive
#                     fp[idx] = 1

#             # compute precision/recall
#             cumsum = 0
#             for idx, val in enumerate(fp):
#                 fp[idx] += cumsum
#                 cumsum += val
#             cumsum = 0
#             for idx, val in enumerate(tp):
#                 tp[idx] += cumsum
#                 cumsum += val
#             #print(tp)
#             rec = tp[:]
#             for idx, val in enumerate(tp):
#                 rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
#             #print(rec)
#             prec = tp[:]
#             for idx, val in enumerate(tp):
#                 prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
#             #print(prec)

#             ap, mrec, mprec = voc_ap(rec, prec)
#             sum_AP += ap
#             text = "{0:.3f}%".format(ap*100) + " = " + class_name + " AP  " #class_name + " AP = {0:.2f}%".format(ap*100)

#             rounded_prec = [ '%.3f' % elem for elem in prec ]
#             rounded_rec = [ '%.3f' % elem for elem in rec ]
#             # Write to results.txt
#             results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")

#             print(text)
#             ap_dictionary[class_name] = ap

#         results_file.write("\n# mAP of all classes\n")
#         mAP = sum_AP / num_classes

#         text = "mAP = {:.3f}%, {:.2f} FPS".format(mAP*100, fps)
#         results_file.write(text + "\n")
#         print(text)
        
#         return mAP*100















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
            CLASSES_PATH=YOLO_LG_CLASS_PATH, GT_DIR=VALIDATE_GT_RESULTS_DIR, mAP_PATH=VALIDATE_MAP_RESULT_PATH):
    MIN_OVERLAP = 0.5   #value to define true/false positive
    
    CLASS_NAMES = read_class_names(CLASSES_PATH)
    #Check and create folder to store ground truth and mAP result
    ground_truth_dir_path = GT_DIR
    if not os.path.exists("YOLOv4-for-studying/mAP"):
        os.mkdir("YOLOv4-for-studying/mAP")
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
            print("\nCompleted calculating for class '{}' with AP = {:0.4f}\n".format(class_name, ap))

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
    weights_file = YOLO_V4_LG_WEIGHTS
    if COCO_VALIDATATION:
        weights_file = YOLO_V4_COCO_WEIGHTS
    
    if USE_CUSTOM_WEIGHTS:
        if COCO_VALIDATATION:
            yolo = YOLOv4_Model(input_size=YOLO_INPUT_SIZE, CLASSES_PATH=YOLO_COCO_CLASS_PATH)
            load_yolov4_weights(yolo, weights_file)
            testset = Dataset('test', TEST_INPUT_SIZE=YOLO_INPUT_SIZE)
            get_mAP(yolo, testset, score_threshold=VALIDATE_SCORE_THRESHOLD, iou_threshold=VALIDATE_IOU_THRESHOLD, TEST_INPUT_SIZE=YOLO_INPUT_SIZE,
                    CLASSES_PATH=YOLO_COCO_CLASS_PATH, GT_DIR=VALIDATE_GT_RESULTS_DIR, mAP_PATH=VALIDATE_MAP_RESULT_PATH)
        else:
            yolo = YOLOv4_Model(input_size=YOLO_INPUT_SIZE, CLASSES_PATH=YOLO_LG_CLASS_PATH)
            yolo.load_weights(weights_file) # use custom weights
            testset = Dataset('test', TEST_INPUT_SIZE=YOLO_INPUT_SIZE)
            get_mAP(yolo, testset, TEST_INPUT_SIZE=YOLO_INPUT_SIZE, CLASSES_PATH=YOLO_LG_CLASS_PATH, score_threshold=VALIDATE_SCORE_THRESHOLD, iou_threshold=VALIDATE_IOU_THRESHOLD)

    
    



