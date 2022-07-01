#===============================================================#
#                                                               #
#   File name   : YOLOv4_slicing.py                             #
#   Author      : hxnghia99                                     #
#   Created date: May 24th, 2022                                #
#   GitHub      : https://github.com/hxnghia99/Yolov4_Learning  #
#   Description : Slicing Image Technique to augment the data   #
#                                                               #
#===============================================================#


import os
import numpy as np
import cv2
from YOLOv4_config import *
from YOLOv4_utils import *
import time
import matplotlib.pyplot as plt



#Postprocess to merge sliced_prediction into original prediction
class PostprocessPredictions:
    """Utilities for calculating IOU/IOS based match for given ObjectPredictions"""
    def __init__(   self,
                    match_metric: str = "IOS",
                    match_threshold: float = 0.5):
        self.match_threshold = match_threshold
        self.match_metric = match_metric

    def __call__(self, bboxes):
        #First settings
        bboxes = np.array(bboxes)
        diff_classes_in_pred = list(set(bboxes[:, 5]))
        best_bboxes = []
        #Do GREEDY-NMM for each specific class
        for cls in diff_classes_in_pred:
            cls_mask = np.array(bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]
            #Select best bbox of same class for each object in image
            while len(cls_bboxes) > 0:
                max_conf_bbox_idx = np.argmax(cls_bboxes[:, 4])                 #index of best bbox : highest confidence score
                best_bbox = cls_bboxes[max_conf_bbox_idx]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.delete(cls_bboxes, max_conf_bbox_idx, axis=0)   #remove best bbox from list of bboxes

                assert self.match_metric in ['IOU', 'IOS']
                if self.match_metric == "IOU":
                    iou = bboxes_iou_from_minmax(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])  #calculate list of iou between best bbox and other bboxes
                    weight = np.ones(len(iou), dtype=np.float32)   
                    iou_mask = np.array(iou > self.match_threshold)
                    weight[iou_mask] = 0.0 
                if self.match_metric == "IOS":
                    best_bbox[np.newaxis, :4]
                    cls_bboxes[:, :4]
                    intersect_tf = np.maximum(best_bbox[np.newaxis, :2], cls_bboxes[:, :2])
                    intersect_br = np.minimum(best_bbox[np.newaxis, 2:4], cls_bboxes[:, 2:4])
                    intersect_area = np.multiply.reduce(np.maximum(intersect_br - intersect_tf, 0.0), axis=-1)
                    
                    best_bbox_area =  np.multiply.reduce(np.maximum(best_bbox[np.newaxis, 2:4] - best_bbox[np.newaxis, :2], 0.0), axis=-1)
                    cls_bboxes_area = np.multiply.reduce(np.maximum(cls_bboxes[:, 2:4] - cls_bboxes[:, :2], 0.0), axis=-1)
                    bboxes_smaller_area = np.minimum(cls_bboxes_area, best_bbox_area)

                    ios = intersect_area / bboxes_smaller_area
                    weight = np.ones(len(ios), dtype=np.float32)   
                    ios_mask = np.array(ios > self.match_threshold)
                    weight[ios_mask] = 0.0 

                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight    #detele bboxes predicting same objects
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]
        return best_bboxes




#Get image, slice image, make predictions, merge predictions
class PredictionResult:
    def __init__(self, model, image, input_size, sliced_input_size, score_threshold, iou_threshold):
        self.model = model
        self.image = image
        self.input_size = input_size
        self.sliced_input_size = sliced_input_size
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        
        self.postprocess_slicing_predictions_into_original_image = PostprocessPredictions(match_metric="IOS", match_threshold=0.5)


        # self.make_prediciton()

    #get sliced images, make each prediction, scale prediction to original image
    def make_sliced_predictions(self):
        sliced_images_obj = Original_Image_Into_Sliced_Images(np.copy(self.image))
        sliced_images = sliced_images_obj.load_sliced_images_for_export()
        sliced_image_with_prediction_list = []
        for sliced_image in sliced_images:
            image_data = cv2.cvtColor(np.copy(sliced_image.image), cv2.COLOR_BGR2RGB)

            #Create a new model using image original size scaling to 32
            if EVALUATION_DATASET_TYPE == "VISDRONE" and EVALUATE_ORIGINAL_SIZE:
                original_h, original_w, _ = image_data.shape
                self.input_size = [int(np.ceil(original_w/32))*32, int(np.ceil(original_h/32))*32]

            image_data = image_preprocess(image_data, self.sliced_input_size)                  #scale to size 416
            image_data = image_data[np.newaxis, ...].astype(np.float32)                         #reshape [1, 416, 416, 3]
            pred_bbox = self.model(image_data, training=False)
            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]               #reshape to [3, bbox_num, 85]
            pred_bbox = tf.concat(pred_bbox, axis=0)                                            #concatenate to [bbox_num, 85]
            pred_bboxes = postprocess_boxes(pred_bbox, np.copy(sliced_image.image), self.sliced_input_size, self.score_threshold)      #scale to origional and select valid bboxes
            if len(pred_bboxes) == 0:
                continue
            pred_bboxes = tf.convert_to_tensor(nms(pred_bboxes, self.iou_threshold, method='nms'))                                       #Non-maximum suppression: xymin, xymax        
            sliced_image.predictions = pred_bboxes
            sliced_image_with_prediction_list.append(sliced_image)
        return sliced_image_with_prediction_list

    #make prediction in original image, make prediciton in sliced images and merge into original image
    def make_prediciton(self):
        start_time = time.time()
        image_data = cv2.cvtColor(np.copy(self.image), cv2.COLOR_BGR2RGB)
        #Create a new model using image original size scaling to 32
        if EVALUATION_DATASET_TYPE == "VISDRONE" and EVALUATE_ORIGINAL_SIZE:
            original_h, original_w, _ = image_data.shape
            self.input_size = [int(np.ceil(original_w/32))*32, int(np.ceil(original_h/32))*32]

        image_data = image_preprocess(image_data, self.input_size)                  #scale to size 416
        image_data = image_data[np.newaxis, ...].astype(np.float32)                         #reshape [1, 416, 416, 3]
        pred_bbox = self.model(image_data, training=False)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]               #reshape to [3, bbox_num, 85]
        pred_bbox = tf.concat(pred_bbox, axis=0)                                            #concatenate to [bbox_num, 85]
        pred_bboxes = postprocess_boxes(pred_bbox, np.copy(self.image), self.input_size, self.score_threshold)      #scale to origional and select valid bboxes
        pred_bboxes = tf.convert_to_tensor(nms(pred_bboxes, self.iou_threshold, method='nms'))                                       #Non-maximum suppression: xymin, xymax              

        # pred_bboxes = [[0,0,0,0,0,0]]
        image_test = draw_bbox(np.copy(self.image),pred_bboxes, YOLO_CLASS_PATH, show_label=True)
        cv2.imshow("Test before slicing prediciton", cv2.resize(image_test, [1280, 720]))

        sliced_image_with_prediction_list = self.make_sliced_predictions()
        for index, sliced_image_with_prediction in enumerate(sliced_image_with_prediction_list):
            pred_offset = np.concatenate([sliced_image_with_prediction.starting_point, sliced_image_with_prediction.starting_point, [0], [0]], axis=-1)[np.newaxis, :]
            sliced_image_with_prediction.predictions = sliced_image_with_prediction.predictions + pred_offset
            pred_bboxes = np.concatenate([pred_bboxes, sliced_image_with_prediction.predictions], axis=0)
            pred_bboxes = self.postprocess_slicing_predictions_into_original_image(pred_bboxes)

            image_test = draw_bbox(np.copy(self.image),pred_bboxes, YOLO_CLASS_PATH, show_label=True)
            #draw bbox to image
            (x1, y1), (x2, y2) = (sliced_image_with_prediction.starting_point[0], sliced_image_with_prediction.starting_point[1]), (sliced_image_with_prediction.starting_point[0] + SLICED_IMAGE_SIZE[0], sliced_image_with_prediction.starting_point[1] + SLICED_IMAGE_SIZE[1])
            cv2.rectangle(image_test, (x1,y1), (x2, y2), (255,0,0), 5)

            cv2.imwrite("Images/Test for slicing image " + str(index) + ".png", image_test)
            cv2.imshow("Test for slicing image " + str(index) , cv2.resize(image_test, [1280, 720]))
            if cv2.waitKey() == "q":
                pass
            cv2.destroyAllWindows()

        end_time = time.time() - start_time
        
        # image_test = draw_bbox(np.copy(self.image),pred_bboxes, YOLO_CLASS_PATH, show_label=True)
        # cv2.imshow("Test after slicing prediciton", cv2.resize(image_test, [1280, 720]))
        # if cv2.waitKey() == "q":
        #     pass
        # cv2.destroyAllWindows()
        return [pred_bboxes, end_time]







#Create format of each sliced_image object including 4 attributes
class SlicedImage:
    def __init__(   self,                           
                    image: np.ndarray,                      #cv2 image in Numpy array
                    bboxes = None,                          #List of [4 coordinates, class_idx]
                    starting_point: list =[int, int],       #[xmin, ymin]
                    predictions = None):                    #List of [4 coordinates, score, classs_idx]
        self.image = image
        self.bboxes = bboxes
        self.starting_point = starting_point
        self.predictions = predictions

#Create format of object processed for each original image
class Original_Image_Into_Sliced_Images:
    def __init__(self, original_image=None, original_bboxes=None, TESTING=False):              #inputs as original image and all gt bboxes inside that image
        #Setting for slicing original image into set of sliced images
        self.original_image = original_image
        self.original_bboxes = original_bboxes
        self.original_image_height = self.original_image.shape[0]
        self.original_image_width = self.original_image.shape[1]
        self.sliced_image_size = SLICED_IMAGE_SIZE
        self.overlap_ratio = OVERLAP_RATIO
        self.min_area_ratio = MIN_AREA_RATIO
        
        #List of sliced images
        self.sliced_image_list = []

        self.testing = TESTING
        if self.testing:
            self.slice_image(self.original_image, self.original_bboxes, *self.sliced_image_size, *self.overlap_ratio, self.min_area_ratio)

    #Get the bbox coordinate of sliced images in origional image
    def get_sliced_image_coordinates(   self,
                                        image_width: int,                   
                                        image_height: int,
                                        slice_width: int,
                                        slice_height: int,
                                        overlap_width_ratio: float,
                                        overlap_height_ratio: float):  
        sliced_image_coordinates = []
        x_overlap = int(overlap_width_ratio * slice_width)
        y_overlap = int(overlap_height_ratio * slice_height)
        #Run in y-axis, at each value y, calculate x
        y_max = y_min = 0
        while y_max < image_height:    
            #update new ymax for this iterative
            y_max = y_min + slice_height
            #run in x-axis, at each value (xmin,xmax), save the patch coordinates
            x_min = x_max = 0
            while x_max < image_width:
                #update new xmax for this iterative
                x_max = x_min + slice_width
                #if the patch coordinates is outside original image, cut at the borders to inside area inversely
                if y_max > image_height or x_max > image_width:
                    xmax = min(image_width, x_max)
                    ymax = min(image_height, y_max)
                    xmin = max(0, xmax - slice_width)   
                    ymin = max(0, ymax - slice_height)
                    sliced_image_coordinates.append([xmin, ymin, xmax, ymax])
                else:
                    sliced_image_coordinates.append([x_min, y_min, x_max, y_max])
                #update new xmin for next iterative
                x_min = x_max - x_overlap
            #update new ymin for next iterative
            y_min = y_max - y_overlap
        return sliced_image_coordinates


    #check if gt_coordinates is inside the sliced image
    def check_gt_coordinates_inside_slice(  self, 
                                            gt_coordinates,                 #format [xmin, ymin, xmax, ymax]                   
                                            sliced_image_coordinates):      #format [xmin, ymin, xmax, ymax]                   
        if gt_coordinates[0] >= sliced_image_coordinates[2]:    #if gt is left to sliced_image
            return False    
        if gt_coordinates[1] >= sliced_image_coordinates[3]:    #if gt is below sliced_image
            return False
        if gt_coordinates[2] <= sliced_image_coordinates[0]:    #if gt is right to sliced_image
            return False        
        if gt_coordinates[3] <= sliced_image_coordinates[1]:    #if gt is above sliced_image
            return False
        return True


    #Tranform gt_bboxes in original image into those in sliced images
    def process_gt_bboxes_to_sliced_image(  self, 
                                            original_gt_bboxes,                         #List of gt bboxes with format [4 coordinates, class_idx]
                                            sliced_image_coordinates,                   #format [xmin, ymin, xmax, ymax]
                                            min_area_ratio):                            #area ratio to remove gt bbox from sliced image
        #Each ground truth bbox is compared to sliced_image_coordinates to create bbox_coordinates inside sliced_image
        sliced_image_gt_bboxes = []
        for original_gt_bbox in original_gt_bboxes:
            if self.check_gt_coordinates_inside_slice(original_gt_bbox[:4], sliced_image_coordinates):
                #Calculate intersection area
                top_left        = np.maximum(original_gt_bbox[:2], sliced_image_coordinates[:2])
                bottom_right    = np.minimum(original_gt_bbox[2:4], sliced_image_coordinates[2:])
                gt_bbox_area = np.multiply.reduce(original_gt_bbox[2:4] - original_gt_bbox[:2])
                intersection_area = np.multiply.reduce(bottom_right - top_left)
                if intersection_area/gt_bbox_area >= min_area_ratio:
                    sliced_image_gt_bbox = np.concatenate([top_left - sliced_image_coordinates[:2], bottom_right - sliced_image_coordinates[:2], np.array([original_gt_bbox[4]])])  #minus starting point
                    sliced_image_gt_bboxes.append(sliced_image_gt_bbox)
        if len(sliced_image_gt_bboxes) != 0:
            return sliced_image_gt_bboxes
        else:
            return np.array([[0,0,0,0,-1]])


    #slice the original image into objects of class SliceImage
    def slice_image(self, 
                    original_image,                 #original image
                    original_gt_bboxes,             #list of original bboxes with shape [4 coordinates, class_idx]
                    slice_width,                
                    slice_height,
                    overlap_width_ratio,
                    overlap_height_ratio,
                    min_area_ratio):

        original_image_height, original_image_width, _ = original_image.shape
        if not (original_image_width != 0 and original_image_height != 0):
            raise RuntimeError(f"Error from invalid image size: {original_image.shape}")
       
        # 1) Get cooridnates of sliced images       
        sliced_image_coordinates_list = self.get_sliced_image_coordinates(*[original_image_width, original_image_height], *[slice_width, slice_height], *[overlap_width_ratio, overlap_height_ratio])
        
        # 2) Slice image according to sliced coordinates
        number_images = 0
        # iterate over slices
        for sliced_image_coordinates in sliced_image_coordinates_list:
            # Extract starting point of the sliced image
            starting_point = [sliced_image_coordinates[0], sliced_image_coordinates[1]]
            # Extract sliced image
            tl_x, tl_y, br_x, br_y = sliced_image_coordinates
            sliced_image = np.copy(original_image[tl_y:br_y, tl_x:br_x])
            
            # 3) Slice bboxes according to sliced coordinates
            if original_gt_bboxes is not None:   
                # Extract gt bboxes
                sliced_image_gt_bboxes = self.process_gt_bboxes_to_sliced_image(np.copy(original_gt_bboxes), sliced_image_coordinates, min_area_ratio)  #at least 1 bbox
                sliced_image_obj = SlicedImage(sliced_image, sliced_image_gt_bboxes, starting_point)
                self.sliced_image_list.append(sliced_image_obj)
                    
                if self.testing:
                    sliced_image = draw_bbox(sliced_image, sliced_image_gt_bboxes, YOLO_CLASS_PATH.replace(".txt","_test.txt"), show_label=True)
                    cv2.imshow("test", sliced_image)
                    if cv2.waitKey() == 'q':
                        pass
                    cv2.destroyAllWindows()
                    print("OK!")
            #No original_bboxes input : inference time for slicing prediction
            else:
                sliced_image_obj = SlicedImage(sliced_image, None, starting_point)
                self.sliced_image_list.append(sliced_image_obj)

            #count the number of sliced images for each image
            number_images += 1
        return number_images


    def load_sliced_images_for_export(self):
        #Load sliced images into sliced_image_list
        num_sliced_images = self.slice_image(self.original_image, self.original_bboxes, *self.sliced_image_size, *self.overlap_ratio, self.min_area_ratio)
        return self.sliced_image_list




class Generate_sliced_images_and_annotations:
    def __init__(self, IMAGE_FOLDER, READ_ANNOTATION_PATH):
        self.image_folder = IMAGE_FOLDER
        self.read_annotation_path = READ_ANNOTATION_PATH
        self.save_annotation_path = self.read_annotation_path.split(".txt")[0] + "_slice.txt"
        self.prefix_image_path = "./"
        if not os.path.exists(self.image_folder + "_slice"):
                os.mkdir(self.image_folder + "_slice")

    #Get the list of annotations for each line
    def load_annotations(self, annotation_path):
        with open(annotation_path, 'r') as f:
            all_texts_by_line = f.read().splitlines()
            #Ensure that image has objects
            annotations = [text_by_line.strip() for text_by_line in all_texts_by_line if len(text_by_line.strip().split()[1:]) != 0]
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
            else:
                final_annotations.append([image_path, bboxes_annotations])
        return final_annotations        

    
    def slice_images_and_save(self):
        read_annotation_path = self.read_annotation_path
        save_annotation_path = self.save_annotation_path
        prefix_image_path = self.prefix_image_path
        image_folder = self.image_folder

        if os.path.exists(save_annotation_path):        #remove old annotation file
            os.remove(save_annotation_path)
        
        annotations_list = self.load_annotations(read_annotation_path)
        # maxe = []
        # max_count = 0
        # image_path_test = []
        # num_bbox_test = []
        excessive_image_num = 0
        for index, annotation in enumerate(annotations_list):
            print("Finished slicing and saving image ", index)
            #Get data inside annotation
            image_path, bboxes_annotations = annotation
            bboxes = np.array([list(map(int, box.split(','))) for box in bboxes_annotations])
            image = cv2.imread(image_path)
            image_name = image_path.split("/")[-1].split(".")[0]
            sliced_images_obj = Original_Image_Into_Sliced_Images(image, bboxes)
            sliced_image_obj_list = sliced_images_obj.load_sliced_images_for_export()
            
            
        #     num_bbox = [len(x.bboxes) for x in sliced_image_obj_list]
        #     maxe.extend(num_bbox)

        #     if np.max(num_bbox) >= 200: 
        #         image_path_test.append(image_path)
        #         num_bbox_test.append(num_bbox)

        # print(maxe)
        # print(image_path_test)
        # print(num_bbox_test)
        # unique, count = np.unique(maxe, return_counts=True)
        # count = count + 10

        # fig = plt.figure()
        # plt.bar(unique, count, color="green", width=1)
        # plt.xlabel("Number of bboxes per image")
        # plt.ylabel("Number of image")
        # plt.show()

            with open(save_annotation_path, "a") as f:
                if len(bboxes) <= 150:
                    bboxes_annotation = [",".join(list(map(str, bbox))) for bbox in bboxes]
                    all_info_annotation = prefix_image_path + image_path + " " + " ".join(bboxes_annotation)
                    f.write(all_info_annotation + "\n")
            
            for sliced_image_obj in sliced_image_obj_list:
                excessive_image_num += self.export_sliced_image_and_annotation(sliced_image_obj, image_name, image_folder, save_annotation_path)
        print(f"\n Finished slicing and saving for {self.save_annotation_path}! \n")
        return excessive_image_num


    def export_sliced_image_and_annotation(self, sliced_image, image_name, image_folder, save_annotation_path):
        
        bbox_annotation = [",".join(list(map(str, bbox))) for bbox in sliced_image.bboxes]
        if len(bbox_annotation) <= 150:
            prefix_image_path = self.prefix_image_path
            x_tf, y_tf = sliced_image.starting_point
            
            saved_image_path = prefix_image_path + image_folder + "_slice" + "/" + image_name + "_sp_" + str(x_tf) + "_" + str(y_tf) + ".png"
            cv2.imwrite(saved_image_path, sliced_image.image)
            all_info_annotation = saved_image_path + " " + " ".join(bbox_annotation)
            with open(save_annotation_path, "a") as f:
                f.write(all_info_annotation + "\n")
            return 0
        else:
            return 1
        


if __name__=="__main__":
    # Testing
    # text_by_line = './YOLOv4-for-studying/dataset/Visdrone_DATASET/VisDrone2019-DET-train/images/0000182_01220_d_0000039.jpg 413,60,490,150,-1 516,134,546,158,-1 496,125,509,133,3 490,144,503,151,-1 536,169,551,185,-1 621,210,652,225,3 636,230,641,247,0 632,219,663,236,3 646,225,674,239,3 646,235,680,250,3 699,310,713,336,9 703,308,712,328,1 688,325,698,350,0 678,317,687,344,0 646,271,654,289,9 646,268,653,284,1 741,447,765,480,9 747,446,759,471,1 659,415,691,461,7 607,301,616,325,0 649,374,663,396,9 651,367,663,388,1 514,301,574,329,3 630,317,636,335,9 631,313,637,331,1 610,288,620,305,9 613,286,620,302,1 487,163,517,178,4 527,179,533,188,6 518,191,530,208,7 547,205,552,217,9 548,203,552,215,1 523,168,527,175,9 524,167,528,174,1 510,202,515,212,9 510,199,516,209,1 415,159,421,170,-1 430,189,434,199,9 431,187,434,198,1 444,183,449,191,9 444,180,448,190,1 345,274,360,293,9 226,274,238,286,9 233,266,245,275,9 174,318,237,351,3 331,282,354,318,7 286,437,313,463,9 235,514,285,539,9 399,235,406,248,9 399,232,405,245,1 412,245,417,260,9 383,286,393,303,9 372,399,384,428,9 412,240,417,256,1 384,282,392,298,1 371,392,384,417,1 373,399,383,422,1 373,396,384,420,1'
    # text = text_by_line.split()
    # bboxes = []
    # for t in text:
    #     if not t.replace(',', '').replace("-1","").isnumeric():
    #         temp_path   = os.path.relpath(t, RELATIVE_PATH)
    #         temp_path   = os.path.join(PREFIX_PATH, temp_path)
    #         image_path  = temp_path.replace('\\','/')
    #     else:
    #         t = list(map(int, t.split(',')))
    #         bboxes.append(t)
    # image = cv2.imread(image_path)
    # bboxes = np.array(bboxes)
    # test = Original_Image_Into_Sliced_Images(image, bboxes, TESTING=True)
    
    # IMAGE_FOLDER = "YOLOv4-for-studying/dataset/Visdrone_DATASET/VisDrone2019-DET-train/images"
    # READ_ANNOTATION_FILE = "./YOLOv4-for-studying/dataset/Visdrone_DATASET/train.txt"
    # Generate_sliced_images_and_annotations(IMAGE_FOLDER, READ_ANNOTATION_FILE).slice_images_and_save()



    IMAGE_FOLDER = ["YOLOv4-for-studying/dataset/Visdrone_DATASET/VisDrone2019-DET-train/images", "YOLOv4-for-studying/dataset/Visdrone_DATASET/VisDrone2019-DET-val/images"]
    READ_ANNOTATION_FILE =["./YOLOv4-for-studying/dataset/Visdrone_DATASET/train.txt", "./YOLOv4-for-studying/dataset/Visdrone_DATASET/validation.txt"]
    for i, _ in enumerate(IMAGE_FOLDER):
        failed_image_num = Generate_sliced_images_and_annotations(IMAGE_FOLDER[i], READ_ANNOTATION_FILE[i]).slice_images_and_save()
        print(i, " : Number of images that have more 150 objects = ", failed_image_num)

