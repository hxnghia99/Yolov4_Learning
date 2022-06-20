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


#Create format of each sliced_image object including 4 attributes
class SlicedImage:
    def __init__(   self,                           
                    image: np.ndarray,              #cv2 image in Numpy array
                    bboxes,                         #List of [4 coordinates, class_idx]
                    starting_point,                 #[xmin, ymin]
                    predictions = None):                   #List of [4 coordinates, score, classs_idx]
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
                if intersection_area/gt_bbox_area >=min_area_ratio:
                    sliced_image_gt_bbox = np.concatenate([top_left - sliced_image_coordinates[:2], bottom_right - sliced_image_coordinates[:2], np.array([original_gt_bbox[4]])])  #minus starting point
                    sliced_image_gt_bboxes.append(sliced_image_gt_bbox)
        return sliced_image_gt_bboxes


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
       
        sliced_image_coordinates_list = self.get_sliced_image_coordinates(*[original_image_width, original_image_height], *[slice_width, slice_height], *[overlap_width_ratio, overlap_height_ratio])
        
        
        number_images = 0
        # iterate over slices
        for sliced_image_coordinates in sliced_image_coordinates_list:
            # count number of sliced images
            
            # Extract starting point of the sliced image
            starting_point = [sliced_image_coordinates[0], sliced_image_coordinates[1]]
            # Extract sliced image
            tl_x, tl_y, br_x, br_y = sliced_image_coordinates
            sliced_image = np.copy(original_image[tl_y:br_y, tl_x:br_x])
            # Extract gt bboxes
            sliced_image_gt_bboxes = self.process_gt_bboxes_to_sliced_image(np.copy(original_gt_bboxes), sliced_image_coordinates, min_area_ratio)

            if len(sliced_image_gt_bboxes) != 0:
                number_images += 1
                sliced_image_obj = SlicedImage(sliced_image, sliced_image_gt_bboxes, starting_point)
                self.sliced_image_list.append(sliced_image_obj)
                
                    
                if self.testing:
                    sliced_image = draw_bbox(sliced_image, sliced_image_gt_bboxes, YOLO_CLASS_PATH, show_label=False)
            if self.testing:
                cv2.imshow("test", sliced_image)
                if cv2.waitKey() == 'q':
                    pass
                cv2.destroyAllWindows()
                print("OK!")
       
        return number_images


    def load_sliced_images(self):
        #Load sliced images into sliced_image_list
        number_sliced_images = self.slice_image(self.original_image, self.original_bboxes, *self.sliced_image_size, *self.overlap_ratio, self.min_area_ratio)
        #extract images and gt bboxes
        sliced_images = []
        sliced_images_gt_bboxes = []
        for i in range(number_sliced_images):
            sliced_images.append(self.sliced_image_list[i].image)
            sliced_images_gt_bboxes.append(self.sliced_image_list[i].bboxes)
        return [sliced_images, sliced_images_gt_bboxes]

    def load_sliced_images_for_export(self):
        #Load sliced images into sliced_image_list
        self.slice_image(self.original_image, self.original_bboxes, *self.sliced_image_size, *self.overlap_ratio, self.min_area_ratio)
        return self.sliced_image_list



class Generate_sliced_images_and_annotations:
    def __init__(self, IMAGE_FOLDER, READ_ANNOTATION_PATH):
        self.image_folder = IMAGE_FOLDER
        self.read_annotation_path = READ_ANNOTATION_PATH
        self.save_annotation_path = self.read_annotation_path.split(".txt")[0] + "_slice.txt"
        self.prefix_image_path = "./"


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
                if not text.replace(',','').replace('-','').isnumeric():
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

        if os.path.exists(save_annotation_path):
            os.remove(save_annotation_path)
        
        annotations_list = self.load_annotations(read_annotation_path)
        for annotation in annotations_list:
            #Get data inside annotation
            image_path, bboxes_annotations = annotation
            bboxes = np.array([list(map(int, box.split(','))) for box in bboxes_annotations])
            image = cv2.imread(image_path)
            image_name = image_path.split("/")[-1].split(".")[0]
            sliced_images_obj = Original_Image_Into_Sliced_Images(image, bboxes)
            sliced_image_obj_list = sliced_images_obj.load_sliced_images_for_export()
            
            with open(save_annotation_path, "a") as f:
                bboxes_annotation = [",".join(list(map(str, bbox))) for bbox in bboxes]
                all_info_annotation = prefix_image_path + image_path + " " + " ".join(bboxes_annotation)
                f.write(all_info_annotation + "\n")

            for sliced_image_obj in sliced_image_obj_list:
                self.export_sliced_image_and_annotation(sliced_image_obj, image_name, self.image_folder, self.save_annotation_path)
        print(f"\n Finished slicing and saving for {self.save_annotation_path}! \n")

    def export_sliced_image_and_annotation(self, sliced_image, image_name, image_folder, save_annotation_path):
        prefix_image_path = self.prefix_image_path
        x_tf, y_tf = sliced_image.starting_point
        saved_image_path = prefix_image_path + image_folder + "/" + image_name + "_sp_" + str(x_tf) + "_" + str(y_tf) + ".png"
        cv2.imwrite(saved_image_path, sliced_image.image)
        bbox_annotation = [",".join(list(map(str, bbox))) for bbox in sliced_image.bboxes]
        all_info_annotation = saved_image_path + " " + " ".join(bbox_annotation)
        with open(save_annotation_path, "a") as f:
            f.write(all_info_annotation + "\n")
        


if __name__=="__main__":
    # # Testing
    # text_by_line = './YOLOv4-for-studying/dataset/Visdrone_DATASET/VisDrone2019-DET-train/images/9999965_00000_d_0000023.jpg 682,661,765,695,3 904,567,934,644,3 958,484,1023,558,3 979,286,1013,359,3 818,554,890,584,3 799,494,891,531,3 821,455,890,484,3 813,402,890,432,3 816,358,893,388,3 817,309,892,338,3 825,267,896,299,3 815,227,885,259,3 697,152,773,193,3 810,49,890,86,3 805,10,885,42,3 988,83,1019,156,4 824,190,896,219,4 331,25,369,103,4 800,138,898,183,5 689,287,774,326,3 700,344,777,379,3 697,402,777,432,3 697,502,774,537,3 701,550,765,579,4 544,389,598,586,8 314,625,346,702,3 318,535,351,609,3 321,438,357,515,3 329,333,363,405,3 329,227,356,302,3 317,119,364,213,5 308,733,346,786,3 633,77,647,117,9 633,87,648,106,1 954,69,969,86,0 1003,462,1018,474,0 1041,445,1055,456,0 917,133,928,146,0 740,740,753,761,0 1016,681,1028,695,0 1053,471,1062,483,0'
    # text = text_by_line.split()
    # bboxes = []
    # for t in text:
    #     if not t.replace(',', '').isnumeric():
    #         temp_path   = os.path.relpath(t, RELATIVE_PATH)
    #         temp_path   = os.path.join(PREFIX_PATH, temp_path)
    #         image_path  = temp_path.replace('\\','/')
    #     else:
    #         t = list(map(int, t.split(',')))
    #         bboxes.append(t)
    # image = cv2.imread(image_path)
    # bboxes = np.array(bboxes)
    # test = Original_Image_Into_Sliced_Images(image, bboxes)
    
    IMAGE_FOLDER = ["YOLOv4-for-studying/dataset/Visdrone_DATASET/VisDrone2019-DET-train/images", "YOLOv4-for-studying/dataset/Visdrone_DATASET/VisDrone2019-DET-val/images", "YOLOv4-for-studying/dataset/Visdrone_DATASET/VisDrone2019-DET-test-dev/images"]
    READ_ANNOTATION_FILE =["./YOLOv4-for-studying/dataset/Visdrone_DATASET/train.txt", "./YOLOv4-for-studying/dataset/Visdrone_DATASET/validation.txt", "./YOLOv4-for-studying/dataset/Visdrone_DATASET/test.txt"]
    for i, _ in enumerate(IMAGE_FOLDER):
        Generate_sliced_images_and_annotations(IMAGE_FOLDER[i], READ_ANNOTATION_FILE[i]).slice_images_and_save()

