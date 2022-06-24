#===============================================================#
#                                                               #
#   File name   : YOLOv4_utils.py                               #
#   Author      : hxnghia99                                     #
#   Created date: May 24th, 2022                                #
#   GitHub      : https://github.com/hxnghia99/Yolov4_Learning  #
#   Description : YOLOv4 network utilizations                   #
#                                                               #
#===============================================================#


import tensorflow as tf
import numpy as np
import cv2
from YOLOv4_config import *
from YOLOv4_model import *
import colorsys

'''##############################################
input:  the path of class file
output: enumerate of class names
obj:    read class name datas in coco file
##############################################'''
def read_class_names(class_file_name):
    # loads class name from a file
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


'''####################################################################
input: (2) model, weight file
output: model with pretrained weights
obj:    load pretrained weights into the model (just apply for YOLOv4)
######################################################################'''
#Function to load trained parameters into YOLOv4 model
def load_yolov4_weights(model, weights_file):
    tf.keras.backend.clear_session() # used to reset layer names
    # Layer quantity in YOLOv4 Model
    range1 = 110                        #Total of layers
    range2 = [93, 101, 109]             #Index of output layers
    with open(weights_file, 'rb') as wf:
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
        j = 0
        for i in range(range1):

            if i == 78:
                print(" \n Finished loading weights of CSPDarknet53 + SPP block ... \n")
                break

            #Get name of convolutional layer
            if i > 0: conv_layer_name = 'conv2d_%d' %i
            else: conv_layer_name = 'conv2d'
            #Get name of bn layer 
            if j > 0: bn_layer_name = 'batch_normalization_%d' %j
            else: bn_layer_name = 'batch_normalization'
            conv_layer = model.get_layer(conv_layer_name)
            filters = conv_layer.filters                        # number of filters: output dimensions
            k_size = conv_layer.kernel_size[0]                  # kernel_size
            in_dim = conv_layer.input_shape[-1]                 # input dimensions
            
            #Get bn weights or bias from file
            if i not in range2:
                # bn weights from file:  [beta, gamma, mean, variance]
                bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)           
                # bn weights in model:       [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]                   #swap rows
                bn_layer = model.get_layer(bn_layer_name)
                j += 1
            else:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            
            #Get convolutional weights from file
            # shape in file (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
            # shape in model (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if i not in range2:
                conv_layer.set_weights([conv_weights])
                bn_layer.set_weights(bn_weights)
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

        # assert len(wf.read()) == 0, 'failed to read all data'



'''####################################################################
input:  None
output: YOLOv4 model
obj:    select GPU, create YOLOv3 model and load pretrained weights
######################################################################'''
#Config using GPU and create YOLOv3_Model with loaded parameters
def Load_YOLOv4_Model():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        print(f'GPUs {gpus}')
        try: tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError: pass
    yolo = YOLOv4_Model(CLASSES_PATH=YOLO_CLASS_PATH)
    if USE_LOADED_WEIGHT:
        YOLOv4_weights = PREDICTION_WEIGHT_FILE
        if TRAINING_DATASET_TYPE == "COCO":
            load_yolov4_weights(yolo, YOLOv4_weights)
        else:
            yolo.load_weights(YOLOv4_weights) 
        print("Loading Darknet_weights from:", YOLOv4_weights)
    return yolo

'''##################################
input: (3) image, target_size, gt_boxes(opt)
output: new image padded the resized old image 
obj:    create image to put into YOLO model
##################################'''
def image_preprocess(image, target_size, gt_boxes=None):
    target_size_w, target_size_h = target_size
    image_h, image_w, _ = image.shape   
    resize_ratio = min(target_size_w/image_w, target_size_h/image_h)                      #resize ratio of the larger coordinate into 416
    new_image_w, new_image_h = int(resize_ratio*image_w), int(resize_ratio*image_h)
    image_resized = cv2.resize(image, (new_image_w, new_image_h))                     #the original image is resized into 416 x smaller coordinate

    image_padded = np.full(shape=[target_size_h, target_size_w, 3], fill_value=128.0)
    dw, dh = (target_size_w - new_image_w) // 2, (target_size_h - new_image_h) // 2
    image_padded[dh:new_image_h+dh, dw:new_image_w+dw] = image_resized                #pad the resized image into image_padded
    image_padded = image_padded/255.0

    if gt_boxes is None:
        return image_padded

    else: #gt_boxes have shape of [xmin, ymin, xmax, ymax]
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * resize_ratio + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * resize_ratio + dh
        return image_padded, gt_boxes
    


'''#################################################################################
input: (4)  predictions, original image, input size, confidence score threshold
output: list of valid bboxes, each contains 6 elements: 4 coordinates, score, class
obj:    rescale predictions into size of original image and remove invalid bboxes
##################################################################################'''
def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold, use_sliced_image=None):
    original_scale = np.sqrt(original_image.shape[0]*original_image.shape[1])
    valid_scale =   [0, original_scale]
    pred_bbox   =   np.array(pred_bbox)
    num_bbox    =   len(pred_bbox)
    #divide prediction of shape [..., 85] into [..., 4], [..., 1], [..., 80]
    pred_xywh = pred_bbox[:, 0:4]   
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]
    # (x, y, w, h) --> (xmin, ymin, xmax, ymax) : size 416 x 416
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # prediction (xmin, ymin, xmax, ymax) -> prediction (xmin_org, ymin_org, xmax_org, ymax_org)
    org_image_h, org_image_w = original_image.shape[:2]
    resize_ratio = min(input_size[0] / org_image_w, input_size[1] / org_image_h)
    dw = (input_size[0] - resize_ratio * org_image_w) / 2                      #pixel position recalculation
    dh = (input_size[1] - resize_ratio * org_image_h) / 2
    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio     #(pixel_pos - dw)/resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio
    # constrain the bbox inside image and set invalid box to 0
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),                                       #check if xmin, ymin <= 0, get 0 instead
                                np.minimum(pred_coor[:, 2:], [org_image_w - 1, org_image_h - 1])], axis=-1) #check if xmax, ymax > border of origional image, get border instead
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))  #get mask of invalid bbox: xmin > xmax or ymin > ymax
    pred_coor[invalid_mask] = 0                                                                             #the invalid box will be 0
    # mask for valid bboxes: having valid area and being inside image
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))
    # combine valid mask with high scores mask to discard boxes
    classes = np.argmax(pred_prob, axis=-1)                         #shape [num_box]
    scores = pred_conf * pred_prob[np.arange(num_bbox), classes]    #shape [num_box]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]   #length = num_True_box
    #output shape [num_True_box, 6]
    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1) 


'''###############################################################################
input:  (2) 1 bbox or list of bboxes, list of bboxes
output: list of IoU
obj:    Use normal IoU method to calculate IoU between 1 bbox and list of bboxes
from (Xmin, Ymin, Xmax, Ymax)
################################################################################'''
def bboxes_iou_from_minmax(boxes1, boxes2):
    #area of bboxes1 and bboxes2
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    #coordinates of intersection
    inters_top_left     = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    inters_bottom_right = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
    #area of intersection and union
    intersection = tf.maximum(inters_bottom_right - inters_top_left, 0.)
    intersection_area = tf.math.reduce_prod(intersection, axis=-1)
    union_area = boxes1_area + boxes2_area - intersection_area
    #ious for list of bboxes
    ious = intersection_area / union_area
    return ious


'''###############################################################################
input:  (2) 1 bbox or list of bboxes, list of bboxes
output: list of IoU
obj:    Use normal IoU method to calculate IoU between 1 bbox and list of bboxes
from XYWH
################################################################################'''
def bboxes_iou_from_xywh(boxes1, boxes2):
    #convert xywh to minmax
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
    #calculate IOU
    ious = bboxes_iou_from_minmax(boxes1, boxes2)
    return ious


'''###############################################################################
input:  (2) 1 bbox or list of bboxes, list of bboxes
output: list of gIoU
obj:    Use generalized IoU method to calculate IoU between 1 bbox and list of bboxes
from (Xmin, Ymin, Xmax, Ymax)
################################################################################'''
def bboxes_giou_from_minmax(boxes1, boxes2):
    #area of bboxes1 and bboxes2
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    #coordinates of intersection
    inters_top_left     = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    inters_bottom_right = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
    #area of intersection and union
    intersection = tf.maximum(inters_bottom_right - inters_top_left, 0.)
    intersection_area = tf.math.reduce_prod(intersection, axis=-1)
    union_area = boxes1_area + boxes2_area - intersection_area
    #ious for list of bboxes
    ious = intersection_area / union_area
    #enclose area
    enclose_top_left    = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_bottom_right= tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose             = tf.maximum(enclose_bottom_right - enclose_top_left, 0.0)
    enclose_area        = tf.math.reduce_prod(enclose, axis=-1)
    #gious for list of bboxes
    gious = ious - 1.0 * (enclose_area - union_area) / enclose_area
    return gious


'''###############################################################################
input:  (2) 1 bbox or list of bboxes, list of bboxes
output: list of gIoU
obj:    Use generalized IoU method to calculate IoU between 1 bbox and list of bboxes
from XYWH
################################################################################'''
def bboxes_giou_from_xywh(boxes1, boxes2):
    #convert xywh to minmax
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
    #calculate gious from minmax
    gious = bboxes_giou_from_minmax(boxes1, boxes2)
    return gious




def bboxes_ciou_from_xywh(boxes1, boxes2):
    #convert xywh to minmax
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
    #calculate gious from minmax
    cious = bboxes_ciou_from_minmax(boxes1, boxes2)
    return cious

def bboxes_ciou_from_minmax(boxes1, boxes2):
    #iou
    ious = bboxes_iou_from_minmax(boxes1, boxes2)
    #center
    xy1 = (boxes1[...,0:2] + boxes1[...,2:])/2
    xy2 = (boxes2[...,0:2] + boxes2[...,2:])/2
    d_center = tf.reduce_sum(tf.square(xy1 - xy2), axis=-1)
    #enclose area
    enclose_top_left    = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_bottom_right= tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    d_enclose = tf.reduce_sum(tf.square(enclose_bottom_right - enclose_top_left), axis=-1)
    #wh
    wh1 = boxes1[...,2:] - boxes1[...,0:2]
    wh2 = boxes2[...,2:] - boxes2[...,0:2]
    
    w1_mask = tf.cast(tf.zeros(tf.shape(wh1[...,0:1])), tf.bool)
    h1_mask = (wh1[...,1:] == 0.0) 
    wh1_mask = tf.cast(tf.concat([w1_mask, h1_mask], axis=-1), tf.float32) * tf.constant(1e-10)
    wh1 = wh1 + wh1_mask

    w2_mask = tf.cast(tf.zeros(tf.shape(wh2[...,0:1])), tf.bool)
    h2_mask = (wh2[...,1:] == 0) 
    wh2_mask = tf.cast(tf.concat([w2_mask, h2_mask], axis=-1), tf.float32) * tf.constant(1e-10)
    wh2 = wh2 + wh2_mask

    v = 4 * (tf.math.atan(wh1[...,0] / wh1[...,1]) - tf.math.atan(wh2[...,0] / wh2[...,1]))**2 / (np.pi ** 2)
    alpha = v / (1 - ious + v)
    #ciou    
    cious = ious - d_center / d_enclose - alpha * v
    return cious




def nms_center_d(boxes1, boxes2):
    boxes1_center = (boxes1[...,:2] + boxes1[...,2:])*0.5
    boxes2_center = (boxes2[...,:2] + boxes2[...,2:])*0.5
    center_d = np.sqrt(np.sum(np.square(boxes1_center - boxes2_center), axis=-1))
    return center_d

'''##################################
input:  bboxes as (xmin, ymin, xmax, ymax, score, class), Iou threshold, sigma, method
output: list of best bboxes for each object
obj:    Use nms (or soft-nms) to eliminate the bboxes with lower confidence score
##################################'''
def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    #First settings
    bboxes = np.array(bboxes)
    diff_classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []
    #Do nms for each specific class
    for cls in diff_classes_in_img:
        cls_mask = np.array(bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        #Select best bbox of same class for each object in image
        while len(cls_bboxes) > 0:
            max_conf_bbox_idx = np.argmax(cls_bboxes[:, 4])                 #index of best bbox
            best_bbox = cls_bboxes[max_conf_bbox_idx]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.delete(cls_bboxes, max_conf_bbox_idx, axis=0)   #remove best bbox from list of bboxes
            iou = bboxes_iou_from_minmax(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])  #calculate list of iou between best bbox and other bboxes
            
            if USE_NMS_CENTER_D:
                """ TESTING """
                center_d = nms_center_d(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                """#########"""

            weight = np.ones(len(iou), dtype=np.float32)                    
            assert method in ['nms', 'soft-nms']
            if method == 'nms':
                iou_mask = np.array(iou > iou_threshold)
                if USE_NMS_CENTER_D:
                    center_d_mask = np.array(center_d < 5)
                    iou_mask = np.logical_or(iou_mask, center_d_mask)
                weight[iou_mask] = 0.0                      #mask to detele bboxes predicting same objects          
            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou**2 / sigma))    #bigger iou -> smaller weight
            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight    #detele bboxes predicting same objects
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]
    return best_bboxes




'''##################################
input: (8) image, bboxes as (coordinates, score, class), class name path
output: image with bboxes and labels
obj:    add bboxes and labels to original image
##################################'''
def draw_bbox(image, bboxes, CLASSES_PATH=YOLO_COCO_CLASS_PATH, show_label=True, show_confidence=True, Text_colors='', rectangle_colors='', tracking=False):
    #Initial readings
    CLASS_NAMES = read_class_names(CLASSES_PATH)
    num_classes = len(CLASS_NAMES)
    image_h, image_w, _ = image.shape
    bboxes = np.array(bboxes)
    #generate random color for bboxes and labels
    rectangle_hsv_tuples     = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    label_hsv_tuples         = [(1.0 * x / num_classes, 1., 1.) for x in range(int(num_classes/2), num_classes)] 
    label_hsv_tuples        += [(1.0 * x / num_classes, 1., 1.) for x in range(0, int(num_classes/2))]                                
    rand_rectangle_colors    = list(map(lambda x: colorsys.hsv_to_rgb(*x), rectangle_hsv_tuples))
    rand_rectangle_colors    = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), rand_rectangle_colors))
    rand_text_colors         = list(map(lambda x: colorsys.hsv_to_rgb(*x), label_hsv_tuples))
    rand_text_colors         = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), rand_text_colors))
    #draw each bbox and label
    if (bboxes.shape[1] == 6):      #draw predicted bboxes
        for bbox in bboxes:
            coordinates = np.array(bbox[:4], dtype=np.int32)
            score = bbox[4]
            class_id = int(bbox[5])
            #select color
            bbox_color = rectangle_colors if rectangle_colors != '' else rand_rectangle_colors[class_id]
            label_color = Text_colors if Text_colors != ''  else rand_text_colors[class_id]
            #calculate thickness and fontSize
            bbox_thick = int(0.6 * (image_h + image_w) / 1000)
            if bbox_thick < 1: bbox_thick = 1
            fontScale = 0.75 * bbox_thick
            #draw bbox to image
            (x1, y1), (x2, y2) = (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3])
            cv2.rectangle(image, (x1,y1), (x2, y2), bbox_color, bbox_thick * 2)
            #draw label to image
            if show_label:
                score_str = " {:.2f}".format(score) if show_confidence else ""
                try:
                    label = "{}".format(CLASS_NAMES[class_id]) + score_str
                except KeyError:
                    print("You received KeyError")
                #draw filled rectangle and add text to this
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=fontScale, thickness=bbox_thick)
                cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)
                cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=fontScale, color=label_color, thickness=bbox_thick, lineType=cv2.LINE_AA)
    elif (bboxes.shape[1] == 5):       #draw ground truth bboxes
        for bbox in bboxes:
            coordinates = np.array(bbox[:4], dtype=np.int32)
            class_id = int(bbox[4])
            #select color
            bbox_color = rectangle_colors if rectangle_colors != '' else rand_rectangle_colors[class_id]
            label_color = Text_colors if Text_colors != ''  else rand_text_colors[class_id]
            #calculate thickness and fontSize
            bbox_thick = int(0.6 * (image_h + image_w) / 1000)
            if bbox_thick < 1: bbox_thick = 1
            fontScale = 0.75 * bbox_thick
            #draw bbox to image
            (x1, y1), (x2, y2) = (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3])
            cv2.rectangle(image, (x1,y1), (x2, y2), bbox_color, bbox_thick * 2)
            #draw label to image
            if show_label:
                try:
                    label = "{}".format(CLASS_NAMES[class_id if class_id !=-1 else class_id+12])
                except KeyError:
                    print("You received KeyError")
                #draw filled rectangle and add text to this
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=fontScale, thickness=bbox_thick)
                cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)
                cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=fontScale, color=label_color, thickness=bbox_thick, lineType=cv2.LINE_AA)
    return image


'''##################################
input: (10) YOLO model, image path, input size, class file path
output: image with predicted bboxes
obj:    detect objects in one image using YOLOv3 model
##################################'''
def detect_image(Yolo, image_path, output_path='', input_size=YOLO_INPUT_SIZE, show=False, save=False, CLASSES_PATH=YOLO_COCO_CLASS_PATH,
                 score_threshold=VALIDATE_SCORE_THRESHOLD, iou_threshold=VALIDATE_IOU_THRESHOLD, rectangle_colors='', show_label=True):
    original_image      = cv2.imread(image_path)
    original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image_data = image_preprocess(np.copy(original_image), input_size)                  #scale to size 416
    image_data = image_data[np.newaxis, ...].astype(np.float32)                         #reshape [1, 416, 416, 3]
    if np.isnan(image_data).any() or not (np.isfinite(image_data).all()):
        print("\n there is Nan number in image \n")
    pred_bbox = Yolo(image_data, training=False)                                                 #shape [3, batch_size, output_size, output_size, 3, 85]
    
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]               #reshape to [3, bbox_num, 85]
    pred_bbox = tf.concat(pred_bbox, axis=0)                                            #concatenate to [bbox_num, 85]
    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)      #scale to origional and select valid bboxes
    bboxes = nms(bboxes, iou_threshold, method='nms')                                       #Non-maximum suppression

    original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image = draw_bbox(original_image, bboxes, CLASSES_PATH=CLASSES_PATH, rectangle_colors=rectangle_colors, show_label=show_label) #draw bboxes
    
    if save:
        if output_path != '': cv2.imwrite(output_path, image)
    if show:
        # Show the image
        cv2.imshow("predicted image", image)
        # Load and hold the image
        if cv2.waitKey() == 'q':
            pass
        # To close the window after the required kill value was provided
        cv2.destroyAllWindows()
    return image

