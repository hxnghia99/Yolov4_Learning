#===============================================================#
#                                                               #
#   File name   : YOLOv3_loss.py                                #
#   Author      : hxnghia99                                     #
#   Created date: May 18th, 2022                                #
#   GitHub      : https://github.com/hxnghia99/Yolov4_Learning  #
#   Description : YOLOv3 loss computation                       #
#                                                               #
#===============================================================#

from YOLOv3_config import *
from YOLOv3_utils import *


#logits = inverse of sigmoid
logits = lambda x: tf.math.log(x/(1.0 - x) + 1e-7)

#Compute loss in YOLOv3 at each scale
#conv shape [batch, output size, output size, 255]
#pred shape [batch, output size, output size, 3, 85]
#label shape [batch, output size, output size, 3, 85]
def compute_loss_original(pred, conv, label, gt_bboxes, i=0, CLASSES_PATH=YOLO_COCO_CLASS_DIR):
    NUM_CLASSES = len(read_class_names(CLASSES_PATH))
    pred_shape  = tf.shape(pred)
    batch_size  = pred_shape[0]
    output_size = pred_shape[1]
    input_size  = output_size * YOLO_SCALE_OFFSET[i]
    anchors     = np.array(YOLO_ANCHORS[i])                       #shape [3,2]
    #change shape of raw convolutional output
    conv = tf.reshape(conv, (batch_size, output_size, output_size, ANCHORS_PER_GRID_CELL, 5 + NUM_CLASSES)) #shape [batch, output size, output size, 3, 85]
    #get individual data: xywh, confidence score, probability
    # 1) raw convolutional output
    conv_txtytwth       = conv[:, :, :, :, :4]
    conv_conf_raw       = conv[:, :, :, :, 4:5]
    conv_prob_raw       = conv[:, :, :, :, 5:]
    # 2) prediction
    pred_xywh           = pred[:, :, :, :, :4]
    pred_conf           = pred[:, :, :, :, 4:5]
    pred_prob           = pred[:, :, :, :, 5:]
    # 3) label
    label_xywh          = label[:, :, :, :, :4]
    label_respond       = label[:, :, :, :, 4:5]                  #shape [batch, output size, output size, 3]
    label_prob          = label[:, :, :, :, 5:]
    #Convert xywh to tx,ty,tw,th to calculate loss
    label_xy            = label_xywh[:, :, :, :, :2] / YOLO_SCALE_OFFSET[i]
    label_txty          = tf.map_fn(logits, tf.subtract(label_xy, tf.math.floor(label_xy)))              #logits(label_xy - round(label_xy))
    # print(label_txty)
    label_wh            = label_xywh[:, :, :, :, 2:]
    label_twth          = tf.cast(tf.math.log(label_wh / anchors[np.newaxis, np.newaxis, np.newaxis, :, :] + 1e-7), dtype=tf.float32)
    label_txtytwth      = tf.concat([label_txty, label_twth], axis=-1)
    #Part 1: coordinate loss
    # scale to get higher weights for small bboxes
    coor_loss_scale     = tf.cast(2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2), tf.float32)
    coor_loss           = label_respond * coor_loss_scale * tf.expand_dims(tf.reduce_sum(tf.square(label_txtytwth - conv_txtytwth), axis=-1), axis=-1)     
    #shape [batch, output size, output size, 3]
    #Part 2: confidence score loss
    # pred_xywh shape [batch, output, output, 3, 1, 4] and gt_bboxes shape [batch, output, output, 3, 100, 4]
    ious = bboxes_iou_from_xywh(pred_xywh[:, :, :, :, np.newaxis, :], gt_bboxes[:, np.newaxis,  np.newaxis, np.newaxis, :, :])      
    #shape [batch, output, output, 3, 100]
    max_ious = tf.expand_dims(tf.reduce_max(ious, axis=-1), axis=-1)                #shape [batch, output, output, 3, 1]
    # mask for background prediction (predictions with max_iou being less than threshold)
    bkgrd_respond = (1 - label_respond) * tf.cast(max_ious < YOLO_LOSS_IOU_THRESHOLD, dtype=tf.float32)
    # calculate confidence score loss
    conf_loss   =   label_respond * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_respond, logits=conv_conf_raw) \
                  + LAMBDA_NOOBJ * bkgrd_respond * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_respond, logits=conv_conf_raw)             
                  #shape [batch, output, output, 3, 1]
    #Part 3: class probability loss
    prob_loss = label_respond * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_prob_raw)                    
    #shape [batch, output, output, 3, num_class]
    #Summary of 3 types of loss
    coor_loss = LAMBDA_COORD * tf.reduce_mean(tf.reduce_sum(coor_loss, axis=[1,2,3]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))
    # print(coor_loss, '\n', conf_loss, '\n', prob_loss)

    return coor_loss, conf_loss, prob_loss


#Compute YOLOv3 loss for each scale using reference code
def compute_loss(pred, conv, label, gt_bboxes, i=0, CLASSES_PATH=YOLO_COCO_CLASS_DIR):
    NUM_CLASSES = len(read_class_names(CLASSES_PATH))
    pred_shape  = tf.shape(pred)
    batch_size  = pred_shape[0]
    output_size = pred_shape[1]
    input_size  = tf.cast(output_size * YOLO_SCALE_OFFSET[i], tf.float32)
    #change shape of raw convolutional output
    conv = tf.reshape(conv, (batch_size, output_size, output_size, ANCHORS_PER_GRID_CELL, 5 + NUM_CLASSES)) #shape [batch, output size, output size, 3, 85]
    #get individual data:
    # 1) raw convolutional output
    conv_conf_raw       = conv[:, :, :, :, 4:5]
    conv_prob_raw       = conv[:, :, :, :, 5:]
    # 2) prediction
    pred_xywh           = pred[:, :, :, :, :4]
    pred_conf           = pred[:, :, :, :, 4:5]
    # 3) label
    label_xywh          = label[:, :, :, :, :4]
    label_respond       = label[:, :, :, :, 4:5]                  #shape [batch, output size, output size, 3, 1]
    label_prob          = label[:, :, :, :, 5:]
    
    # *** Calculate giou loss from prediction and label ***
    giou = tf.expand_dims(bboxes_giou_from_xywh(pred_xywh, label_xywh), axis=-1)    #shape [batch, output size, output size, 3, 1]
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = label_respond * bbox_loss_scale * (1 - giou)
    
    # *** Calculate confidence score loss for grid cell containing objects and background ***
    ious = bboxes_iou_from_xywh(pred_xywh[:, :, :, :, np.newaxis, :], gt_bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])   #shape [batch, output, output, 3, 100]
    #              shape [batch, output size, output size, 3, 1, 4]  shape [batch, 1, 1, 1, 100, 4]
    max_iou = tf.expand_dims(tf.reduce_max(ious, axis=-1), axis=-1)    #shape [batch, output, output, 3, 1]
    bkgrd_respond = (1 - label_respond) * tf.cast(max_iou < YOLO_LOSS_IOU_THRESHOLD, tf.float32)
    conf_focal = tf.pow(label_respond - pred_conf, 2) 
    conf_loss = conf_focal * (
                    label_respond * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_respond, logits=conv_conf_raw)
                    +
                    bkgrd_respond * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_respond, logits=conv_conf_raw)
    )

    # *** Calculate class probability loss ***
    prob_loss = label_respond * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_prob_raw)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    return giou_loss, conf_loss, prob_loss


def compute_loss(pred, conv, label, bboxes, i=0, CLASSES_PATH=YOLO_COCO_CLASS_DIR):
    NUM_CLASS = len(read_class_names(CLASSES_PATH))
    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = YOLO_SCALE_OFFSET[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh     = pred[:, :, :, :, 0:4]
    pred_conf     = pred[:, :, :, :, 4:5]

    label_xywh    = label[:, :, :, :, 0:4]
    respond_bbox  = label[:, :, :, :, 4:5]
    label_prob    = label[:, :, :, :, 5:]

    giou = tf.expand_dims(bboxes_giou_from_xywh(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    # Find the value of IoU with the real box The largest prediction box
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    

    # If the largest iou is less than the threshold, it is considered that the prediction box contains no objects, then the background box
    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < YOLO_LOSS_IOU_THRESHOLD, tf.float32 )

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    # Calculate the loss of confidence
    # we hope that if the grid contains objects, then the network output prediction box has a confidence of 1 and 0 when there is no object.
    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    return giou_loss, conf_loss, prob_loss