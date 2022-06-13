#===============================================================#
#                                                               #
#   File name   : YOLOv4_loss.py                                #
#   Author      : hxnghia99                                     #
#   Created date: May 18th, 2022                                #
#   GitHub      : https://github.com/hxnghia99/Yolov4_Learning  #
#   Description : YOLOv4 loss computation                       #
#                                                               #
#===============================================================#



import tensorflow as tf
from YOLOv4_config import *
from YOLOv4_utils import *

#Compute YOLOv4 loss for each scale using reference code
def compute_loss(pred, conv, label, gt_bboxes, i=0, CLASSES_PATH=YOLO_COCO_CLASS_PATH):
    label       = tf.convert_to_tensor(label)
    gt_bboxes   = tf.convert_to_tensor(gt_bboxes)
    NUM_CLASSES = len(read_class_names(CLASSES_PATH))
    pred_shape  = tf.shape(pred)
    batch_size  = pred_shape[0]
    output_size_h = pred_shape[1]
    output_size_w = pred_shape[2]
    input_size_h  = tf.cast(output_size_h * YOLO_SCALE_OFFSET[i], tf.float32)
    input_size_w  = tf.cast(output_size_w * YOLO_SCALE_OFFSET[i], tf.float32)
    #change shape of raw convolutional output
    conv = tf.reshape(conv, (batch_size, output_size_h, output_size_w, ANCHORS_PER_GRID_CELL, 5 + NUM_CLASSES)) #shape [batch, output size, output size, 3, 85]
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
    
    # # *** Calculate giou loss from prediction and label ***
    # giou = tf.expand_dims(bboxes_giou_from_xywh(pred_xywh, label_xywh), axis=-1)    #shape [batch, output size, output size, 3, 1]
    # bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size_h * input_size_w)
    # giou_loss = label_respond * bbox_loss_scale * (1 - giou)

    # *** Calculate ciou loss from prediction and label ***
    ciou = tf.expand_dims(bboxes_ciou_from_xywh(pred_xywh, label_xywh), axis=-1)    #shape [batch, output size, output size, 3, 1]
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size_h * input_size_w)
    giou_loss = label_respond * bbox_loss_scale * (1 - ciou)
    
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

    