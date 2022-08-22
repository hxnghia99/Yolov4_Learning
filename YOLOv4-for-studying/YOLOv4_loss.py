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
def compute_loss(pred, conv, label, gt_bboxes, i=0, CLASSES_PATH=YOLO_COCO_CLASS_PATH, fmap_student=None, fmap_teacher=None):
    label       = tf.convert_to_tensor(label)
    # gt_bboxes   = tf.convert_to_tensor(gt_bboxes)
    NUM_CLASSES = len(read_class_names(CLASSES_PATH))
    pred_shape  = tf.shape(pred)
    batch_size  = pred_shape[0]
    output_size_h = pred_shape[1]
    output_size_w = pred_shape[2]
    input_size_h = tf.Variable(TRAIN_INPUT_SIZE[1], dtype=tf.float32)
    input_size_w = tf.Variable(TRAIN_INPUT_SIZE[0], dtype=tf.float32)
    
    
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
    
    if not USE_CIOU_LOSS:
        # *** Calculate giou loss from prediction and label ***
        giou = tf.expand_dims(bboxes_giou_from_xywh(pred_xywh, label_xywh), axis=-1)    #shape [batch, output size, output size, 3, 1]
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size_h * input_size_w)
        giou_loss = label_respond * bbox_loss_scale * (1 - giou)  
    else:
        # # *** Calculate ciou loss from prediction and label ***
        # ciou = tf.expand_dims(bboxes_ciou_from_xywh(pred_xywh, label_xywh), axis=-1)    #shape [batch, output size, output size, 3, 1]
        # bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size_h * input_size_w)
        # giou_loss = label_respond * bbox_loss_scale * (1 - ciou)
        pass
    
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


    #if use featuremap teacher to teach feature map student
    if fmap_teacher != None:
        #global loss
        # gb_loss = tf.norm(fmap_teacher - fmap_student, ord=2, axis=-1)
        gb_loss = tf.math.abs(fmap_teacher - fmap_student)
        gb_loss = tf.reduce_mean(tf.reduce_sum(gb_loss, axis=[1,2]))
        #positive object loss
        flag_pos_obj = np.zeros(fmap_teacher.shape)
        num_channels = fmap_teacher.shape[-1]
        for k in range(batch_size):
            for j in range(YOLO_MAX_BBOX_PER_SCALE):
                if np.multiply.reduce(gt_bboxes[k,j][2:4]) != 0:        #gt_bboxes: xywh
                    gt_bbox = np.concatenate([gt_bboxes[k,j][:2]-gt_bboxes[k,j][2:4]*0.5, gt_bboxes[k,j][:2]+gt_bboxes[k,j][2:4]*0.5], axis=-1).astype(np.int32)
                    xmin, ymin, xmax, ymax = np.array(gt_bbox / YOLO_SCALE_OFFSET[i]).astype(np.int32)
                    temp = np.ones([ymax-ymin, xmax-xmin, num_channels])
                    flag_pos_obj[k][ymin:ymax, xmin:xmax, :] = temp
        flag_pos_obj = np.array(flag_pos_obj, dtype=np.bool)
        pos_obj_loss = (fmap_teacher - fmap_student) * tf.cast(flag_pos_obj, tf.float32)
        # pos_obj_loss = (fmap_teacher - fmap_student)[flag_pos_obj]
        # pos_obj_loss = tf.reduce_sum(tf.norm(pos_obj_loss, ord=1, axis=-1))
        pos_obj_loss = tf.reduce_mean(tf.reduce_sum(tf.math.abs(pos_obj_loss), axis=[1, 2]))
        # pos_obj_loss = tf.divide(pos_obj_loss, tf.cast(batch_size, tf.float32))

    if fmap_teacher!=None:
        return giou_loss, conf_loss, prob_loss, gb_loss, pos_obj_loss
    else:
        return giou_loss, conf_loss, prob_loss

    