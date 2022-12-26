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
from YOLOv4_model import *
import math

# from numpy import hstack
# from numpy.random import normal
# from sklearn.mixture import GaussianMixture
# from matplotlib import pyplot


#Compute YOLOv4 loss for each scale using reference code
def compute_loss(pred, conv, label, gt_bboxes, i=0, CLASSES_PATH=YOLO_COCO_CLASS_PATH, fmap_student=None, fmap_teacher=None, fmap_student_mid=None, fmap_teacher_mid=None):
    NUM_CLASSES = len(read_class_names(CLASSES_PATH))
    label       = tf.convert_to_tensor(label)
    gt_bboxes   = tf.convert_to_tensor(gt_bboxes)
    pred_shape  = tf.shape(pred)
    batch_size  = pred_shape[0]
    output_size_h = pred_shape[1]
    output_size_w = pred_shape[2]
    input_size_h = tf.Variable(TRAIN_INPUT_SIZE[1], dtype=tf.float32)
    input_size_w = tf.Variable(TRAIN_INPUT_SIZE[0], dtype=tf.float32)
    
    #change shape of raw convolutional output
    conv = tf.reshape(conv, (batch_size, output_size_h, output_size_w, ANCHORS_PER_GRID_CELL_SMALL if (i==0 and USE_5_ANCHORS_SMALL_SCALE) else ANCHORS_PER_GRID_CELL, PRED_NUM_PARAMETERS + NUM_CLASSES)) #shape [batch, output size, output size, 3, 85]
    #get individual data:
    # 1) raw convolutional output
    conv_conf_raw       = conv[:, :, :, :, 4:5]
    conv_prob_raw       = conv[:, :, :, :, 5:(5+NUM_CLASSES)]
    if PRED_NUM_PARAMETERS == 6:
        conv_roi_raw        = conv[:, :, :, :, (5+NUM_CLASSES):(6+NUM_CLASSES)]
    
    # 2) prediction
    pred_xywh           = pred[:, :, :, :, :4]
    pred_conf           = pred[:, :, :, :, 4:5]
    if PRED_NUM_PARAMETERS == 6:
        pred_roi            = pred[:, :, :, :, (5+NUM_CLASSES):(6+NUM_CLASSES)]
    # 3) label
    label_xywh          = label[:, :, :, :, :4]
    label_respond       = label[:, :, :, :, 4:5]                  #shape [batch, output size, output size, 3, 1]
    label_prob          = label[:, :, :, :, 5:(5+NUM_CLASSES)]
    if PRED_NUM_PARAMETERS == 6:
        label_roi           = label[:, :, :, :, (5+NUM_CLASSES):(6+NUM_CLASSES)]
    
    # *** Calculate giou loss from prediction and label ***
    giou = tf.expand_dims(bboxes_giou_from_xywh(pred_xywh, label_xywh), axis=-1)    #shape [batch, output size, output size, 3, 1]
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size_h * input_size_w)
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

    if PRED_NUM_PARAMETERS == 6:
        # *** ROI loss
        roi_focal = tf.pow(label_roi - pred_roi, 2)
        roi_loss = roi_focal * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_roi, logits=conv_roi_raw)
        # roi_loss = tf.abs(label_roi - pred_roi)
        
        roi_loss  = tf.reduce_mean(tf.reduce_sum(roi_loss, axis=[1,2,3,4]))

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))



#**** TESTING ****
    # max_iou         = tf.reshape(max_iou[0], (output_size_h*output_size_w*3,1))
    # pred_prob       = tf.reshape(tf.reduce_max(pred[:, :, :, :, 5:][0], axis=-1),(output_size_h*output_size_w*3,1))
    # pred_conf       = tf.reshape(pred_conf[0], (output_size_h*output_size_w*3,1))
    # conf_score      = tf.math.multiply(pred_conf, pred_prob)
    # X = hstack((max_iou, conf_score))
    # pyplot.hist(X, bins=50, density=True)
    # model = GaussianMixture(n_components=2, init_params='random')
    # model.fit(X)
    # yhat = model.predict(X)
    # pyplot.show()
    # print(yhat[:100])


    #if use featuremap teacher to teach feature map student
    if fmap_teacher != None:
        # """ FEATURE MAP DIFFERENCE LOSS """
        # #global loss
        # gb_loss = tf.square(fmap_teacher - fmap_student)
        # gb_loss = tf.reduce_mean(tf.reduce_sum(gb_loss, axis=[1,2,3])) #/ tf.cast((tf.shape(gb_loss)[1]*tf.shape(gb_loss)[2]*tf.shape(gb_loss)[3]), tf.float32))  #each pixel in hxwxc
        # #positive object loss
        # flag_pos_obj = np.zeros(fmap_teacher.shape)
        # num_channels = fmap_teacher.shape[-1]
        # list_num_pos_pixel = []
        # num_fmap_w_pos_pixel = 0
        # for k in range(batch_size):         #each image
        #     num_pos_pixel = 0
        #     for j in range(YOLO_MAX_BBOX_PER_SCALE):    #each gt bbox
        #         if np.multiply.reduce(gt_bboxes[k,j][2:4]) != 0:        #gt_bboxes: xywh
        #             gt_bbox = np.concatenate([gt_bboxes[k,j][:2]-gt_bboxes[k,j][2:4]*0.5, gt_bboxes[k,j][:2]+gt_bboxes[k,j][2:4]*0.5], axis=-1).astype(np.int32)
        #             xmin, ymin, xmax, ymax = np.array(gt_bbox / YOLO_SCALE_OFFSET[i]).astype(np.int32)
        #             num_pos_pixel += (ymax-ymin)*(xmax-xmin)*num_channels
        #             temp = np.ones([ymax-ymin, xmax-xmin, num_channels])
        #             flag_pos_obj[k][ymin:ymax, xmin:xmax, :] = temp
        #     if num_pos_pixel==0:
        #         num_pos_pixel=1
        #     else:
        #         num_fmap_w_pos_pixel+=1  
        #     list_num_pos_pixel.append(num_pos_pixel)
        # if num_fmap_w_pos_pixel == 0:
        #     num_fmap_w_pos_pixel=1
        # num_fmap_w_pos_pixel = tf.cast(num_fmap_w_pos_pixel, tf.float32)
        # list_num_pos_pixel = tf.cast(np.array(list_num_pos_pixel), tf.float32)
        # flag_pos_obj = np.array(flag_pos_obj, dtype=np.bool)
        # pos_obj_loss = (fmap_teacher - fmap_student) * tf.cast(flag_pos_obj, tf.float32)
        # pos_obj_loss = tf.reduce_sum(tf.reduce_sum(tf.square(pos_obj_loss), axis=[1, 2, 3])) / num_fmap_w_pos_pixel #/ list_num_pos_pixel) / num_fmap_w_pos_pixel
        # # gb_loss = tf.Variable(0.0)
        # # pos_obj_loss = tf.Variable(0.0)

        
        
        # def normalize_fmap(fmap):
        #     fmin = tf.math.reduce_min(fmap)
        #     fmax = tf.math.reduce_max(fmap)
        #     return (fmap - fmin) / (fmax - fmin)

        # def gaussian_patch(size, sigma):
        #     gauss = tf.Variable([math.exp(-tf.cast(x-size//2, tf.float32)**2 / (2.0*sigma**2)) for x in range(size)])     #e**(-(x-m)**2/2*sigma**2)
        #     return gauss / tf.math.reduce_sum(gauss)
        # def create_window(size, in_channel=1, out_channel=1, sigma=1.5):
        #     _1d_window = tf.expand_dims(gaussian_patch(size, sigma=sigma), axis=-1)     #shape (11, 1)
        #     _2d_window = tf.expand_dims(tf.expand_dims(tf.matmul(_1d_window, tf.transpose(_1d_window)), axis=-1), axis=-1) #shape (11, 11, 1, 1)
        #     window = tf.broadcast_to(_2d_window, (size, size, in_channel, out_channel))
        #     return window
        
        # def ssim(input1, input2, val_range=255, size=11, window=None):
        #     try:
        #         batch_size, height, width, in_channels = tf.shape(input1)
        #     except:
        #         height, width, in_channels = tf.shape(input1)
            
        #     if window is None:
        #         real_size = min(size, height, width)
        #         window = create_window(real_size, out_channel=in_channels)      #shape (batch, size, size, channels)

        #     #Calculate luminance params
        #     mu1         = tf.nn.conv2d(input1, window, strides=1, padding="SAME")
        #     mu2         = tf.nn.conv2d(input2, window, strides=1, padding="SAME")
            
        #     mu1_sq      = mu1 * mu1        #for denominator
        #     mu2_sq      = mu2 * mu2
        #     mu12        = mu1 * mu2             

        #     #Caclulate contrast and structural components
        #     sigma1_sq   = tf.nn.conv2d((input1 - mu1_sq)*(input1 - mu1_sq), window, strides=1, padding="SAME")
        #     sigma2_sq   = tf.nn.conv2d((input2 - mu2_sq)*(input2 - mu2_sq), window, strides=1, padding="SAME")
        #     sigma1      = tf.sqrt(sigma1_sq)
        #     sigma2      = tf.sqrt(sigma2_sq)
        #     sigma12     = tf.nn.conv2d(input1 * input2, window, strides=1, padding="SAME") - mu12

        #     #Some constants for stability
        #     C1 = (0.01 * val_range) ** 2    #may remove L later
        #     C2 = (0.03 * val_range) ** 2
        #     C3 = C2 / 2
            
        #     numerator1      = 2*mu1*mu2 + C1
        #     numerator2      = 2*sigma1*sigma2 + C2
        #     numerator3      = sigma12 + C3
        #     denominator1    = mu1_sq + mu2_sq + C1
        #     denominator2    = sigma1_sq + sigma2_sq + C2
        #     denominator3    = sigma1 + sigma2 + C3

        #     ssim_score = (numerator1 * numerator2 * numerator3)/(denominator1 * denominator2 * denominator3)

        #     ret = tf.math.reduce_mean(ssim_score)
        #     return ret


        # """ Distillation-loss-1: Detection loss with teacher prediction as soft label """
        # conv_shape      = tf.shape(fmap_teacher)  
        # batch_size      = conv_shape[0]
        # output_size_h   = conv_shape[1]
        # output_size_w   = conv_shape[2]
        # fmap_size_c     = tf.shape(fmap_student_mid)[3]
        # input_size_h    = tf.Variable(TRAIN_INPUT_SIZE[1], dtype=tf.float32)
        # input_size_w    = tf.Variable(TRAIN_INPUT_SIZE[0], dtype=tf.float32)
        # yolo_scale_offset       = [4, 8, 16]
        # yolo_anchors            = YOLO_ANCHORS
        # decode_fmap_teacher     = decode(fmap_teacher, NUM_CLASSES, i, np.array(yolo_scale_offset)*2, yolo_anchors)      #shape [batch, height, width, 3, 8]  --> prediction in 448x256
        # decode_fmap_student     = decode(fmap_student, NUM_CLASSES, i, yolo_scale_offset, yolo_anchors)      #understand student output as prediction in x2 image
        


        # scores                  = decode_fmap_teacher[:,:,:,:,4:5] * tf.math.reduce_max(decode_fmap_teacher[:,:,:,:,5:], axis=-1, keepdims=True)
        # frgrd_respond_1         = tf.cast(scores >= 0.5, tf.float32)
        # teacher_xywh            = decode_fmap_teacher[:, :, :, :, :4]
        # ious                    = bboxes_iou_from_xywh(teacher_xywh[:, :, :, :, np.newaxis, :], gt_bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])   #shape [batch, output, output, 3, 100]
        # #                           shape [batch, output size, output size, 3, 1, 4]  shape [batch, 1, 1, 1, 100, 4]
        # max_iou                 = tf.expand_dims(tf.reduce_max(ious, axis=-1), axis=-1)    #shape [batch, output, output, 3, 1]
        # frgrd_respond_2         = tf.cast(max_iou >= 0.5, tf.float32)
        # frgrd_respond           = tf.cast(tf.math.logical_and(tf.cast(frgrd_respond_1, tf.bool), tf.cast(frgrd_respond_2, tf.bool)), tf.float32)



        # """ Combine teacher prediction and hard label """
        # label                   = np.array(label, np.float32)
        # decode_fmap_teacher = np.array(decode_fmap_teacher, np.float32)
        # decode_fmap_teacher[:,:,:,:,:4] = decode_fmap_teacher[:,:,:,:,:4] / 2
        # flag = np.array(frgrd_respond, np.float32)
        # for batch in range(batch_size):
        #     for h in range(output_size_h):
        #         for w in range(output_size_w):
        #             for anchor_id in range(3):
        #                 if flag[batch][h][w][anchor_id][0]==1 and label[batch][h][w][anchor_id][4] == 0 and np.argmax(decode_fmap_teacher[batch, h, w, anchor_id,5:])==np.argmax(label[batch, h, w, anchor_id,5:]):
        #                     label[batch][h][w][anchor_id][:] = decode_fmap_teacher[batch][h][w][anchor_id][:]
        # decode_fmap_teacher     = tf.cast(label, tf.float32)
        
        
        # # scores                  = decode_fmap_teacher[:,:,:,:,4:5] * tf.math.reduce_max(decode_fmap_teacher[:,:,:,:,5:], axis=-1, keepdims=True)
        # # frgrd_respond_1         = tf.cast(scores >= 0.5, tf.float32)
        # # teacher_xywh            = decode_fmap_teacher[:, :, :, :, :4]
        # # ious                    = bboxes_iou_from_xywh(teacher_xywh[:, :, :, :, np.newaxis, :], gt_bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])   #shape [batch, output, output, 3, 100]
        # # #                           shape [batch, output size, output size, 3, 1, 4]  shape [batch, 1, 1, 1, 100, 4]
        # # max_iou                 = tf.expand_dims(tf.reduce_max(ious, axis=-1), axis=-1)    #shape [batch, output, output, 3, 1]
        # # frgrd_respond_2         = tf.cast(max_iou >= 0.5, tf.float32)
        # # frgrd_respond           = tf.cast(tf.math.logical_and(tf.cast(frgrd_respond_1, tf.bool), tf.cast(frgrd_respond_2, tf.bool)), tf.float32)
        # # bkgrd_respond           = (1 - frgrd_respond) * tf.cast(max_iou < 0.5, tf.float32)
        # # conf_flag               = tf.cast(tf.math.logical_or(tf.cast(frgrd_respond,tf.bool), tf.cast(bkgrd_respond,tf.bool)),tf.float32)

        # frgrd_respond           = tf.cast(decode_fmap_teacher[:,:,:,:,4:5] > 0.1, tf.float32)
        # ious                    = bboxes_iou_from_xywh(teacher_xywh[:, :, :, :, np.newaxis, :], gt_bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])   #shape [batch, output, output, 3, 100]
        # max_iou                 = tf.expand_dims(tf.reduce_max(ious, axis=-1), axis=-1)    #shape [batch, output, output, 3, 1]
        # bkgrd_respond           = (1 - frgrd_respond) * tf.cast(max_iou < 0.5, tf.float32)
        # conf_flag               = tf.cast(tf.math.logical_or(tf.cast(frgrd_respond,tf.bool), tf.cast(bkgrd_respond,tf.bool)),tf.float32)


        # fmap_teacher = tf.reshape(fmap_teacher, (batch_size, output_size_h, output_size_w, 3, 5 + NUM_CLASSES))
        # fmap_student = tf.reshape(fmap_student, (batch_size, output_size_h, output_size_w, 3, 5 + NUM_CLASSES))
        # #Having 4 fmaps: fmap_student + decode_fmap_student, fmap_teacher + decode_fmap_teacher
        
        # #GIoU loss
        # student_pred_xywh = decode_fmap_student[:, :, :, :, :4]
        # teacher_pred_xywh = decode_fmap_teacher[:, :, :, :, :4]
        # giou = tf.expand_dims(bboxes_giou_from_xywh(student_pred_xywh, teacher_pred_xywh), axis=-1)    #shape [batch, output size, output size, 3, 1]
        # bbox_loss_scale = 2.0 - 1.0 * teacher_pred_xywh[:, :, :, :, 2:3] * teacher_pred_xywh[:, :, :, :, 3:4] / (input_size_h * input_size_w * 4)
        # giou_loss = frgrd_respond * bbox_loss_scale * (1 - giou)
        
        # #Confidence score loss
        # student_raw_conf    = fmap_student[:,:,:,:,4:5]
        # student_pred_conf   = decode_fmap_student[:,:,:,:,4:5]
        # teacher_pred_conf   = decode_fmap_teacher[:,:,:,:,4:5]
        # conf_focal = tf.pow(teacher_pred_conf - student_pred_conf, 2) 
        # conf_loss = conf_flag * conf_focal * tf.nn.sigmoid_cross_entropy_with_logits(labels=teacher_pred_conf, logits=student_raw_conf)

        # #Class probability loss
        # student_raw_prob    = fmap_student[:,:,:,:,5:]
        # teacher_pred_prob   = decode_fmap_teacher[:,:,:,:,5:]
        # prob_loss           = frgrd_respond * tf.nn.sigmoid_cross_entropy_with_logits(labels=teacher_pred_prob, logits=student_raw_prob)

        # giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
        # conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
        # prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))
        
        # if fmap_teacher_mid == None:
        #     gb_loss = tf.Variable(0.0)
        # # elif fmap_teacher_mid != None and i==0:
        # else:
        #     # fmap_student_mid = tf.math.top_k(fmap_student_mid, k=fmap_size_c)[0]
        #     # fmap_teacher_mid = tf.math.top_k(fmap_teacher_mid, k=fmap_size_c)[0]
        #     fmap_student_mid = normalize_fmap(fmap_student_mid)
        #     fmap_teacher_mid = normalize_fmap(fmap_teacher_mid)
        #     # gb_loss = tf.math.reduce_mean(tf.abs(fmap_teacher_mid - fmap_student_mid))
        #     gb_loss = 1 - ssim(fmap_teacher_mid, fmap_student_mid)
        # # else:
        # #     gb_loss = tf.Variable(0.0)
        # pos_obj_loss = tf.Variable(0.0)



        # """ Distillation-loss-2: compare the pre-decoded output of teacher and student using sum-squared error """
        # conv_shape      = tf.shape(fmap_teacher)  
        # batch_size      = conv_shape[0]
        # output_size_h   = conv_shape[1]
        # output_size_w   = conv_shape[2]
        # #Change the output_shape of each scale into [batch_size, output_size_h, output_size_w, 3, 85]
        # fmap_teacher_2 = tf.reshape(fmap_teacher, (batch_size, output_size_h, output_size_w, 3, 5 + NUM_CLASSES))
        # #Split the final dimension into 4 information (offset xy, offset wh, confidence, class probabilities)
        # teacher_raw_dxdy, teacher_raw_dwdh, teacher_raw_conf, teacher_raw_prob = tf.split(fmap_teacher_2, (2, 2, 1, NUM_CLASSES), axis=-1)
        # #shape [batch_size, output_size_h, output_size_w, 3, ...]
        # teacher_raw_dxdy = tf.sigmoid(teacher_raw_dxdy)
        # teacher_raw_dwdh = tf.exp(teacher_raw_dwdh) / 1.0
        # teacher_raw_xywh = tf.concat([teacher_raw_dxdy, teacher_raw_dwdh], axis=-1)
        # #Predicted box confidence scores
        # teacher_raw_conf = tf.sigmoid(teacher_raw_conf)
        # #Predicted box class probabilities 
        # teacher_raw_prob = tf.sigmoid(teacher_raw_prob)
        # # fmap_teacher_2 = tf.concat([teacher_raw_dxdy, teacher_raw_dwdh, teacher_raw_conf, teacher_raw_prob], axis=-1)

        # fmap_student_2 = tf.reshape(fmap_student, (batch_size, output_size_h, output_size_w, 3, 5 + NUM_CLASSES))
        # #Split the final dimension into 4 information (offset xy, offset wh, confidence, class probabilities)
        # student_raw_dxdy, student_raw_dwdh, student_raw_conf, student_raw_prob = tf.split(fmap_student_2, (2, 2, 1, NUM_CLASSES), axis=-1)
        # #shape [batch_size, output_size_h, output_size_w, 3, ...]
        # student_raw_dxdy = tf.sigmoid(student_raw_dxdy)
        # student_raw_dwdh = tf.exp(student_raw_dwdh)
        # student_raw_xywh = tf.concat([student_raw_dxdy, student_raw_dwdh], axis=-1)
        # #Predicted box confidence scores
        # student_raw_conf = tf.sigmoid(student_raw_conf)
        # #Predicted box class probabilities 
        # student_raw_prob = tf.sigmoid(student_raw_prob)
        # # fmap_student_2 = tf.concat([student_raw_dxdy, student_raw_dwdh, student_raw_conf, student_raw_prob], axis=-1)

        # yolo_scale_offset       = [8, 16, 32]
        # yolo_anchors            = YOLO_ANCHORS * 1
        # decode_fmap_teacher     = decode(fmap_teacher, NUM_CLASSES, i, yolo_scale_offset, yolo_anchors)      #shape [batch, height, width, 3, 8]  --> prediction in 448x256
        # scores                  = decode_fmap_teacher[:,:,:,:,4:5] * tf.math.reduce_max(decode_fmap_teacher[:,:,:,:,5:], axis=-1, keepdims=True)
        # frgrd_respond_1         = tf.cast(scores >= 0.5, tf.float32)
        # teacher_xywh            = decode_fmap_teacher[:, :, :, :, :4] / 2.0
        # ious                    = bboxes_iou_from_xywh(teacher_xywh[:, :, :, :, np.newaxis, :], gt_bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])   #shape [batch, output, output, 3, 100]
        # #                           shape [batch, output size, output size, 3, 1, 4]  shape [batch, 1, 1, 1, 100, 4]
        # max_iou                 = tf.expand_dims(tf.reduce_max(ious, axis=-1), axis=-1)    #shape [batch, output, output, 3, 1]
        # frgrd_respond_2         = tf.cast(max_iou >= 0.5, tf.float32)
        # frgrd_respond           = tf.cast(tf.math.logical_and(tf.cast(frgrd_respond_1, tf.bool), tf.cast(frgrd_respond_2, tf.bool)), tf.float32)
        # bkgrd_respond = (1 - frgrd_respond) * tf.cast(max_iou < 0.5, tf.float32)
        # conf_flag = tf.cast(tf.math.logical_or(tf.cast(frgrd_respond,tf.bool), tf.cast(bkgrd_respond,tf.bool)),tf.float32)

        # conf_loss = tf.math.reduce_mean(tf.math.reduce_sum(conf_flag * tf.square(teacher_raw_conf - student_raw_conf), axis=[1,2,3,4]))
        # giou_loss = tf.math.reduce_mean(tf.math.reduce_sum(frgrd_respond * tf.square(teacher_raw_xywh - student_raw_xywh), axis=[1,2,3,4]))
        # prob_loss = tf.math.reduce_mean(tf.math.reduce_sum(frgrd_respond * tf.square(teacher_raw_prob - student_raw_prob), axis=[1,2,3,4]))

        gb_loss = tf.Variable(0.0)
        pos_obj_loss = tf.Variable(0.0)


    if fmap_teacher!=None:
        return giou_loss, conf_loss, prob_loss, LAMDA_FMAP_LOSS*gb_loss, LAMDA_FMAP_LOSS*pos_obj_loss
    else:
        if PRED_NUM_PARAMETERS==6:
            return giou_loss, conf_loss, prob_loss, roi_loss
        else:
            return giou_loss, conf_loss, prob_loss
    