from os import O_NONBLOCK
from matplotlib.pyplot import box
import tensorflow.keras.backend as K


def xywh2minmax(xy, wh):
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2

    return xy_min, xy_max

def yolo_head(feats):
    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = K.shape(feats)[1:3]  # assuming channels last               #1x2 matrix [7,7]
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
     
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = K.tile(
        K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])       # size 1 x 7 x 7 x 1 x 2
    conv_index = K.cast(conv_index, K.dtype(feats))

    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))       # size 1 x 1 x 1 x 1 x 2
    print(feats[..., :2].shape)
    box_xy = (feats[..., :2] + conv_index) / conv_dims * 448        #  featsp[]: batch x 7 x 7 x B x 2
    box_wh = feats[..., 2:4] * 448                                  #width and height of boxes

    return box_xy, box_wh

def iou(pred_mins, pred_maxes, true_mins, true_maxes):
    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_wh = pred_maxes - pred_mins
    true_wh = true_maxes - true_mins
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    return iou_scores

def yolov1_loss(y_true, y_pred):                    #size: batch x 7 x 7 x 30
    label_class = y_true[..., :20]  # batch * 7 * 7 * 20
    label_box = y_true[..., 20:24]  # batch * 7 * 7 * 4
    response_mask = y_true[..., 24:25]  # batch * 7 * 7 * 1

    predict_class = y_pred[..., :20]  # batch * 7 * 7 * 20
    predict_conf_score = y_pred[..., 20:22]  # batch * 7 * 7 * 2
    predict_box = y_pred[..., 22:]  # batch * 7 * 7 * 8

    _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])         # batch x 7 x 7 x 1 x 4
    _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])     # batch x 7 x 7 x 2 x 4

    label_xy, label_wh = yolo_head(_label_box)  # batch * 7 * 7 * 1 * 2, batch * 7 * 7 * 1 * 2
    label_xy = K.expand_dims(label_xy, 3)  # batch * 7 * 7 * 1 * 1 * 2
    label_wh = K.expand_dims(label_wh, 3)  # batch * 7 * 7 * 1 * 1 * 2
    label_xy_min, label_xy_max = xywh2minmax(label_xy, label_wh)  # batch * 7 * 7 * 1 * 1 * 2, batch * 7 * 7 * 1 * 1 * 2

    predict_xy, predict_wh = yolo_head(_predict_box)  # batch * 7 * 7 * 2 * 2, batch * 7 * 7 * 2 * 2
    predict_xy = K.expand_dims(predict_xy, 4)  # batch * 7 * 7 * 2 * 1 * 2
    predict_wh = K.expand_dims(predict_wh, 4)  # batch * 7 * 7 * 2 * 1 * 2
    predict_xy_min, predict_xy_max = xywh2minmax(predict_xy, predict_wh)  # batch * 7 * 7 * 2 * 1 * 2, batch * 7 * 7 * 2 * 1 * 2

    iou_scores = iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # batch * 7 * 7 * 2 * 1
    best_ious = K.max(iou_scores, axis=4)  # batch * 7 * 7 * 2
    best_box = K.max(best_ious, axis=3, keepdims=True)  # batch * 7 * 7 * 1

    box_mask = K.cast(best_ious >= best_box, K.dtype(best_ious))  # batch * 7 * 7 * 2

    no_object_loss = 0.5 * (1 - box_mask * response_mask) * K.square(0 - predict_conf_score)
    object_loss = box_mask * response_mask * K.square(1 - predict_conf_score)
    confidence_loss = no_object_loss + object_loss
    confidence_loss = K.sum(confidence_loss)

    class_loss = response_mask * K.square(label_class - predict_class)
    class_loss = K.sum(class_loss)

    _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])
    _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

    label_xy, label_wh = yolo_head(_label_box)  # batch * 7 * 7 * 1 * 2, batch * 7 * 7 * 1 * 2
    predict_xy, predict_wh = yolo_head(_predict_box)  # batch * 7 * 7 * 2 * 2, batch * 7 * 7 * 2 * 2

    box_mask = K.expand_dims(box_mask)
    response_mask = K.expand_dims(response_mask)

    box_loss = 5 * box_mask * response_mask * K.square((label_xy - predict_xy) / 448)
    box_loss += 5 * box_mask * response_mask * K.square( (K.sqrt(label_wh) - K.sqrt(predict_wh)) / 448 )
    box_loss = K.sum(box_loss)

    loss = confidence_loss + class_loss + box_loss

    return loss



# #Define a function to calculate IOU 
# def iou(label_xymin, label_xymax, pred_xymin, pred_xymax):          # input size: batch x 7 x 7 x B x 1 x 2 (B = 1 for label, B = 2 for prediction)
#     #calculate intersection areas
#     intersect_xymin = K.maximum(pred_xymin, label_xymin)            # size : batch x 7 x 7 x 2 x 1 x 2
#     intersect_xymax = K.minimum(pred_xymax, label_xymax)            # size : batch x 7 x 7 x 2 x 1 x 2
#     intersect_wh = K.maximum(intersect_xymax - intersect_xymin, 0)  #calculate width and height of intersection, size: batch x 7 x 7 x 2 x 1 x 2
#     intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]   #size: batch x 7 x 7 x 2 x 1
#     #calculate label areas and prediction areas
#     label_wh = label_xymax - label_xymin                            #size : batch x 7 x 7 x 2 x 1 x 2
#     label_areas = label_wh[..., 0] * label_wh[..., 1]               #size : batch x 7 x 7 x 2 x 1
#     pred_wh = pred_xymax - pred_xymin                               #same size with label
#     pred_areas = pred_wh[...,0] * pred_wh[..., 1]
#     #calculate union areas
#     union_areas = label_areas + pred_areas - intersect_areas        #size: batch x 7 x 7 x 2 x 1
#     return intersect_areas/union_areas


# #calculate center and (width, height) of B boxes, input size: batch x 7 x 7 x B x 4
# def yolo_box_cal(box):
#     #Create the dynamic convolutional dimensions of grid cell
#     conv_dims = K.shape(box)[1:3]                       # 7 x 7
#     #Create the matrix of row index and column index in YOLO
#     conv_row_idx = K.arange(0, stop=conv_dims[0])                       #Create 1D tensor for row idx: [0,...,6]
#     conv_column_idx = K.arange(0, stop=conv_dims[1])                    #Create 1D tensor for column idx: [0,...,6]

#     conv_row_idx = K.tile(conv_row_idx, [conv_dims[0]])                   #tiling row_idx by 7 in same row: 1D tensor with 49 terms
#     conv_column_idx = K.flatten(                                        #flatten matrix
#                         K.transpose(                                    #tranpose matrix
#                             K.tile(                                     #tile column_idx by 7 in different row: 2D tensor sized 7x7
#                                 K.expand_dims(conv_column_idx,0), [conv_dims[1], 1])))  #expand dims of column_idx to (1,7)

#     conv_idx = K.transpose(K.stack([conv_row_idx, conv_column_idx]))    #stack row_idx and column_idx to 2 x 49, then transpose to 49 x 2
#     conv_idx = K.reshape(conv_idx, [1, conv_dims[0], conv_dims[1], 1, 2])       #change shape 49 x 2      to      1 x 7 x 7 x 1 x 2
#     conv_idx = K.cast(conv_idx, K.dtype(box))                           #casting to the same type of box
#     #Casting the dimensions of grid cell to the same as box        
#     conv_dims = K.cast(K.reshape(conv_dims, [1,1,1,1,2]), K.dtype(box))         #size: 1 x 1 x 1 x 1 x 2

#     #calculate the box coordinate with value from 0 to 1
#     box_xy = (box[...,:2] + conv_idx) / conv_dims       # The matrix with 1 at k_th dimension can replicate and become the number of k_th dimension of other matrix
#     box_wh = box[...,2:4]

#     return box_xy, box_wh           #size: batch x 7 x 7 x B x 2


# #Define the function to calculate yolov1 loss
# def yolov1_loss(label, pred):
#     #Extract data from label  :  batch x 7 x 7 x 25
#     label_class = label[..., :20]                    #size: batch x 7 x 7 x 20
#     label_box = label[..., 20:24]                    #size: batch x 7 x 7 x 4
#     label_conf_score = label[..., 24:25]             #size: batch x 7 x 7 x 1
#     #Extract data from prediction   :   batch x 7 x 7 x 30
#     pred_class = pred[...,:20]                      #size : batch x 7 x 7 x 20
#     pred_conf_score = pred[...,20:22]               #size : batch x 7 x 7 x 2
#     pred_box = pred[...,22:30]                      #size : batch x 7 x 7 x 8

#     #add dimension to box matrix
#     _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])     # -1 represent for unknown batch_size, size: batch x 7 x 7 x 1 x 4
#     _pred_box = K.reshape(pred_box, [-1, 7, 7, 2, 4])       # size: batch x 7 x 7 x 2 x 4

#     #calculate coordinates of label box
#     label_xy, label_wh = yolo_box_cal(_label_box)               #size: batch x 7 x 7 x 1 x 2, batch x 7 x 7 x 1 x 2
#     label_xy = K.expand_dims(label_xy, 4)                       #make sure that the final 2 values represent correctly later, size: batch x 7 x 7 x 1 x 1 x 2
#     label_wh = K.expand_dims(label_wh, 4)                       #size : batch x 7 x 7 x 1 x 1 x 2
#     label_xymin = label_xy - label_wh/2                         #size : batch x 7 x 7 x 1 x 1 x 2
#     label_xymax = label_xy + label_wh/2                         #size : batch x 7 x 7 x 1 x 1 x 2
    
#     #calculate coordinates of prediction boxes
#     pred_xy, pred_wh = yolo_box_cal(_pred_box)                  #size: batch x 7 x 7 x 2 x 2, batch x 7 x 7 x 2 x 2
#     pred_xy = K.expand_dims(pred_xy, 4)                         #expand dimension to store iou value later, size: batch x 7 x 7 x 2 x 1 x 2
#     pred_wh = K.expand_dims(pred_wh, 4)                         #size : batch x 7 x 7 x 2 x 1 x 2
#     pred_xymin = pred_xy - pred_wh/2                            #size : batch x 7 x 7 x 2 x 1 x 2
#     pred_xymax = pred_xy + pred_wh/2                            #size : batch x 7 x 7 x 2 x 1 x 2
    
#     #caluclate confidence scores and best box positions
#     conf_scores = iou(label_xymin, label_xymax, pred_xymin, pred_xymax)  #size: batch x 7 x 7 x 2 x 1
#     best_conf_scores = conf_scores[..., 0]                              #size : batch x 7 x 7 x 2
#     best_box = K.max(best_conf_scores, axis=3, keepdims=True)                #size: batch x 7 x 7 x 1
#     box_mask = K.cast(best_conf_scores >= best_box, K.dtype(conf_scores))    #box position mask in total matrix, size: batch x 7 x 7 x 2

#     no_object_loss = 0.5 * (1 - box_mask * label_conf_score) * K.square(0 - pred_conf_score)
#     object_loss = box_mask * label_conf_score * K.square(1 - pred_conf_score)
#     confidence_loss = no_object_loss + object_loss
#     confidence_loss = K.sum(confidence_loss)

#     class_loss = label_conf_score * K.square(label_class - pred_class)
#     class_loss = K.sum(class_loss)

#     label_xy, label_wh = yolo_box_cal(_label_box)       #size: batch x 7 x 7 x 1 x 2, batch x 7 x 7 x 1 x 2
#     pred_xy, pred_wh = yolo_box_cal(_pred_box)          #size: batch x 7 x 7 x 2 x 2, batch x 7 x 7 x 2 x 2
#     box_mask = K.expand_dims(box_mask)
#     label_conf_score = K.expand_dims(label_conf_score)
#     box_loss = 5 * box_mask * label_conf_score * K.square(label_xy - pred_xy)
#     box_loss += 5 * box_mask * label_conf_score * K.square(label_wh - pred_wh)
#     box_loss = K.sum(box_loss)

#     loss = object_loss + class_loss + box_loss
#     return loss


