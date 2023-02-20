# #===============================================================#
# #                                                               #
# #   File name   : Test.py                                       #
# #   Author      : hxnghia99                                     #
# #   Created date: May 24th, 2022                                #
# #   GitHub      : https://github.com/hxnghia99/Yolov4_Learning  #
# #   Description : YOLOv4 image prediction testing               #
# #                                                               #
# #===============================================================#



import os
import cv2
import numpy as np
from YOLOv4_config import *
from YOLOv4_utils import *
from YOLOv4_dataset import *
from YOLOv4_model import *


def convert_into_original_size(array_bboxes, original_size): #array: shape [w, h, 3, 8]
    input_size                  = YOLO_INPUT_SIZE
    org_image_h, org_image_w    = original_size
    if len(array_bboxes)!=0:
        pred_xywh = array_bboxes[:,0:4]
        # (x, y, w, h) --> (xmin, ymin, xmax, ymax) : size 416 x 416
        pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                    pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
        # prediction (xmin, ymin, xmax, ymax) -> prediction (xmin_org, ymin_org, xmax_org, ymax_org)
        resize_ratio = min(input_size[0] / org_image_w, input_size[1] / org_image_h)
        dw = (input_size[0] - resize_ratio * org_image_w) / 2                      #pixel position recalculation
        dh = (input_size[1] - resize_ratio * org_image_h) / 2
        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio     #(pixel_pos - dw)/resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio
        array_orig_bboxes = np.concatenate([pred_coor, array_bboxes[:,4:5]], axis=-1)
        return array_orig_bboxes
    else:
        return np.array([])

RELATIVE_PATH               = "E:/dataset/TOTAL/"
PREFIX_PATH                 = "C:/Users/Claw/VSCode_Gitrepo/Yolov4_Learning/YOLOv4-for-studying/dataset/LG_DATASET"

text_by_line = "E:/dataset/TOTAL/train\images\c1_2020-10-292020-10-29-12-06-57-000119.jpg 779,128,802,185,1 402,52,421,102,1 644,46,658,91,1 885,65,902,106,1 293,69,308,116,1 620,43,633,84,1 900,35,914,72,1 632,44,643,88,1 338,126,376,187,1 288,0,297,16,1 345,148,376,212,1"
text_by_line = "E:/dataset/TOTAL/train\images/frame_20210417_090930_00518_51.jpg 443,493,636,608,0 125,254,168,293,1 152,211,198,256,1 150,177,194,215,1 330,77,377,117,1 298,124,339,170,1 328,206,378,252,1 962,1007,1052,1056,2 1184,981,1274,1029,2 689,1042,773,1080,2 526,297,591,353,2"
text_by_line = "E:/dataset/TOTAL/train\images\c1_2020-10-292020-10-29-12-15-58-000137.jpg 1087,343,1143,438,1 421,260,461,354,1 640,149,672,216,1 1034,151,1062,202,1 505,88,522,144,1 749,103,767,159,1 385,105,406,156,1 579,97,595,146,1 411,98,433,154,1 471,84,494,139,1 286,75,307,121,1 408,29,425,64,1 579,48,589,85,1 425,93,444,149,1 619,37,633,77,1 907,27,916,59,1 259,77,278,120,1 681,89,703,145,1 666,103,683,160,1 400,0,414,23,1 710,100,723,156,1 352,0,365,15,1 599,1,607,30,1 513,2,522,27,1 276,65,292,112,1 405,32,422,65,1 714,107,732,160,1 713,114,726,174,1 724,121,738,177,1 734,135,749,185,1 747,0,757,27,1 276,2,286,31,1 284,0,291,21,1 291,0,300,28,1 316,0,327,22,1 331,0,338,17,1 259,63,271,106,1"
text_by_line = "E:/dataset/TOTAL/test\images\cyclist_223_171_53_30_11002_2020-11-062020-11-06-12-11-26-000382.jpg 543,212,662,263,0 810,382,905,500,0 546,179,622,222,0 91,452,136,536,1 245,255,265,309,1 1140,62,1151,116,1 94,321,121,380,1 949,118,962,144,1 295,272,312,327,1 385,219,401,268,1 311,260,331,306,1 283,275,299,329,1 396,260,418,311,1 450,204,463,244,1 1119,21,1126,46,1 892,114,901,138,1 527,170,538,206,1 7,314,29,368,1 1126,21,1133,46,1 962,101,971,128,1 403,223,419,267,1 418,226,434,271,1 1106,23,1112,48,1 1111,23,1117,48,1 952,100,962,126,1 280,251,296,277,1 293,252,309,277,1 415,209,428,244,1 425,210,440,254,1 459,206,472,246,1 479,181,492,217,1 492,181,502,220,1 511,169,524,208,1 671,394,703,483,2"
# text_by_line = "E:/dataset/TOTAL/test\images\cyclist_223_171_39_182_11000_2020-11-092020-11-09-12-00-06-000359.jpg 724,108,747,159,1 758,121,775,170,1 629,112,663,166,1 485,116,507,179,1 885,111,902,154,1 672,59,684,105,1 683,63,696,109,1 698,62,718,114,1 291,136,326,197,1 1102,117,1118,164,1 1122,131,1140,180,1 291,122,309,180,1 765,112,777,162,1 431,366,483,499,2"
# text_by_line = "E:/dataset/TOTAL/test\images/frame_20210424_095010_00681_51.jpg 1242,221,1415,335,0 28,509,146,661,0 1505,129,1735,258,0 760,425,951,550,0 753,314,948,432,0 1064,339,1305,486,0 1688,218,1920,356,0 1888,90,1919,180,0 277,183,317,231,1 319,90,358,133,1 338,276,408,325,2"
# text_by_line = "E:/dataset/TOTAL/test\images/frame_20210501_100506_00371.jpg 155,694,293,781,1"
# text_by_line = "E:/dataset/TOTAL/train\images/frame_20210425_120031_01203.jpg 1607,220,1737,330,1 1565,253,1705,361,1 1369,386,1544,482,1 359,304,630,723,1 490,829,634,1047,2 37,502,187,643,2"



test_path = "YOLOv4-for-studying/dataset/LG_DATASET/test_1.txt"
# test_path = "YOLOv4-for-studying/dataset/Visdrone_DATASET/test_1.txt"

test_1_image = Dataset("test", TEST_LABEL_GT_PATH=test_path)
original_image, image_data, bboxes, label, resized_label = test_1_image.test_label_gt()
original_h, original_w, _ = original_image.shape

imagex2 = image_preprocess(np.copy(original_image), (448, 256))
imagex2_data = imagex2[np.newaxis,...].astype(np.float32)


# test1, test2 = image_preprocess(np.copy(original_image), (224,128), np.copy(bboxes))
# test3, test4 = image_preprocess(np.copy(original_image), (448,256), np.copy(bboxes))


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
# bboxes = np.array(bboxes)
# original_image = cv2.imread(image_path)
# image_data = image_preprocess(np.copy(original_image), np.array(size))

image_truth             = draw_bbox(np.copy(original_image), bboxes, CLASSES_PATH=YOLO_CLASS_PATH, show_label=False, show_grids=False)
image_label_sbboxes     = draw_bbox(np.copy(original_image), label[0], CLASSES_PATH=YOLO_CLASS_PATH, show_label=False, show_grids=True)
image_gt_sbboxes        = draw_bbox(np.copy(original_image), label[3], CLASSES_PATH=YOLO_CLASS_PATH, show_label=False, show_grids=True)


size = YOLO_INPUT_SIZE
compare_fmap_w_teacher = False
compare_pred_w_teacher = False
show_only_pred_student = True
student_weight = "YOLOv4-for-studying/checkpoints/lg_dataset_transfer_224x128/epoch-43_valid-loss-11.40/yolov4_lg_transfer"
# teacher_weight = "YOLOv4-for-studying/checkpoints/lg_dataset_transfer_448x256/epoch-36_valid-loss-8.48_origin-dilate-bb/yolov4_lg_transfer"
# teacher_weight = "YOLOv4-for-studying/checkpoints/Num-105_lg_dataset_transfer_224x128/epoch-45_valid-loss-10.95_medium-anchor/yolov4_lg_transfer"
#Create YOLO model
yolo = YOLOv4_Model(CLASSES_PATH=YOLO_CLASS_PATH, training=False)
# yolo = create_YOLOv4_backbone(CLASSES_PATH=YOLO_CLASS_PATH)
# yolo.load_weights(student_weight)
yolo.load_weights(student_weight)
pred_bboxes = yolo(image_data, training=False)


# #Create YOLO model: TEACHER
# yolo = YOLOv4_Model(CLASSES_PATH=YOLO_CLASS_PATH, training=False)
# yolo.load_weights(teacher_weight)
# pred_bboxes = yolo(image_data, training=False)


if compare_fmap_w_teacher:
    teacher = create_YOLOv4_backbone(CLASSES_PATH=YOLO_CLASS_PATH)
    teacher.load_weights(teacher_weight)
    _, _, _, teacher_out_sbboxes, teacher_out_mbboxes, teacher_out_lbboxes = teacher(imagex2_data, training=False)
    student_pred_small      = pred_bboxes[0]

    # scores                  = student_pred_small[:,:,:,:,4:5] * tf.math.reduce_max(student_pred_small[:,:,:,:,5:], axis=-1, keepdims=True)
    # frgrd_respond_1         = tf.cast(scores >= 0.5, tf.float32)
    # teacher_xywh            = student_pred_small[:, :, :, :, :4]
    # ious                    = bboxes_iou_from_xywh(teacher_xywh[0, :, :, :, np.newaxis, :], np.array(resized_label[3], dtype=np.float32)[np.newaxis, np.newaxis, np.newaxis, :, :])   #shape [batch, output, output, 3, 100]
    # max_iou                 = tf.expand_dims(tf.reduce_max(ious, axis=-1), axis=-1)    #shape [batch, output, output, 3, 1]
    # frgrd_respond_2         = tf.cast(max_iou >= 0.5, tf.float32)
    # student_frgrd_respond   = tf.squeeze(tf.cast(tf.math.logical_and(tf.cast(frgrd_respond_1, tf.bool), tf.cast(frgrd_respond_2, tf.bool)), tf.float32))
    
    
    yolo_scale_offset       = np.array([4, 8, 16])
    yolo_anchors            = np.array(YOLO_ANCHORS)
    teacher_pred_sbboxes    = decode(teacher_out_sbboxes, NUM_CLASS=3 , i=0, YOLO_SCALE_OFFSET=yolo_scale_offset, YOLO_ANCHORS=yolo_anchors)
    scores                  = teacher_pred_sbboxes[:,:,:,:,4:5] * tf.math.reduce_max(teacher_pred_sbboxes[:,:,:,:,5:], axis=-1, keepdims=True)
    frgrd_respond_1         = tf.cast(scores >= 0.5, tf.float32)
    teacher_xywh            = teacher_pred_sbboxes[:, :, :, :, :4]
    ious                    = bboxes_iou_from_xywh(teacher_xywh[0, :, :, :, np.newaxis, :], np.array(resized_label[3], dtype=np.float32)[np.newaxis, np.newaxis, np.newaxis, :, :])   #shape [batch, output, output, 3, 100]
    max_iou                 = tf.expand_dims(tf.reduce_max(ious, axis=-1), axis=-1)    #shape [batch, output, output, 3, 1]
    frgrd_respond_2         = tf.cast(max_iou >= 0.5, tf.float32)
    #***
    teacher_frgrd_respond   = tf.squeeze(tf.cast(tf.math.logical_and(tf.cast(frgrd_respond_1, tf.bool), tf.cast(frgrd_respond_2, tf.bool)), tf.float32))
    #***
    label_frgrd_respond     = tf.cast(resized_label[0][:,:,:,4], tf.float32)    #have overlaping anchors
    #***
    intersect_teacher_label_respond = tf.math.logical_and(tf.cast(teacher_frgrd_respond,tf.bool), tf.cast(label_frgrd_respond,tf.bool))
    #***
    teacher_except_intersect_respond = tf.math.logical_xor(tf.cast(teacher_frgrd_respond,tf.bool), intersect_teacher_label_respond)
    #***
    label_except_intersect_respond = tf.math.logical_xor(tf.cast(label_frgrd_respond,tf.bool), intersect_teacher_label_respond)
    

    teacher_pred_all        = np.array(teacher_pred_sbboxes[0][tf.cast(teacher_frgrd_respond, tf.bool)])
    teacher_pred_all        = np.array([np.concatenate([x[0:4],np.array([np.argmax(x[5:8])], dtype=np.float32)],axis=-1) for x in teacher_pred_all], np.float32)
    teacher_pred_all        = convert_into_original_size(teacher_pred_all, [original_h, original_w])
    image_test              = draw_bbox(np.copy(original_image), teacher_pred_all, CLASSES_PATH=YOLO_CLASS_PATH, show_label=False, show_grids=True)
    cv2.imshow('Teacher Pred ALL', cv2.resize(image_test,(960, 540)))

    teacher_pred_all        = np.array(teacher_pred_sbboxes[0][tf.cast(teacher_except_intersect_respond, tf.bool)])
    teacher_pred_all        = np.array([np.concatenate([x[0:4],np.array([np.argmax(x[5:8])], dtype=np.float32)],axis=-1) for x in teacher_pred_all], np.float32)
    teacher_pred_all        = convert_into_original_size(teacher_pred_all, [original_h, original_w])
    image_test              = draw_bbox(np.copy(original_image), teacher_pred_all, CLASSES_PATH=YOLO_CLASS_PATH, show_label=False, show_grids=True)
    cv2.imshow('Teacher Pred except intersection', cv2.resize(image_test,(960, 540)))
    
    teacher_pred_all        = np.array(teacher_pred_sbboxes[0][tf.cast(intersect_teacher_label_respond, tf.bool)])
    teacher_pred_all        = np.array([np.concatenate([x[0:4],np.array([np.argmax(x[5:8])], dtype=np.float32)],axis=-1) for x in teacher_pred_all], np.float32)
    teacher_pred_all        = convert_into_original_size(teacher_pred_all, [original_h, original_w])
    image_test              = draw_bbox(np.copy(original_image), teacher_pred_all, CLASSES_PATH=YOLO_CLASS_PATH, show_label=False, show_grids=True)
    cv2.imshow('Teacher Pred Intersection', cv2.resize(image_test,(960, 540)))

    teacher_pred_all        = np.array(resized_label[0][tf.cast(label_frgrd_respond, tf.bool)])
    teacher_pred_all        = np.array([np.concatenate([x[0:4],np.array([np.argmax(x[5:8])], dtype=np.float32)],axis=-1) for x in teacher_pred_all], np.float32)
    teacher_pred_all        = convert_into_original_size(teacher_pred_all, [original_h, original_w])
    image_test_2              = draw_bbox(np.copy(original_image), teacher_pred_all, CLASSES_PATH=YOLO_CLASS_PATH, show_label=False, show_grids=True)
    cv2.imshow('Label ALL', cv2.resize(image_test_2,(960, 540)))

    teacher_pred_all        = np.array(resized_label[0][tf.cast(label_except_intersect_respond, tf.bool)])
    teacher_pred_all        = np.array([np.concatenate([x[0:4],np.array([np.argmax(x[5:8])], dtype=np.float32)],axis=-1) for x in teacher_pred_all], np.float32)
    teacher_pred_all        = convert_into_original_size(teacher_pred_all, [original_h, original_w])
    image_test_2              = draw_bbox(np.copy(original_image), teacher_pred_all, CLASSES_PATH=YOLO_CLASS_PATH, show_label=False, show_grids=True)
    cv2.imshow('Label except intersection', cv2.resize(image_test_2,(960, 540)))

    teacher_pred_all        = np.array(resized_label[0][tf.cast(intersect_teacher_label_respond, tf.bool)])
    teacher_pred_all        = np.array([np.concatenate([x[0:4],np.array([np.argmax(x[5:8])], dtype=np.float32)],axis=-1) for x in teacher_pred_all], np.float32)
    teacher_pred_all        = convert_into_original_size(teacher_pred_all, [original_h, original_w])
    image_test_2              = draw_bbox(np.copy(original_image), teacher_pred_all, CLASSES_PATH=YOLO_CLASS_PATH, show_label=False, show_grids=True)
    cv2.imshow('Label Intersection', cv2.resize(image_test_2,(960, 540)))

    if cv2.waitKey() == 'q':
        pass

elif compare_pred_w_teacher:
    teacher = create_YOLOv4_backbone(CLASSES_PATH=YOLO_CLASS_PATH)
    teacher.load_weights(teacher_weight)
    _, _, _, teacher_out_sbboxes, teacher_out_mbboxes, teacher_out_lbboxes = teacher(imagex2_data, training=False)
    
    teacher_pred_bboxes = [decode(x, NUM_CLASS=3, i=i, YOLO_SCALE_OFFSET=np.array(YOLO_SCALE_OFFSET)*2, YOLO_ANCHORS=np.array(YOLO_ANCHORS)*2) for i,x in enumerate([teacher_out_sbboxes, teacher_out_mbboxes, teacher_out_lbboxes])]
    teacher_pred_bboxes = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in teacher_pred_bboxes]               #reshape to [3, bbox_num, 85]
    teacher_pred_bboxes = tf.concat(teacher_pred_bboxes, axis=0)                                            #concatenate to [bbox_num, 85]
    # pred_bboxes = pred_bboxes[0]
    teacher_pred_bboxes = postprocess_boxes(teacher_pred_bboxes, np.copy(original_image), [448, 256], 0.5)      #scale to origional and select valid bboxes
    teacher_pred_bboxes = nms(teacher_pred_bboxes, 0.5, method='nms')                                       #Non-maximum suppression
    teacher_image_pred = draw_bbox(np.copy(original_image), teacher_pred_bboxes, CLASSES_PATH=YOLO_CLASS_PATH, show_confidence=True, show_label=False, show_grids=False) #draw bboxes

    
    pred_bboxes = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bboxes]               #reshape to [3, bbox_num, 85]
    pred_bboxes = tf.concat(pred_bboxes, axis=0)                                            #concatenate to [bbox_num, 85]
    # pred_bboxes = pred_bboxes[0]
    pred_bboxes = postprocess_boxes(pred_bboxes, np.copy(original_image), size, 0.5)      #scale to origional and select valid bboxes
    pred_bboxes = nms(pred_bboxes, 0.5, method='nms')                                       #Non-maximum suppression
    


    #having 3 kinds of bboxes: bboxes, pred_bboxes, teacher_pred_bboxes
    error = np.zeros((4,), np.float32)
    
    for bbox in pred_bboxes:
        list_ious = bboxes_iou_from_minmax_np(np.array(bboxes, np.float32)[:,0:4], np.array(bbox[np.newaxis,0:4],np.float32))
        gt_bbox_idx = np.argmax(list_ious)
        error += bboxes[gt_bbox_idx][:4] - bbox[:4]
    error = error/len(pred_bboxes)

    pred_bboxes = np.array(pred_bboxes,np.float32)
    pred_bboxes[:,:4] = pred_bboxes[:,:4] + error[np.newaxis,:]

    image_pred = draw_bbox(np.copy(original_image), pred_bboxes, CLASSES_PATH=YOLO_CLASS_PATH, show_confidence=True, show_label=False, show_grids=False) #draw bboxes

    cv2.imshow('All GT image', cv2.resize(image_truth,(960, 540)))
    cv2.imshow('Student predicted image', cv2.resize(image_pred,(960, 540)))
    cv2.imshow('Teacher predicted image', cv2.resize(teacher_image_pred,(960, 540)))
    if cv2.waitKey() == 'q':
        pass



elif show_only_pred_student:
    # pred_bboxes = pred_bboxes[0::2]
    # pred_bboxes = [decode(x, NUM_CLASS=3, i=i, YOLO_SCALE_OFFSET=YOLO_SCALE_OFFSET, YOLO_ANCHORS=YOLO_ANCHORS*2) for i,x in enumerate(pred_bboxes)]

    pred_bboxes = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bboxes]               #reshape to [3, bbox_num, 85]
    pred_bboxes = tf.concat(pred_bboxes, axis=0)                                            #concatenate to [bbox_num, 85]
    # pred_bboxes = pred_bboxes[0]
    pred_bboxes = postprocess_boxes(pred_bboxes, np.copy(original_image), size, 0.35)      #scale to origional and select valid bboxes
    pred_bboxes = nms(pred_bboxes, 0.5, method='nms')                                       #Non-maximum suppression

    image_pred = draw_bbox(np.copy(original_image), pred_bboxes, CLASSES_PATH=YOLO_CLASS_PATH, show_confidence=True, show_label=False, show_grids=False) #draw bboxes

    # imaget1 = cv2.resize(imaget1, np.array(size)*2)
    # imaget2, bboxes2 = image_preprocess(np.copy(imaget), np.array(size)*2, np.copy(bboxes), sizex2_flag=True)

    # image = draw_bbox(np.copy(imaget), bboxes, YOLO_CLASS_PATH, show_label=False)
    cv2.imshow('All GT image', cv2.resize(image_truth,(960, 540)))
    cv2.imshow('small GT image', cv2.resize(image_gt_sbboxes,(960, 540)))
    cv2.imshow('small label image', cv2.resize(image_label_sbboxes,(960, 540)))
    cv2.imshow('Predicted image', cv2.resize(image_pred,(960, 540)))
    if cv2.waitKey() == 'q':
        pass





# image = imaget[np.newaxis, ...]

# # print(image.shape)
# a = np.ones(image.shape)
# image = np.concatenate([image, a, a, a], axis=-1)
# print(image.shape)

# test = tf.nn.depth_to_space(image, 2)
# print(test.shape)


# import tensorflow as tf

# input = tf.keras.layers.Input([None, None, 3])



# output1 = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=(2,2), padding='same')(input)
# model1 = tf.keras.Model(input, output1)

# output2 = tf.keras.layers.UpSampling2D()(input)
# model2 = tf.keras.Model(input, output2)

# image_o1 = model1(image)
# image_o2 = model2(image)

# image_o3 = tf.keras.layers.UpSampling2D()(image)



# print(image.shape)
# print(image_o1.shape)

# image_o1t = np.array(image_o1[0], np.uint8)



# print(image_o2.shape)
# print(image_o3.shape)

# print(np.array(image_o2[0, 0, 0:10, :]))
# print(np.array(image_o3[0, 0, 0:10, :]))

# image_o1t = np.array(image_o1[0]).astype(np.uint8)
# image_o2t = np.array(image_o2[0]).astype(np.uint8)

# # cv2.imshow('test0', cv2.resize(np.array(image[0]), (1280, 720)))
# cv2.imshow('test1', cv2.resize(image_o1t, (1280, 720)))
# cv2.imshow('test2', cv2.resize(image_o2t, (1280, 720)))
# # cv2.imshow('test3', cv2.resize(np.array(image_o3[0]), (1280, 720)))
# if cv2.waitKey() == 'q':
#     pass



# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# # Import TensorFlow
# import tensorflow as tf

# # Helper libraries
# import numpy as np
# print(tf.__version__)


# fashion_mnist = tf.keras.datasets.fashion_mnist

# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# # Add a dimension to the array -> new shape == (28, 28, 1)
# # This is done because the first layer in our model is a convolutional
# # layer and it requires a 4D input (batch_size, height, width, channels).
# # batch_size dimension will be added later on.
# train_images = train_images[..., None]
# test_images = test_images[..., None]

# # Scale the images to the [0, 1] range.
# train_images = train_images / np.float32(255)
# test_images = test_images / np.float32(255)



# strategy = tf.distribute.MirroredStrategy(devices=["GPU:0"])
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


# BUFFER_SIZE = len(train_images)

# BATCH_SIZE_PER_REPLICA = 64
# GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

# EPOCHS = 10


# train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) 
# test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE) 

# train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
# test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)


# def create_model():
#     model = tf.keras.Sequential([
#         tf.keras.layers.Conv2D(32, 3, activation='relu'),
#         tf.keras.layers.MaxPooling2D(),
#         tf.keras.layers.Conv2D(64, 3, activation='relu'),
#         tf.keras.layers.MaxPooling2D(),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dense(10)
#         ])

#     return model



# # Create a checkpoint directory to store the checkpoints.
# checkpoint_dir = './YOLOv4-for-studying/checkpoints/test'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


# with strategy.scope():
#   # Set reduction to `NONE` so you can do the reduction afterwards and divide by
#   # global batch size.
#     loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
#         from_logits=True,
#         reduction=tf.keras.losses.Reduction.NONE)
#     def compute_loss(labels, predictions):
#         per_example_loss = loss_object(labels, predictions)
#         return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)



# with strategy.scope():
#     test_loss = tf.keras.metrics.Mean(name='test_loss')

#     train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
#         name='train_accuracy')
#     test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
#         name='test_accuracy')



# # A model, an optimizer, and a checkpoint must be created under `strategy.scope`.
# with strategy.scope():
#     model = create_model()

#     optimizer = tf.keras.optimizers.Adam()

#     checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)


# def train_step(inputs):
#     images, labels = inputs

#     with tf.GradientTape() as tape:
#         predictions = model(images, training=True)
#         loss = compute_loss(labels, predictions)

#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#     train_accuracy.update_state(labels, predictions)
#     return loss 

# def test_step(inputs):
#     images, labels = inputs

#     predictions = model(images, training=False)
#     t_loss = loss_object(labels, predictions)

#     test_loss.update_state(t_loss)
#     test_accuracy.update_state(labels, predictions)


# # `run` replicates the provided computation and runs it
# # with the distributed input.
# @tf.function
# def distributed_train_step(dataset_inputs):
#     per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
#     return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
#                          axis=None)

# @tf.function
# def distributed_test_step(dataset_inputs):
#     return strategy.run(test_step, args=(dataset_inputs,))    

# for epoch in range(EPOCHS):
#     # TRAIN LOOP
#     total_loss = 0.0
#     num_batches = 0
#     for x in train_dist_dataset:
#         total_loss += distributed_train_step(x)
#         num_batches += 1
#     train_loss = total_loss / num_batches

#     # TEST LOOP
#     for x in test_dist_dataset:
#         distributed_test_step(x)

#     if epoch % 2 == 0:
#         checkpoint.save(checkpoint_prefix)

#     template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
#                 "Test Accuracy: {}")
#     print(template.format(epoch + 1, train_loss,
#                             train_accuracy.result() * 100, test_loss.result(),
#                             test_accuracy.result() * 100))

#     test_loss.reset_states()
#     train_accuracy.reset_states()
#     test_accuracy.reset_states()









# # ##############################################################################################
# # """
# # INFERENCE TIME
# # """
# # import os
# # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# # from YOLOv4_utils import *
# # from YOLOv4_config import *

# # # IMAGE_PATH = "./YOLOv4-for-studying/IMAGES/kite.jpg"
# # IMAGE_PATH = "./YOLOv4-for-studying/IMAGES/lg_street.jpg"
# # yolo = Load_YOLOv4_Model()
# # detect_image(yolo, IMAGE_PATH, show=True, save=False, CLASSES_PATH=YOLO_CLASS_PATH)

# # ##############################################################################################

# # print('a')




# # import os
# # from YOLOv4_config import *
# # from YOLOv4_utils import *
# # # text_by_line = "E:/dataset/TOTAL/test\images\cyclist_223_171_53_30_11000_2020-10-27-15-17-46-000000.jpg 12,120,69,181,0 80,7,128,27,0 57,5,87,25,0 1069,208,1124,274,2"
# # # text_by_line = "E:/dataset/TOTAL/train\images\c1_2020-10-292020-10-29-12-16-28-000138.jpg 762,114,782,176,1 785,120,804,180,1 676,77,692,120,1 651,64,663,109,1 663,71,677,113,1 327,31,342,67,1 364,119,382,183,1 618,76,639,124,1 320,191,344,264,1 411,0,421,23,1 611,74,631,122,1 359,0,366,17,1 282,4,292,32,1 628,58,642,101,1 343,0,353,19,1 208,598,260,647,1 234,298,279,391,1 367,115,395,168,1 369,0,379,16,1 268,6,277,33,1 256,17,269,48,1 904,24,920,47,1 920,26,928,48,1 407,0,415,22,1"
# # # text_by_line = "E:/dataset/TOTAL/train\images\\frame_20210422_072856_00116_51.jpg 1577,130,1822,263,0 184,0,260,58,0 38,0,118,77,0 381,159,421,201,1 423,152,463,198,1 111,977,176,1049,2"
# # # text_by_line = "./YOLOv4-for-studying/dataset/Visdrone_DATASET/VisDrone2019-DET-test-dev/images/0000006_01111_d_0000003.jpg 17,155,57,208,1 333,9,556,113,6 124,105,418,248,6 515,330,597,435,4 575,322,669,446,4 633,439,806,530,4 815,424,911,552,4 919,435,996,563,4 1008,436,1088,580,4 1089,443,1183,586,4 1106,292,1175,403,4 961,307,1083,408,4 990,151,1073,217,4 965,97,1087,154,4 181,652,409,758,4 1327,25,1350,54,10 1337,16,1349,48,2"
# # # text_by_line = "./YOLOv4-for-studying/dataset/Visdrone_DATASET/VisDrone2019-DET-val/images/0000335_00785_d_0000047.jpg 632,282,678,330,4 738,611,829,712,4 184,354,253,424,4 531,153,561,191,4 494,128,523,157,4 496,92,521,118,4 611,59,634,82,4 503,42,523,61,4 806,23,829,41,4 838,26,859,41,4 814,15,837,28,4 361,88,387,115,4 31,555,144,663,4 191,186,293,229,4 104,422,145,453,10 189,289,224,309,10 197,282,232,306,10 22,392,61,416,10 29,377,70,408,10 716,41,721,57,1 713,37,719,51,1 706,17,710,33,1 714,21,720,33,1 572,4,587,15,4 488,38,494,51,2 620,14,625,22,2 621,10,625,19,2 487,46,495,54,10 492,27,497,33,10 492,21,497,32,2 354,83,361,105,1 344,80,353,104,1 373,732,404,764,2 383,718,407,759,2"

# # # text_by_line = "./YOLOv4-for-studying/dataset/Visdrone_DATASET/VisDrone2019-DET-val/images/0000295_02400_d_0000033.jpg 598,601,645,657,4 660,545,704,595,4 595,512,637,557,4 517,603,568,657,4 676,624,724,686,4 738,589,790,650,4 805,568,854,621,4 750,507,795,550,4 683,495,727,535,4 668,414,738,447,4 578,405,646,435,4 459,379,521,411,4 426,372,460,398,4 395,350,420,372,4 776,423,845,460,4 763,364,803,385,4 643,322,665,343,4 10,302,49,318,4 168,311,212,327,4 98,387,167,410,4 0,473,79,509,4 5,456,96,489,4 48,449,121,479,4 55,439,134,466,4 87,430,156,453,4 93,420,172,446,4 105,414,172,433,4 93,358,141,381,4 0,325,88,364,9 72,339,117,360,4 422,356,446,380,4 538,393,601,415,4 633,402,690,429,4 865,443,932,472,4 925,405,1105,480,9 993,462,1068,498,4 1113,473,1204,512,4 1179,466,1263,502,4 1251,502,1348,541,4 1130,519,1228,563,4 1244,537,1357,590,4 824,631,973,765,9 754,291,791,304,4 730,285,764,301,4 477,330,499,350,4 448,318,469,335,4 543,315,560,332,4 511,314,530,330,4 487,299,503,312,4 459,299,476,314,4 438,288,454,302,4 632,305,650,321,4 611,285,628,299,4 664,297,681,310,4 607,253,620,263,4 588,251,599,259,4 638,265,652,275,4 540,219,548,227,4 604,235,616,243,4 592,227,603,236,4 545,306,562,321,4 545,299,560,311,4 540,286,556,305,4 515,306,532,321,4 514,296,530,311,4 516,291,532,304,4 514,280,527,294,4 515,273,527,286,4 488,290,505,303,4 490,282,504,296,4 489,273,504,289,4 494,266,506,280,4 494,259,506,271,4 464,289,478,301,4 467,283,480,295,4 469,274,484,286,4 473,269,484,280,4 474,263,487,276,4 477,259,486,270,4 476,251,489,264,4 479,243,489,255,4 480,236,489,249,4 478,219,493,240,9 458,233,473,256,9 461,218,476,239,9 442,279,458,292,4 446,274,459,286,4 452,268,464,280,4 423,310,442,326,4 415,319,433,344,5 453,256,466,274,5 583,222,595,228,5 586,217,595,224,4 576,214,584,222,4 666,289,682,300,4 661,280,674,294,5 819,514,834,545,1 869,536,881,572,1 861,524,871,553,1 991,533,1004,567,1 985,501,996,533,1 976,503,985,533,1 1077,591,1090,630,1 1098,567,1116,608,1 1130,577,1143,614,1 1154,573,1172,611,1 1256,581,1276,615,1 1328,584,1341,620,1 1224,608,1237,647,1 1187,627,1202,665,1 1022,494,1032,521,1 1011,481,1024,514,1 999,490,1010,517,1 976,469,983,496,1 792,472,804,495,1 824,474,836,495,1 832,488,845,511,1 863,462,874,486,1 941,508,957,535,1 918,523,933,548,1 912,503,924,531,1 901,527,908,557,1 897,519,906,553,1 885,529,895,566,1 1016,690,1031,734,1 1048,711,1068,754,1 1065,694,1089,736,1 1092,687,1111,731,1 1110,732,1128,765,1 1061,658,1078,695,1 1027,664,1048,702,1 993,656,1010,693,1 1021,625,1039,659,1 990,628,1006,663,1 979,607,996,637,1 987,591,1003,624,1 981,580,998,611,1 980,556,992,586,1 947,561,962,593,1 966,531,980,562,1 1051,549,1060,589,1 1037,541,1053,578,1 1041,526,1053,554,1 1099,537,1113,575,1 1108,555,1123,592,1 1111,549,1124,580,1 1116,540,1130,575,1 1126,553,1139,586,1 1137,550,1149,586,1 1159,541,1174,578,1 1178,537,1191,573,1 1191,547,1205,583,1 1208,543,1219,577,1 1224,513,1237,546,1 1309,645,1333,685,1 1202,702,1222,753,1 1309,732,1329,765,1 1153,624,1174,660,1 1177,599,1193,640,1 1192,608,1206,640,1 1207,606,1219,648,1 1200,624,1211,667,1 1121,527,1132,554,1 1121,518,1130,547,1 1112,518,1123,549,1 1104,518,1116,545,1 1097,509,1110,541,1 1089,500,1102,530,1 1093,489,1103,517,1 1079,483,1089,513,1 1073,488,1085,519,1 1069,481,1078,510,1 1066,495,1075,522,1 1055,499,1067,529,1 1038,504,1050,536,1 1046,501,1057,533,1 1031,486,1041,517,1 1036,479,1047,506,1 1045,480,1054,506,1 1056,484,1063,511,1 982,485,991,510,1 1061,542,1090,570,3 1012,701,1030,743,3 1029,676,1045,712,3 1057,669,1080,709,3 996,666,1009,702,3 988,639,1002,674,3 946,602,963,635,3 977,564,991,592,3 963,538,977,566,3 945,573,963,598,3 825,498,845,517,3 865,489,890,508,3 856,472,879,488,10 790,479,806,504,10 917,531,934,556,10 942,519,959,542,10 964,540,976,567,3 1043,579,1062,610,10 1080,647,1116,671,3 1072,628,1108,652,3 970,680,990,713,1 964,668,982,703,1 1280,564,1298,592,3 1303,568,1318,594,3 1310,563,1327,594,3 1326,566,1340,596,3 1220,550,1237,578,3 1145,560,1181,582,3 873,511,894,533,3 918,419,927,442,1 917,382,925,404,1 1064,382,1072,406,1 1122,378,1127,399,1 1127,378,1131,399,1 1174,388,1183,411,1 1261,382,1269,403,1 1208,386,1216,406,1 1132,372,1140,394,1 1129,379,1135,399,1 1192,403,1200,424,1 1186,393,1196,414,1 1114,359,1123,380,1 1106,358,1112,377,1 1110,356,1117,375,1 1170,370,1181,382,2 1130,351,1140,372,1 1010,335,1017,352,1 1004,362,1014,377,2 987,369,998,391,1 907,352,915,373,1 877,350,885,369,1 830,377,840,400,1 806,365,814,384,1 912,416,919,443,1 913,403,920,424,1 937,388,945,411,1 985,391,994,417,1 1021,388,1030,417,1 1042,401,1051,427,1 966,373,972,393,1 957,372,965,392,1 956,366,965,390,1 882,372,888,392,1 858,359,864,380,1 808,325,814,343,1 787,343,793,357,1 722,353,731,369,1 690,312,697,328,1 699,323,706,336,1 749,339,754,355,1 762,332,766,349,1 1020,633,1037,662,10 1027,676,1043,711,3 1068,713,1083,746,3 1057,672,1079,705,3 1086,697,1110,740,3 1046,721,1070,760,10 1146,631,1177,667,10 1087,574,1095,616,1 67,658,93,706,1 195,512,212,548,1 181,524,193,559,1 177,518,191,550,1 123,538,140,574,1 0,433,13,462,1 149,506,181,530,3 49,505,82,534,10 26,485,35,517,1 0,492,16,525,1 10,544,30,565,2 0,388,11,412,1 35,367,44,389,1 183,369,191,388,1 204,362,213,380,1 224,383,235,401,1 253,379,261,396,1 293,436,305,458,1 340,365,350,385,1 350,359,357,374,1 305,353,312,369,1 262,348,270,367,1 296,340,304,357,1 286,340,293,356,1 286,348,293,364,1 309,343,317,359,1 391,310,397,328,1 372,328,377,344,1 376,328,384,344,1 377,338,383,357,1 368,335,374,350,1 382,337,389,356,1 210,336,217,355,1 221,334,228,350,1 216,328,221,346,1 246,314,252,332,1 258,300,262,314,1 264,300,270,313,1 229,336,235,355,1 236,336,241,354,1 239,335,245,354,1 243,337,249,355,1 248,336,254,352,1 253,335,259,356,1 257,337,264,355,1 254,357,276,368,3 304,362,312,374,10 294,347,302,359,10 332,361,339,374,10 352,367,358,380,10 335,375,354,385,3 348,343,357,355,10 352,328,359,340,10 284,314,300,322,3 289,307,295,323,1 292,444,306,466,10 153,498,167,525,1"


# # # text_by_line = "./YOLOv4-for-studying/dataset/Visdrone_DATASET/VisDrone2019-DET-test-dev/images/9999938_00000_d_0000207.jpg 76,14,82,21,1 39,54,54,75,4 98,40,103,49,1 136,32,164,47,4 179,21,207,33,4 76,110,82,118,1 73,131,79,139,1 45,185,62,208,4 61,175,80,203,4 96,165,116,192,4 131,154,149,178,4 149,152,165,173,4 165,145,182,171,4 183,141,199,167,4 214,130,233,155,4 264,114,281,138,4 296,103,314,130,4 207,221,225,247,4 240,207,258,234,4 256,199,275,228,4 273,194,291,221,4 325,183,343,207,4 288,186,310,218,5 75,153,105,172,5 0,136,11,177,9 352,0,355,10,1 356,0,360,8,1 306,74,310,82,1 298,98,302,104,1 265,169,285,196,4 249,176,268,203,4 262,272,280,296,4 208,287,228,315,4 131,315,151,343,4 310,300,313,308,1 309,314,312,322,1 327,324,332,332,1 186,382,190,391,1 187,388,194,398,1 165,381,169,388,1 221,332,235,349,5 287,342,302,364,5 280,413,286,422,1 274,435,280,445,1 290,421,297,430,1 304,411,308,419,1 309,406,314,417,1 306,408,310,419,1 287,435,294,444,1 282,438,286,447,1 292,499,296,507,1 289,515,293,525,1 298,514,303,523,1 309,511,313,521,1 297,526,301,534,1 282,539,289,549,1 296,536,302,546,1 309,529,314,539,1 326,524,330,534,1 349,546,358,553,1 345,538,352,547,1 350,528,355,536,1 287,510,292,518,1 305,499,311,509,1 306,521,312,533,1 337,532,341,541,1 277,538,283,546,1 251,583,257,590,1 269,589,281,594,2 286,615,290,623,1 293,613,299,619,1 352,611,358,619,1 331,624,338,631,1 303,684,310,690,1 368,731,379,739,1 492,703,497,712,1 395,779,400,785,2 387,750,393,755,2 389,757,395,762,2 391,716,397,722,2 387,721,391,727,2 376,719,381,724,2 382,724,388,731,2 386,711,392,718,2 378,704,383,712,2 390,680,397,687,2 383,679,389,688,2 458,649,463,656,2 451,652,458,660,2 572,581,579,589,2 469,560,474,568,2 484,565,488,569,2 496,561,500,566,2 517,540,523,547,1 599,560,603,569,1 380,557,385,565,1 400,542,406,550,1 412,537,416,547,1 409,545,413,552,1 421,542,425,549,1 414,543,418,551,1 495,501,501,509,1 488,504,492,513,1 489,515,492,522,1 493,509,498,518,1 501,503,506,513,1 585,463,591,468,1 505,433,509,438,1 524,435,528,440,1 403,404,501,466,9 460,475,464,484,1 456,480,461,489,1 484,302,574,357,9 542,331,579,355,5 369,393,412,424,5 500,346,540,373,5 463,366,496,388,4 394,296,412,326,5 557,255,562,268,1 404,156,426,182,4 379,137,396,159,4 348,149,366,174,4 442,117,460,140,4 496,118,518,147,5 535,29,556,55,4 564,23,580,44,4 579,17,597,40,4 435,63,452,88,4 389,74,406,99,4 500,96,518,119,4 517,93,535,114,4 542,108,562,134,4 585,93,607,118,4 487,103,506,126,4 498,189,515,214,4 514,181,535,211,4 529,175,550,201,4 465,200,483,226,4 572,73,593,99,5 459,27,506,56,5 504,16,531,33,5 556,350,561,360,1 544,353,549,363,1 415,46,445,61,4 378,62,410,77,4 359,22,363,31,1 377,2,381,11,1 380,0,384,8,1 395,10,401,19,1 411,26,416,34,1 406,31,409,39,1 403,34,406,41,1 438,18,441,25,1 445,7,449,15,1 454,12,459,18,1 471,0,475,9,1 477,8,482,19,1 489,9,493,15,1 536,7,539,16,1 441,0,445,7,1 400,32,403,40,1 443,7,447,17,1 454,8,458,16,1 492,17,496,25,1 457,30,461,39,1 410,41,414,47,1 408,45,411,54,1 414,43,417,50,1 365,47,368,55,1 632,1,651,20,4 603,64,622,86,4 716,113,739,139,4 592,12,610,35,4 636,84,653,104,4 740,33,772,72,9 734,60,763,100,6 774,118,807,159,9 790,97,823,141,9 828,87,849,111,4 755,97,759,106,1 765,98,769,106,1 761,88,766,96,1 707,140,723,160,4 661,154,673,175,5 647,163,660,182,5 545,202,560,220,4 629,186,643,204,4 615,177,626,193,4 646,184,658,198,4 691,257,721,285,4 666,192,669,203,1 676,188,679,196,1 752,177,764,184,10 738,224,743,231,1 743,212,746,219,1 735,221,737,230,1 739,285,742,292,1 733,286,739,295,1 712,316,716,325,1 603,313,606,325,1 590,316,623,337,4 684,236,688,248,1 725,414,730,423,1 715,420,718,427,1 718,417,721,425,1 719,408,723,416,1 775,492,780,501,1 685,772,689,777,2 690,772,695,779,2 699,762,703,770,2 730,750,735,755,2 712,757,716,762,2 733,745,737,749,2 729,744,732,748,2 724,750,729,757,2 719,757,725,765,1 696,769,699,775,1 742,740,745,752,1 738,745,741,754,1 789,14,793,25,1 840,32,844,39,1 844,29,847,37,1 854,56,858,68,1 858,68,862,75,1 865,60,869,69,1 877,65,880,73,1 864,55,867,61,1 859,18,863,26,2 886,96,889,104,1 893,92,898,99,1 901,112,904,119,1 904,113,907,120,1 914,132,917,140,1 919,138,922,146,1 924,138,926,146,1 918,133,921,140,1 925,132,927,141,1 862,112,866,118,1 857,103,860,111,1 876,118,879,126,1 873,121,875,125,1 879,118,884,125,1 892,127,896,136,1 888,128,891,137,1 884,145,887,152,1 903,129,906,138,1 906,134,909,142,1 909,141,912,148,1 943,146,947,154,1 944,152,947,160,1 947,156,950,166,1 953,149,956,157,1 893,156,898,163,1 898,164,902,173,1 892,161,896,169,1 884,164,888,174,1 877,160,881,169,1 873,161,876,169,1 923,196,928,205,1 918,202,922,210,1 949,81,953,90,1 952,97,957,104,1 925,0,940,10,4 941,16,969,51,6 953,165,956,169,2 958,164,960,170,2 976,162,980,170,1 913,201,916,208,1 865,236,869,245,1 854,236,858,244,1 851,238,854,244,1 845,256,850,264,1 905,202,909,208,1 910,205,913,212,1 854,171,867,196,4 827,199,857,230,5 790,222,823,244,5 748,234,792,262,5 843,218,846,227,1 838,229,843,240,1 838,222,841,229,1 823,232,828,241,1 795,244,799,253,1 820,274,826,287,1 835,279,840,291,1 796,313,802,321,1 885,271,889,279,1 916,255,919,260,1 945,216,949,226,1 951,213,954,220,1 931,220,935,223,1 935,225,939,230,1 946,233,950,244,1 866,311,871,318,1 903,313,909,322,1 856,320,861,329,1 847,318,851,325,1 978,298,982,306,1 827,368,830,376,1 946,377,951,384,1 952,376,956,384,1 950,373,953,381,1 836,430,840,438,1 875,435,880,445,1 894,512,900,521,1 835,442,840,448,2 840,449,846,457,2 940,556,944,565,1 815,485,820,493,1 803,526,807,535,1 808,532,810,539,1 816,569,820,576,1 810,563,813,569,1 811,575,815,582,1 873,561,877,570,1 868,557,872,566,1 835,706,839,712,1 852,687,858,695,2 863,681,870,688,2 828,699,832,706,2 834,692,838,700,2 840,694,846,701,2 803,712,807,719,2 788,718,793,727,2 793,716,798,726,2 798,712,804,722,2 778,721,784,729,2 1116,680,1123,687,2 1123,686,1128,694,2 1113,676,1118,682,2 1034,611,1040,618,2 1039,616,1044,622,2 1057,643,1062,648,2 1062,648,1067,653,2 1069,657,1075,663,2 1002,577,1007,586,2 1008,579,1013,586,2 1171,706,1180,715,1 1183,708,1190,715,1 1188,713,1198,720,1 1183,727,1192,734,1 1170,730,1180,737,1 1171,724,1181,730,1 1190,722,1197,729,1 1179,722,1184,728,1 1180,718,1184,723,1 1127,648,1134,656,1 1151,627,1159,634,1 1122,609,1128,616,1 1142,645,1150,652,1 1139,648,1147,656,1 1121,655,1131,662,1 1206,609,1216,615,1 1143,622,1151,627,1 1148,624,1153,633,1 1148,633,1155,639,1 1120,611,1124,617,1 1070,614,1075,622,1 1094,552,1103,559,1 1103,544,1111,551,1 1096,521,1100,532,1 1123,531,1126,541,1 1108,535,1116,544,1 1055,520,1060,529,1 1126,530,1132,538,1 1117,533,1121,542,1 1202,430,1209,441,1 1161,429,1166,441,1 1155,428,1162,438,1 1095,415,1103,423,1 1108,389,1117,400,1 1088,379,1095,390,1 1119,383,1125,392,1 1107,375,1113,383,1 1023,390,1028,400,1 1067,393,1071,399,1 1072,394,1077,400,1 1203,391,1215,398,1 1116,348,1122,357,1 1124,351,1127,359,1 1125,348,1131,356,1 1179,357,1186,365,2 1177,356,1185,360,2 1174,352,1183,356,2 1171,349,1181,356,2 1175,345,1181,351,2 1167,314,1172,319,2 1157,311,1163,317,2 1124,259,1126,264,2 1176,223,1182,231,1 1183,213,1189,220,1 1158,202,1164,212,1 1156,189,1163,199,1 1122,146,1128,156,1 1019,149,1025,161,1 1006,195,1011,202,1 1016,185,1019,191,1 1058,185,1063,193,1 1083,184,1088,192,1 1091,166,1096,171,1 985,159,989,165,2 988,164,992,170,2 1005,169,1008,174,2 1052,176,1055,182,1 1050,187,1053,193,1 1058,177,1061,182,1 1072,176,1077,181,1 1078,182,1081,189,1 1072,184,1076,188,1 1079,174,1082,182,1 1047,185,1049,190,2 1051,182,1056,186,2 1038,175,1042,182,2 1029,180,1034,188,2 1020,177,1027,184,2 1026,172,1029,176,2 1030,175,1034,181,2 1011,174,1015,180,2 999,173,1004,180,2 1000,181,1004,186,2 1007,178,1013,187,2 1004,175,1009,181,2 1126,166,1132,172,2 1005,42,1009,51,1 1008,65,1013,74,1 1109,95,1114,104,1 1111,90,1115,100,1 1106,84,1110,92,1 1107,88,1110,96,1 1033,11,1037,19,1 1027,12,1031,20,1 1030,13,1035,22,1 1325,59,1330,69,1 1338,62,1341,72,1 1343,63,1347,71,1 1232,278,1241,288,1 1284,348,1289,357,1 1277,345,1285,355,1 1295,377,1301,385,1 1302,376,1307,382,1 1133,146,1136,155,1 1335,54,1338,62,1 1331,57,1333,65,1 1358,283,1364,292,1 1250,452,1257,462,1 1259,459,1263,469,1 1276,501,1282,510,1 1282,506,1289,514,1 1266,557,1274,565,1 1302,560,1309,565,1 1343,557,1353,565,1 1296,511,1306,517,2 1305,513,1310,521,2 1235,621,1243,629,1 1244,621,1248,630,1 1385,670,1395,675,1 1369,630,1376,635,1 1356,745,1360,754,1 1362,741,1367,748,1 1250,614,1254,624,1 1246,617,1249,625,1"
# # text_by_line = './YOLOv4-for-studying/dataset/Visdrone_DATASET/VisDrone2019-DET-train/images/9999982_00000_d_0000034.jpg 1168,406,1193,435,5 590,330,657,352,0'


# # text = text_by_line.split()


# # bboxes = []
# # for t in text:
# #     if not t.replace(',', '').isnumeric():
# #         temp_path   = os.path.relpath(t, RELATIVE_PATH)
# #         temp_path   = os.path.join(PREFIX_PATH, temp_path)
# #         image_path  = temp_path.replace('\\','/')
# #     else:
# #         t = list(map(int, t.split(',')))
# #         bboxes.append(t)
# # bboxes = np.array(bboxes)

# # gpus = tf.config.experimental.list_physical_devices('GPU')
# # if len(gpus) > 0:
# #     print(f'GPUs {gpus}')
# #     try: tf.config.experimental.set_memory_growth(gpus[0], True)
# #     except RuntimeError: pass

# # yolo = Load_YOLOv4_Model()
# # pred_image = detect_image(yolo, image_path, show=False, show_label=False, save=False, CLASSES_PATH=YOLO_CLASS_PATH, score_threshold=VALIDATE_SCORE_THRESHOLD)

# # image = cv2.imread(image_path)
# # image = draw_bbox(image, bboxes, YOLO_CLASS_PATH, show_label=False)
# # cv2.imshow('truth', cv2.resize(image,(1280, 720)))
# # cv2.imshow("prediction", cv2.resize(pred_image,(1280, 720)))

# # if cv2.waitKey() == 'q':
# #     pass



# # class PostprocessPredictions:
# #     """Utilities for calculating IOU/IOS based match for given ObjectPredictions"""
# #     def __init__(   self,
# #                     match_metric: str = "IOS",
# #                     match_threshold: float = 0.5):
# #         self.match_threshold = match_threshold
# #         self.match_metric = match_metric

# #     def __call__(self, bboxes):
# #         #First settings
# #         bboxes = np.array(bboxes)
# #         diff_classes_in_pred = list(set(bboxes[:, 5]))
# #         best_bboxes = []
# #         #Do GREEDY-NMM for each specific class
# #         for cls in diff_classes_in_pred:
# #             cls_mask = np.array(bboxes[:, 5] == cls)
# #             cls_bboxes = bboxes[cls_mask]
# #             #Select best bbox of same class for each object in image
# #             while len(cls_bboxes) > 0:
# #                 max_conf_bbox_idx = np.argmax(cls_bboxes[:, 4])                 #index of best bbox : highest confidence score
# #                 best_bbox = cls_bboxes[max_conf_bbox_idx]
# #                 best_bboxes.append(best_bbox)
# #                 cls_bboxes = np.delete(cls_bboxes, max_conf_bbox_idx, axis=0)   #remove best bbox from list of bboxes

# #                 assert self.match_metric in ['IOU', 'IOS']
# #                 if self.match_metric == "IOU":
# #                     iou = bboxes_iou_from_minmax(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])  #calculate list of iou between best bbox and other bboxes
# #                     weight = np.ones(len(iou), dtype=np.float32)   
# #                     iou_mask = np.array(iou > self.match_threshold)
# #                     weight[iou_mask] = 0.0 
# #                 if self.match_metric == "IOS":
# #                     best_bbox[np.newaxis, :4]
# #                     cls_bboxes[:, :4]
# #                     intersect_tf = np.maximum(best_bbox[np.newaxis, :2], cls_bboxes[:, :2])
# #                     intersect_br = np.minimum(best_bbox[np.newaxis, 2:4], cls_bboxes[:, 2:4])
# #                     intersect_area = np.multiply.reduce(np.maximum(intersect_br - intersect_tf, 0.0), axis=-1)
                    
# #                     best_bbox_area =  np.multiply.reduce(np.maximum(best_bbox[np.newaxis, 2:4] - best_bbox[np.newaxis, :2], 0.0), axis=-1)
# #                     cls_bboxes_area = np.multiply.reduce(np.maximum(cls_bboxes[:, 2:4] - cls_bboxes[:, :2], 0.0), axis=-1)
# #                     bboxes_smaller_area = np.minimum(cls_bboxes_area, best_bbox_area)

# #                     ios = intersect_area / bboxes_smaller_area
# #                     weight = np.ones(len(ios), dtype=np.float32)   
# #                     ios_mask = np.array(ios > self.match_threshold)
# #                     weight[ios_mask] = 0.0 

# #                 cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight    #detele bboxes predicting same objects
# #                 score_mask = cls_bboxes[:, 4] > 0.
# #                 cls_bboxes = cls_bboxes[score_mask]
# #         return best_bboxes



# # #Get image, slice image, make predictions, merge predictions
# # class PredictionResult:
# #     def __init__(self, model, image, input_size, score_threshold, iou_threshold):
# #         self.model = model
# #         self.image = image
# #         self.input_size = input_size
# #         self.score_threshold = score_threshold
# #         self.iou_threshold = iou_threshold
        
# #         self.postprocess_slicing_predictions_into_original_image = PostprocessPredictions(match_metric="IOS", match_threshold=0.5)


# #         # self.make_prediciton()

# #     #get sliced images, make each prediction, scale prediction to original image
# #     def make_sliced_predictions(self):
# #         sliced_images_obj = Original_Image_Into_Sliced_Images(np.copy(self.image))
# #         sliced_images = sliced_images_obj.load_sliced_images_for_export()
# #         sliced_image_with_prediction_list = []
# #         for sliced_image in sliced_images:
# #             image_data = cv2.cvtColor(np.copy(sliced_image.image), cv2.COLOR_BGR2RGB)
# #             image_data = image_preprocess(image_data, self.input_size)                  #scale to size 416
# #             image_data = image_data[np.newaxis, ...].astype(np.float32)                         #reshape [1, 416, 416, 3]
# #             pred_bbox = self.model(image_data, training=False)
# #             pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]               #reshape to [3, bbox_num, 85]
# #             pred_bbox = tf.concat(pred_bbox, axis=0)                                            #concatenate to [bbox_num, 85]
# #             pred_bboxes = postprocess_boxes(pred_bbox, np.copy(sliced_image.image), self.input_size, self.score_threshold)      #scale to origional and select valid bboxes
# #             if len(pred_bboxes) == 0:
# #                 continue
# #             pred_bboxes = tf.convert_to_tensor(nms(pred_bboxes, self.iou_threshold, method='nms'))                                       #Non-maximum suppression: xymin, xymax        
# #             sliced_image.predictions = pred_bboxes
# #             sliced_image_with_prediction_list.append(sliced_image)
# #         return sliced_image_with_prediction_list

# #     #make prediction in original image, make prediciton in sliced images and merge into original image
# #     def make_prediciton(self):
# #         image_data = cv2.cvtColor(np.copy(self.image), cv2.COLOR_BGR2RGB)
# #         image_data = image_preprocess(image_data, self.input_size)                  #scale to size 416
# #         image_data = image_data[np.newaxis, ...].astype(np.float32)                         #reshape [1, 416, 416, 3]
# #         pred_bbox = self.model(image_data, training=False)
# #         pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]               #reshape to [3, bbox_num, 85]
# #         pred_bbox = tf.concat(pred_bbox, axis=0)                                            #concatenate to [bbox_num, 85]
# #         pred_bboxes = postprocess_boxes(pred_bbox, np.copy(self.image), self.input_size, self.score_threshold)      #scale to origional and select valid bboxes
# #         pred_bboxes = tf.convert_to_tensor(nms(pred_bboxes, self.iou_threshold, method='nms'))                                       #Non-maximum suppression: xymin, xymax              

# #         # image_test = draw_bbox(np.copy(self.image),pred_bboxes, YOLO_CLASS_PATH, show_label=True)
# #         # cv2.imshow("Test before slicing prediciton", cv2.resize(image_test, [1280, 720]))

# #         sliced_image_with_prediction_list = self.make_sliced_predictions()
# #         for sliced_image_with_prediction in sliced_image_with_prediction_list:
# #             pred_offset = np.concatenate([sliced_image_with_prediction.starting_point, sliced_image_with_prediction.starting_point, [0], [0]], axis=-1)[np.newaxis, :]
# #             sliced_image_with_prediction.predictions = sliced_image_with_prediction.predictions + pred_offset
# #             pred_bboxes = np.concatenate([pred_bboxes, sliced_image_with_prediction.predictions], axis=0)
# #             pred_bboxes = self.postprocess_slicing_predictions_into_original_image(pred_bboxes)

# #         # image_test = draw_bbox(np.copy(self.image),pred_bboxes, YOLO_CLASS_PATH, show_label=True)
# #         # cv2.imshow("Test after slicing prediciton", cv2.resize(image_test, [1280, 720]))
# #         # if cv2.waitKey() == "q":
# #         #     pass
# #         # cv2.destroyAllWindows()
# #         return pred_bboxes




# import os
# from YOLOv4_config import *
# from YOLOv4_utils import *
# from YOLOv4_slicing import *


# # text_by_line = './YOLOv4-for-studying/dataset/Visdrone_DATASET/VisDrone2019-DET-train/images/9999982_00000_d_0000034.jpg 1168,406,1193,435,5 590,330,657,352,0'
# # text_by_line = './YOLOv4-for-studying/dataset/Visdrone_DATASET/VisDrone2019-DET-train/images/9999999_00650_d_0000295.jpg 147,344,173,359,4 177,326,213,344,4 188,303,221,323,4 202,245,232,258,4 283,232,308,243,4 236,207,248,228,5 204,203,227,217,4 273,195,294,205,4 265,178,286,188,5 209,162,229,170,4 199,151,214,159,5 200,146,219,152,4 258,206,260,213,1 298,89,304,95,4 200,93,214,98,4 204,87,215,92,4 237,50,245,53,4 233,45,242,49,4 231,37,239,39,4 230,27,237,30,4 424,91,436,98,4 383,58,393,63,4 342,25,351,32,6 382,48,395,58,6 197,64,207,68,4 198,47,205,59,9 227,49,234,60,9'
# # text_by_line = './YOLOv4-for-studying/dataset/Visdrone_DATASET/VisDrone2019-DET-train/images/9999965_00000_d_0000023.jpg 682,661,765,695,3 904,567,934,644,3 958,484,1023,558,3 979,286,1013,359,3 818,554,890,584,3 799,494,891,531,3 821,455,890,484,3 813,402,890,432,3 816,358,893,388,3 817,309,892,338,3 825,267,896,299,3 815,227,885,259,3 697,152,773,193,3 810,49,890,86,3 805,10,885,42,3 988,83,1019,156,4 824,190,896,219,4 331,25,369,103,4 800,138,898,183,5 689,287,774,326,3 700,344,777,379,3 697,402,777,432,3 697,502,774,537,3 701,550,765,579,4 544,389,598,586,8 314,625,346,702,3 318,535,351,609,3 321,438,357,515,3 329,333,363,405,3 329,227,356,302,3 317,119,364,213,5 308,733,346,786,3 633,77,647,117,9 633,87,648,106,1 954,69,969,86,0 1003,462,1018,474,0 1041,445,1055,456,0 917,133,928,146,0 740,740,753,761,0 1016,681,1028,695,0 1053,471,1062,483,0'
# text_by_line = './YOLOv4-for-studying/dataset/Visdrone_DATASET/VisDrone2019-DET-train/images/9999998_00038_d_0000030.jpg 900,957,1001,1034,3 939,915,1033,982,3 1307,1073,1430,1156,3 1200,996,1303,1086,3 1267,1106,1375,1184,3 1163,1038,1264,1119,3 1138,1087,1231,1160,3 1247,1157,1360,1239,3 1246,1220,1317,1272,3 787,1002,909,1089,3 311,624,392,671,3 358,639,504,721,-1 265,588,345,638,3 86,723,170,797,-1 6,754,92,825,-1 95,796,197,868,3 136,844,244,917,3 170,874,282,955,3 197,916,301,992,3 234,997,331,1075,3 270,1009,378,1102,3 318,1051,419,1138,3 348,1079,454,1164,3 393,1124,488,1214,3 1566,1013,1583,1046,0 1516,737,1552,766,2 1362,643,1388,665,2 1309,568,1384,636,-1 1388,597,1509,683,-1 1486,640,1571,721,-1 1567,709,1596,740,2 1682,729,1712,747,2 1677,742,1709,759,2 1673,755,1706,776,2 1624,803,1661,832,2 1643,760,1699,811,-1 1605,512,1657,561,3 1693,556,1741,616,3 1747,581,1802,642,3 1793,594,1852,645,3 1857,616,1913,668,3 1247,887,1324,976,3 1675,1358,1780,1435,3 1768,1434,1890,1499,3 1899,638,1957,695,3'

# text = text_by_line.split()
# bboxes = []
# for t in text:
#     if not t.replace(',', '').replace('-1','').isnumeric():
#         temp_path   = os.path.relpath(t, RELATIVE_PATH)
#         temp_path   = os.path.join(PREFIX_PATH, temp_path)
#         image_path  = temp_path.replace('\\','/')
#     else:
#         t = list(map(int, t.split(',')))
#         bboxes.append(t)
# image = cv2.imread(image_path)
# bboxes = np.array(bboxes)

# yolo = Load_YOLOv4_Model()

# sliced_images_obj = Original_Image_Into_Sliced_Images(image, bboxes)
# sliced_images = sliced_images_obj.load_sliced_images_for_export()

# # image_test =draw_bbox(np.copy(image), np.copy(bboxes), "YOLOv4-for-studying/dataset/Visdrone_DATASET/visdrone_class_names_test.txt", show_label=False)
# # cv2.imshow("ground truth image", cv2.resize(image_test, [1280, 720]))


# # for sliced_image in sliced_images:
# #     image_test = draw_bbox(np.copy(sliced_image.image), sliced_image.bboxes, "YOLOv4-for-studying/dataset/Visdrone_DATASET/visdrone_class_names_test.txt", show_label=True)
# #     cv2.imshow("Sliced image", cv2.resize(image_test, [416, 416]))
# #     if cv2.waitKey() == "q":
# #         pass
# #     cv2.destroyAllWindows()




# input_size = [416, 416]
# score_threshold = 0.425
# iou_threshold = 0.5

# image_test =draw_bbox(np.copy(image), np.copy(bboxes), "YOLOv4-for-studying/dataset/Visdrone_DATASET/visdrone_class_names_test.txt", show_label=True)
# cv2.imshow("ground truth", cv2.resize(image_test, [1280, 720]))

# prediction_obj = PredictionResult(yolo, image, input_size, SLICED_IMAGE_SIZE, score_threshold, iou_threshold)
# pred_bboxes = tf.convert_to_tensor(prediction_obj.make_prediciton()[0])

# if EVALUATION_DATASET_TYPE == "VISDRONE":
#     bboxes = tf.cast(np.copy(bboxes), dtype=tf.float64)
#     ignored_bbox_mask   = bboxes[:,4]>-0.5
#     ignored_bboxes      = tf.expand_dims(bboxes, axis=0)[tf.expand_dims(tf.math.logical_not(ignored_bbox_mask), axis=0)]
#     other_bbox_mask     = bboxes[:,4]<9.5
#     other_bboxes        = tf.expand_dims(bboxes, axis=0)[tf.expand_dims(tf.math.logical_not(other_bbox_mask), axis=0)]

    
#     removed_ignored_mask = tf.convert_to_tensor(np.zeros(np.array(tf.shape(pred_bboxes)[0])), dtype=tf.bool)
#     if tf.shape(ignored_bboxes)[0] != 0:
#         pred_bboxes_temp = tf.expand_dims(pred_bboxes[:, :4], axis=1)       #shape [total_bboxes, 1, 4]
#         ignored_bboxes = tf.expand_dims(ignored_bboxes[:, :4], axis=0)      #shape [1, num_bboxes, 4]
#         intersect_tf = tf.maximum(pred_bboxes_temp[..., :2], ignored_bboxes[..., :2])
#         intersect_br = tf.minimum(pred_bboxes_temp[..., 2:], ignored_bboxes[..., 2:])
#         intersection = tf.maximum(intersect_br - intersect_tf, 0.0)
#         intersection_area = tf.math.reduce_sum(tf.math.reduce_prod(intersection, axis=-1), axis=-1, keepdims=True)      #shape [num_pred_bboxes, 2]
#         pred_bboxes_area = tf.math.reduce_prod(tf.maximum(pred_bboxes_temp[...,2:] - pred_bboxes_temp[...,:2], 0.0), axis=-1) #shape [num_pred_bboxes, 1]
#         removed_ignored_mask = tf.reduce_max(intersection_area / pred_bboxes_area , axis=-1) > 0.58
    
#     #getting mask of bboxes that overlap "other" class
#     removed_other_mask = tf.convert_to_tensor(np.zeros(np.array(tf.shape(pred_bboxes)[0])), dtype=tf.bool)
#     if tf.shape(other_bboxes)[0] != 0:
#         pred_bboxes_temp = tf.expand_dims(pred_bboxes[:, :4], axis=1)      #shape [total_bboxes, 1, 4]
#         other_bboxes = tf.expand_dims(other_bboxes[:, :4], axis=0)         #shape [1, num_bboxes, 4]
#         ious = bboxes_iou_from_minmax(pred_bboxes_temp, other_bboxes)   #shape [total_bboxes, num_bboxes]
#         max_ious = tf.reduce_max(ious, axis=-1)
#         removed_other_mask = max_ious > 0.5          #removed_other = True when iou > threshold
#     #getting mask of removed bboxes
#     removed_bbox_mask = tf.math.logical_or(removed_ignored_mask, removed_other_mask)
#     pred_bboxes = tf.expand_dims(pred_bboxes, axis=0)[tf.expand_dims(tf.math.logical_not(removed_bbox_mask), axis=0)]


# image_test = draw_bbox(np.copy(image),pred_bboxes, YOLO_CLASS_PATH, show_label=True)
# cv2.imshow("Test after slicing prediciton", cv2.resize(image_test, [1280, 720]))
# if cv2.waitKey() == "q":
#     pass
# cv2.destroyAllWindows()




# # for sliced_image in sliced_images:
# #     image_data = cv2.cvtColor(np.copy(sliced_image.image), cv2.COLOR_BGR2RGB)
# #     image_data = image_preprocess(image_data, input_size)                  #scale to size 416
# #     image_data = image_data[np.newaxis, ...].astype(np.float32)                         #reshape [1, 416, 416, 3]
# #     pred_bbox = yolo(image_data, training=False)
# #     pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]               #reshape to [3, bbox_num, 85]
# #     pred_bbox = tf.concat(pred_bbox, axis=0)                                            #concatenate to [bbox_num, 85]
# #     pred_bboxes = postprocess_boxes(pred_bbox, np.copy(sliced_image.image), input_size, score_threshold)      #scale to origional and select valid bboxes
# #     pred_bboxes = tf.convert_to_tensor(nms(pred_bboxes, iou_threshold, method='nms'))                                       #Non-maximum suppression

# #     if EVALUATION_DATASET_TYPE == "VISDRONE":
# #         bboxes = tf.cast(sliced_image.bboxes, dtype=tf.float64)
# #         ignored_bbox_mask   = bboxes[:,4]>-0.5
# #         ignored_bboxes      = tf.expand_dims(bboxes, axis=0)[tf.expand_dims(tf.math.logical_not(ignored_bbox_mask), axis=0)]
# #         other_bbox_mask     = bboxes[:,4]<9.5
# #         other_bboxes        = tf.expand_dims(bboxes, axis=0)[tf.expand_dims(tf.math.logical_not(other_bbox_mask), axis=0)]

        
# #         removed_ignored_mask = tf.convert_to_tensor(np.zeros(np.array(tf.shape(pred_bboxes)[0])), dtype=tf.bool)
# #         if tf.shape(ignored_bboxes)[0] != 0:
# #             pred_bboxes_temp = tf.expand_dims(pred_bboxes[:, :4], axis=1)       #shape [total_bboxes, 1, 4]
# #             ignored_bboxes = tf.expand_dims(ignored_bboxes[:, :4], axis=0)      #shape [1, num_bboxes, 4]
# #             intersect_tf = tf.maximum(pred_bboxes_temp[..., :2], ignored_bboxes[..., :2])
# #             intersect_br = tf.minimum(pred_bboxes_temp[..., 2:], ignored_bboxes[..., 2:])
# #             intersection = tf.maximum(intersect_br - intersect_tf, 0.0)
# #             intersection_area = tf.math.reduce_sum(tf.math.reduce_prod(intersection, axis=-1), axis=-1, keepdims=True)      #shape [num_pred_bboxes, 2]
# #             pred_bboxes_area = tf.math.reduce_prod(tf.maximum(pred_bboxes_temp[...,2:] - pred_bboxes_temp[...,:2], 0.0), axis=-1) #shape [num_pred_bboxes, 1]
# #             removed_ignored_mask = tf.reduce_max(intersection_area / pred_bboxes_area , axis=-1) > 0.58
        
# #         #getting mask of bboxes that overlap "other" class
# #         removed_other_mask = tf.convert_to_tensor(np.zeros(np.array(tf.shape(pred_bboxes)[0])), dtype=tf.bool)
# #         if tf.shape(other_bboxes)[0] != 0:
# #             pred_bboxes_temp = tf.expand_dims(pred_bboxes[:, :4], axis=1)      #shape [total_bboxes, 1, 4]
# #             other_bboxes = tf.expand_dims(other_bboxes[:, :4], axis=0)         #shape [1, num_bboxes, 4]
# #             ious = bboxes_iou_from_minmax(pred_bboxes_temp, other_bboxes)   #shape [total_bboxes, num_bboxes]
# #             max_ious = tf.reduce_max(ious, axis=-1)
# #             removed_other_mask = max_ious > 0.5          #removed_other = True when iou > threshold
# #         #getting mask of removed bboxes
# #         removed_bbox_mask = tf.math.logical_or(removed_ignored_mask, removed_other_mask)
# #         pred_bboxes = tf.expand_dims(pred_bboxes, axis=0)[tf.expand_dims(tf.math.logical_not(removed_bbox_mask), axis=0)]

# #     gt_image = draw_bbox(np.copy(sliced_image.image), sliced_image.bboxes, "YOLOv4-for-studying/dataset/Visdrone_DATASET/visdrone_class_names_test.txt", show_label=False)
# #     pred_image = draw_bbox(gt_image, pred_bboxes, YOLO_CLASS_PATH, show_label=False)
# #     cv2.imshow('truth', cv2.resize(gt_image,(1280, 720)))
# #     cv2.imshow("prediction", cv2.resize(pred_image,(1280, 720)))
# #     if cv2.waitKey() == 'q':
# #         pass
# #     cv2.destroyAllWindows()







# # image = draw_bbox(image, bboxes, "YOLOv4-for-studying/dataset/Visdrone_DATASET/visdrone_class_names_test.txt", show_label=True)
# # cv2.imshow('truth', cv2.resize(image,(1280, 720)))
# # if cv2.waitKey() == 'q':
# #     pass



# # #Create format of each sliced_image object including 4 attributes
# # class SlicedImage:
# #     def __init__(   self,                           
# #                     image: np.ndarray,              #cv2 image in Numpy array
# #                     bboxes,                         #List of [4 coordinates, class_idx]
# #                     starting_point,                 #[xmin, ymin]
# #                     predictions = None):                   #List of [4 coordinates, score, classs_idx]
# #         self.image = image
# #         self.bboxes = bboxes
# #         self. starting_point = starting_point
# #         self.predictions = predictions

# # #Create format of object processed for each original image
# # class Original_Image_Into_Sliced_Images:
# #     def __init__(self, original_image=None, original_bboxes=None):              #inputs as original image and all gt bboxes inside that image
# #         #Setting for slicing original image into set of sliced images
# #         self.original_image = original_image
# #         self.original_bboxes = original_bboxes
# #         self.original_image_height = self.original_image.shape[0]
# #         self.original_image_width = self.original_image.shape[1]
# #         self.sliced_image_size = SLICED_IMAGE_SIZE
# #         self.overlap_ratio = OVERLAP_RATIO
# #         self.min_area_ratio = MIN_AREA_RATIO
        
# #         #List of sliced images
# #         self.sliced_image_list = []







# #         """ Test """
# #         self.slice_image(self.original_image, self.original_bboxes, *self.sliced_image_size, *self.overlap_ratio, self.min_area_ratio)


# #     #Get the bbox coordinate of sliced images in origional image
# #     def get_sliced_image_coordinates(   self,
# #                                         image_width: int,                   
# #                                         image_height: int,
# #                                         slice_width: int,
# #                                         slice_height: int,
# #                                         overlap_width_ratio: float,
# #                                         overlap_height_ratio: float):  
# #         sliced_image_coordinates = []
# #         x_overlap = int(overlap_width_ratio * slice_width)
# #         y_overlap = int(overlap_height_ratio * slice_height)
# #         #Run in y-axis, at each value y, calculate x
# #         y_max = y_min = 0
# #         while y_max < image_height:    
# #             #update new ymax for this iterative
# #             y_max = y_min + slice_height
# #             #run in x-axis, at each value (xmin,xmax), save the patch coordinates
# #             x_min = x_max = 0
# #             while x_max < image_width:
# #                 #update new xmax for this iterative
# #                 x_max = x_min + slice_width
# #                 #if the patch coordinates is outside original image, cut at the borders to inside area inversely
# #                 if y_max > image_height or x_max > image_width:
# #                     xmax = min(image_width, x_max)
# #                     ymax = min(image_height, y_max)
# #                     xmin = max(0, xmax - slice_width)   
# #                     ymin = max(0, ymax - slice_height)
# #                     sliced_image_coordinates.append([xmin, ymin, xmax, ymax])
# #                 else:
# #                     sliced_image_coordinates.append([x_min, y_min, x_max, y_max])
# #                 #update new xmin for next iterative
# #                 x_min = x_max - x_overlap
# #             #update new ymin for next iterative
# #             y_min = y_max - y_overlap
# #         return sliced_image_coordinates


# #     #check if gt_coordinates is inside the sliced image
# #     def check_gt_coordinates_inside_slice(  self, 
# #                                             gt_coordinates,                 #format [xmin, ymin, xmax, ymax]                   
# #                                             sliced_image_coordinates):      #format [xmin, ymin, xmax, ymax]                   
# #         if gt_coordinates[0] >= sliced_image_coordinates[2]:    #if gt is left to sliced_image
# #             return False    
# #         if gt_coordinates[1] >= sliced_image_coordinates[3]:    #if gt is below sliced_image
# #             return False
# #         if gt_coordinates[2] <= sliced_image_coordinates[0]:    #if gt is right to sliced_image
# #             return False        
# #         if gt_coordinates[3] <= sliced_image_coordinates[1]:    #if gt is above sliced_image
# #             return False
# #         return True

# #     #Tranform gt_bboxes in original image into those in sliced images
# #     def process_gt_bboxes_to_sliced_image(  self, 
# #                                             original_gt_bboxes,                         #List of gt bboxes with format [4 coordinates, class_idx]
# #                                             sliced_image_coordinates,                   #format [xmin, ymin, xmax, ymax]
# #                                             min_area_ratio):                            #area ratio to remove gt bbox from sliced image
# #         #Each ground truth bbox is compared to sliced_image_coordinates to create bbox_coordinates inside sliced_image
# #         sliced_image_gt_bboxes = []
# #         for original_gt_bbox in original_gt_bboxes:
# #             if self.check_gt_coordinates_inside_slice(original_gt_bbox[:4], sliced_image_coordinates):
# #                 #Calculate intersection area
# #                 top_left        = np.maximum(original_gt_bbox[:2], sliced_image_coordinates[:2])
# #                 bottom_right    = np.minimum(original_gt_bbox[2:4], sliced_image_coordinates[2:])
# #                 gt_bbox_area = np.multiply.reduce(original_gt_bbox[2:4] - original_gt_bbox[:2])
# #                 intersection_area = np.multiply.reduce(bottom_right - top_left)
# #                 if intersection_area/gt_bbox_area >=min_area_ratio:
# #                     sliced_image_gt_bbox = np.concatenate([top_left - sliced_image_coordinates[:2], bottom_right - sliced_image_coordinates[:2], np.array([original_gt_bbox[4]])])  #minus starting point
# #                     sliced_image_gt_bboxes.append(sliced_image_gt_bbox)
# #         return sliced_image_gt_bboxes


# #     #slice the original image into objects of class SliceImage
# #     def slice_image(self, 
# #                     original_image,                 #original image
# #                     original_gt_bboxes,             #list of original bboxes with shape [4 coordinates, class_idx]
# #                     slice_width,                
# #                     slice_height,
# #                     overlap_width_ratio,
# #                     overlap_height_ratio,
# #                     min_area_ratio):

# #         original_image_height, original_image_width, _ = original_image.shape
# #         if not (original_image_width != 0 and original_image_height != 0):
# #             raise RuntimeError(f"Error from invalid image size: {original_image.shape}")
       
# #         sliced_image_coordinates_list = self.get_sliced_image_coordinates(*[original_image_width, original_image_height], *[slice_width, slice_height], *[overlap_width_ratio, overlap_height_ratio])
        
        
# #         number_images = 0
# #         # iterate over slices
# #         for sliced_image_coordinates in sliced_image_coordinates_list:
# #             # count number of sliced images
# #             number_images += 1
# #             # Extract starting point of the sliced image
# #             starting_point = [sliced_image_coordinates[0], sliced_image_coordinates[1]]
# #             # Extract sliced image
# #             tl_x, tl_y, br_x, br_y = sliced_image_coordinates
# #             sliced_image = np.copy(original_image[tl_y:br_y, tl_x:br_x])
# #             # Extract gt bboxes
# #             sliced_image_gt_bboxes = self.process_gt_bboxes_to_sliced_image(np.copy(original_gt_bboxes), sliced_image_coordinates, min_area_ratio)

# #             if len(sliced_image_gt_bboxes) != 0:
# #                 sliced_image_obj = SlicedImage(sliced_image, sliced_image_gt_bboxes, starting_point)
# #                 self.sliced_image_list.append(sliced_image_obj)
                
                
                
                
#             #     sliced_image = draw_bbox(sliced_image, sliced_image_gt_bboxes, YOLO_CLASS_PATH, show_label=False)
#             # cv2.imshow("test", sliced_image)
#             # if cv2.waitKey() == 'q':
#             #     pass
#             # cv2.destroyAllWindows()
#             # print("OK!")




    















# # #Get the bbox coordinate of slicing patches in origional image
# # def get_slicing_patch_coordinates(  image_width: int,
# #                                     image_height: int,
# #                                     slice_width: int = 512,
# #                                     slice_height: int = 512,
# #                                     overlap_width_ratio: int = 0.2,
# #                                     overlap_height_ratio: int = 0.2):
# #     slicing_patch_coordinates = []
# #     x_overlap = int(overlap_width_ratio * slice_width)
# #     y_overlap = int(overlap_height_ratio * slice_height)
# #     #Run in y-axis, at each value y, calculate x
# #     y_max = y_min = 0
# #     while y_max < image_height:    
# #         #update new ymax for this iterative
# #         y_max = y_min + slice_height
# #         #run in x-axis, at each value (xmin,xmax), save the patch coordinates
# #         x_min = x_max = 0
# #         while x_max < image_width:
# #             #update new xmax for this iterative
# #             x_max = x_min + slice_width
# #             #if the patch coordinates is outside original image, cut at the borders to inside area inversely
# #             if y_max > image_height or x_max > image_width:
# #                 xmax = min(image_width, x_max)
# #                 ymax = min(image_height, y_max)
# #                 xmin = max(0, xmax - slice_width)   
# #                 ymin = max(0, ymax - slice_height)
# #                 slicing_patch_coordinates.append([xmin, ymin, xmax, ymax])
# #             else:
# #                 slicing_patch_coordinates.append([x_min, y_min, x_max, y_max])
            
# #             #update new xmin for next iterative
# #             x_min = x_max - x_overlap
# #         #update new ymin for next iterative
# #         y_min = y_max - y_overlap
# #     return slicing_patch_coordinates


# # def slice_image(original_image: np.ndarray,
# #                 slice_width: int = 512,
# #                 slice_height: int = 512,
# #                 overlap_width_ratio: float = 0.2,
# #                 overlap_height_ratio: float = 0.2,
# #                 min_area_ratio: float = 0.1,):

# #     image_height, image_width, _ = original_image.shape
# #     if not (image_width != 0 and image_height != 0):
# #         raise RuntimeError(f"Error from invalid image size: {image.shape}")
# #     slicing_patch_coordinates = get_slicing_patch_coordinates(*[image_width, image_height], *[slice_width,slice_height], *[overlap_width_ratio,overlap_height_ratio])

    

# #     # # init images and annotations lists
# #     # sliced_image_result = SliceImageResult(original_image_size=[image_height, image_width])

# #     sliced_image_list = [] #List of SlicedImage
# #     number_images = 0
# #     # iterate over slices
# #     for slice_bbox in slicing_patch_coordinates:
# #         number_images += 1

# #         # extract image
# #         tl_x, tl_y, br_x, br_y = slice_bbox
# #         sliced_image = image[tl_y:br_y, tl_x:br_x]


        


# #         """ANNOTATION PROCESSING"""
# #         # # process annotations if coco_annotations is given
# #         # if coco_annotation_list is not None:
# #         #     sliced_coco_annotation_list = process_coco_annotations(coco_annotation_list, slice_bbox, min_area_ratio)
# #         #  # append coco annotations (if present) to coco image
# #         # if coco_annotation_list:
# #         #     for coco_annotation in sliced_coco_annotation_list:
# #         #         coco_image.add_annotation(coco_annotation)

       

# #         # create sliced image and append to sliced_image_result
# #         sliced_image = SlicedImage(
# #             image=sliced_image,
# #             bboxes=[],
# #             starting_point=[slice_bbox[0], slice_bbox[1]],
# #             predictions=[],
# #         )
# #         # sliced_image_result.add_sliced_image(sliced_image)

# #         sliced_image_list.append(slice_image)


# #     return sliced_image_list










# # import numpy as np


# # anchor =    np.array([[ 16 , 28],
# #                         [ 14 , 12],
# #                         [ 29 , 18],
# #                         [123 , 97],
# #                         [  8 , 18],
# #                         [ 53 , 29],
# #                         [ 60 , 60],
# #                         [  5 ,  8],
# #                         [ 27 , 43]])
# # print(anchor)

# # anchor_area = np.multiply.reduce(anchor, axis=-1)
# # print(anchor_area)

# # anchor_n = []
# # while len(anchor):
# #     i = np.argmin(anchor_area)
# #     anchor_n.append(anchor[i])

# #     anchor = np.delete(anchor, i, axis=0)
# #     anchor_area = np.delete(anchor_area, i, axis=0)

# # anchor_n = np.array(anchor_n)  
# # print(anchor_n)