#===============================================================#
#                                                               #
#   File name   : Test.py                                       #
#   Author      : hxnghia99                                     #
#   Created date: April 28th, 2022                              #
#   GitHub      : https://github.com/hxnghia99/Yolov4_Learning  #
#   Description : YOLOv3 image prediction testing               #
#                                                               #
#===============================================================#


##############################################################################################
# """
# INFERENCE TIME
# """
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# from YOLOv3_utils import *
# from YOLOv3_config import *

# # IMAGE_PATH = "./YOLOv3-for-studying/IMAGES/kite.jpg"
# IMAGE_PATH = "./YOLOv3-for-studying/IMAGES/lg_street.jpg"
# USE_LG_WEIGHTS = False
# if USE_LG_WEIGHTS:
#     yolo = Load_YOLOv3_Model("LG_WEIGHTS")
#     detect_image(yolo, IMAGE_PATH, show=True, save=False, CLASS_FILE=LG_CLASS_NAMES_PATH)
# else:
#     yolo = Load_YOLOv3_Model("COCO_WEIGHTS")
#     detect_image(yolo, IMAGE_PATH, show=True, save=False, CLASS_FILE=YOLO_COCO_CLASS_DIR)
##############################################################################################






# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# from YOLOv3_config import *
# from YOLOv3_utils import *
# text_by_line = "E:/dataset/TOTAL/test\images\cyclist_223_171_53_30_11000_2020-10-27-15-17-46-000000.jpg 12,120,69,181,0 80,7,128,27,0 57,5,87,25,0 1069,208,1124,274,2"
# # text_by_line = "E:/dataset/TOTAL/train\images\c1_2020-10-292020-10-29-12-16-28-000138.jpg 762,114,782,176,1 785,120,804,180,1 676,77,692,120,1 651,64,663,109,1 663,71,677,113,1 327,31,342,67,1 364,119,382,183,1 618,76,639,124,1 320,191,344,264,1 411,0,421,23,1 611,74,631,122,1 359,0,366,17,1 282,4,292,32,1 628,58,642,101,1 343,0,353,19,1 208,598,260,647,1 234,298,279,391,1 367,115,395,168,1 369,0,379,16,1 268,6,277,33,1 256,17,269,48,1 904,24,920,47,1 920,26,928,48,1 407,0,415,22,1"
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

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if len(gpus) > 0:
#     print(f'GPUs {gpus}')
#     try: tf.config.experimental.set_memory_growth(gpus[0], True)
#     except RuntimeError: pass
# yolo = Load_YOLOv3_Model("LG_WEIGHTS")
# pred_image = detect_image(yolo, image_path, show=False, save=False, CLASS_FILE=LG_CLASS_NAMES_PATH)

# image = cv2.imread(image_path)
# image = draw_bbox(image, bboxes, LG_CLASS_NAMES_PATH)
# cv2.imshow('truth', image)
# cv2.imshow("prediction", pred_image)

# if cv2.waitKey() == 'q':
#     pass



# # YOLOv3_weights = YOLO_V3_COCO_WEIGHTS
# # yolo = YOLOv3_Model(input_size=YOLO_INPUT_SIZE, CLASS_DIR=YOLO_COCO_CLASS_DIR)
# # if USE_LOADED_WEIGHT:
# #     print("Loading Darknet_weights from:", YOLOv3_weights)
# #     # yolo.load_weights(YOLOv3_weights)
# #     load_yolov3_weights(yolo, YOLOv3_weights)
# # pred2_image = detect_image(yolo, image_path, show=False, save=False, CLASS_FILE=YOLO_COCO_CLASS_DIR)

# # cv2.imshow('prediction from darknet', pred2_image)

# # del yolo

# # image = cv2.imread(image_path)
# # image = draw_bbox(image, bboxes, LG_CLASS_NAMES_PATH)
# # cv2.imshow('test', image)


# gpus = tf.config.experimental.list_physical_devices('GPU')
# if len(gpus) > 0:
#     print(f'GPUs {gpus}')
#     try: tf.config.experimental.set_memory_growth(gpus[0], True)
#     except RuntimeError: pass
# # yolo = Load_YOLOv3_Model()
# # _ = detect_image(yolo, image_path, show=False, save=False, CLASS_FILE=LG_CLASS_NAMES_PATH)
# # pred_image = detect_image(yolo, image_path, show=False, save=False, CLASS_FILE=LG_CLASS_NAMES_PATH, i=True)
# # cv2.imshow('prediction', pred_image)

# YOLOv3_weights = YOLO_V3_COCO_WEIGHTS

# if USE_LOADED_WEIGHT:
#     print("Loading Darknet_weights from:", YOLOv3_weights)
#     # yolo.load_weights(YOLOv3_weights)
#     yolo = YOLOv3_Model(input_size=YOLO_INPUT_SIZE, CLASS_DIR=YOLO_COCO_CLASS_DIR)
#     load_yolov3_weights(yolo, YOLOv3_weights)
# # _ = detect_image(yolo, image_path, show=False, save=False, CLASS_FILE=YOLO_COCO_CLASS_DIR)
# # pred2_image = detect_image(yolo, image_path, show=False, save=False, CLASS_FILE=YOLO_COCO_CLASS_DIR)

# IMAGE_PATH = "./YOLOv3-for-studying/IMAGES/lg_street.jpg"
# image_1 = cv2.imread(IMAGE_PATH)
# image_data = image_preprocess(np.copy(image_1), YOLO_INPUT_SIZE)    #scale to size 416
# image_data = image_data[np.newaxis, ...].astype(np.float32)  
# image_pred1 = yolo(image_data, training=False)
# image_pred2 = yolo.predict(image_data)
# image_pred3 = yolo.predict(image_data)
# load_yolov3_weights(yolo, YOLOv3_weights)
# image_pred4 = yolo.predict(image_data)
# image_pred5 = yolo.predict(image_data)

# print('abc')


a = 'alf/abc/'
print(a+'123/')



# import tensorflow as tf

# a = tf.Variable([[[1,2], [3,4], [5,6]]])
# print(tf.shape(a[..., 0] * a[..., 1]))


# b = tf.math.reduce_prod(a, axis=-1)
# print(b)
# print(tf.shape(b))





# if cv2.waitKey() == 'q':
#     pass
# cv2.waitKey(0)


# a = [1,4,3,2]
# a.sort()
# print('\n', a, '\n')

# from decimal import DecimalTuple
# from regex import D
# from YOLOv3_config import *
# from YOLOv3_utils import *
# import cv2

# with open(TRAIN_ANNOTATION_PATH, 'r') as f:
#     all_texts_by_line = f.read().splitlines()
#     annotations = [text_by_line.strip() for text_by_line in all_texts_by_line if len(text_by_line.strip().split()[1:]) != 0]





# test = text.split()
# # print(test)
# full_path = test[0]
# relative_path = 'E:/dataset/TOTAL/'
# temp_path = os.path.relpath(full_path, relative_path)
# prefix_path = '.\YOLOv3-for-studying\LG_DATASET'
# new_path = os.path.join(prefix_path, temp_path)
# # print(new_path)
# new_path = new_path.replace("\\" , "/")
# # print(new_path)

# final_annotations = []
# # annotation = annotations[1]
# # annotation = text
# text_by_line = text
# bboxes_annotations = []
# for text in text_by_line:
#     if not text.replace(',','').isnumeric():
#         temp_path   = os.path.relpath(text, RELATIVE_PATH)
#         temp_path   = os.path.join(PREFIX_PATH, temp_path)
#         image_path  = temp_path.replace('\\','/')
#     else:
#         bboxes_annotations.append(text)
# final_annotations.append([image_path, bboxes_annotations])



# # print(image_path)
# # print(final_annotations)

# def read_class_names(class_file_name):
#     # loads class name from a file
#     names = {}
#     with open(class_file_name, 'r') as data:
#         for ID, name in enumerate(data):
#             names[ID] = name.strip('\n')
#     return names

# # print(read_class_names(LG_CLASS_NAMES_PATH))

# # print(np.array(YOLO_ANCHORS))
# strides = np.array(YOLO_SCALE_OFFSET)
# strides = strides[:, np.newaxis, np.newaxis]
# # print(np.array(YOLO_ANCHORS)/ strides)

# annotation = final_annotations[0]
# image = cv2.imread(annotation[0])
# bboxes = np.array([list(map(int,box.split(','))) for box in annotation[1]])

# x = np.ones((len(bboxes),1)) - 0.1
# classes = bboxes[:, 4]
# classes = classes[:, np.newaxis]

# bboxes = np.concatenate([bboxes[:,0:4], x, classes], axis=-1)
# print(bboxes)


# image = draw_bbox(image, bboxes, LG_CLASS_NAMES_PATH)
# cv2.imshow('test', image)
# if cv2.waitKey() == 'q':
#     pass

# print(random.choice([416]))

# a = np.full(50, 1/50)
# delta = 0.01
# classes = np.zeros((50))
# classes[1] = 1.
# onehot = classes*(1-delta) + delta/len(classes)

# # print(onehot.sum())

# a = [1.2, 2.3 ,3.4]
# # print(np.floor(np.array(a)).astype(np.int32) + 0.5 - 0.5)

# a = np.zeros((3,4))
# b = np.ones((3,2))

# a[:,:2] = b[1,:2]
# # print(b[1,:2].shape)
# # print(a)

# # a = np.zeros((1,2,3,4))
# # a[1][0] = 1

# # a = [1,2,3,4]
# # print(a[1.2])

# # a = np.zeros((2,3,4))
# # for i in range(3):
# #     a[i][2,3,4] = 1

# a = np.zeros((1,2,3,4))
# b = np.ones((3,4))

# c = a,b
# # print(np.array(c[1]).shape)


# # print(tf.math.log(np.exp(2)))


# a = np.array([1,2,3])
# b = lambda x: x+x
# # print(b(a))


# # x_pred = np.array([[2, 10, 20], [9, 20, 4.]])
# # x_true = np.array([[0, 0, 1], [0, 1, 0.]])
# # test = tf.nn.sigmoid_cross_entropy_with_logits(logits= x_pred, labels= x_true)      # --> categorical cross entropy
# # print("softmax cross entropy: ", test, '\n')

# # x = 0.51
# # z = 1

# # # test = -z * tf.math.log(tf.sigmoid(x)) - (1-z) * tf.math.log(1- tf.sigmoid(x))
# # # test = -z * tf.math.log(x) - (1-z) * tf.math.log(1- x)
# # # print("sigmoid = ", test)

# # loss = tf.keras.losses.binary_crossentropy(x_true, x_pred)                #binary + from_logits=True: sigmoid -> cross entropy -> mean      
# #                                                                                 #categorical + from_logits=True: softmax -> cross entropy
# #                                                                                 # categorical + False: each/sum -> cross entropy
# # print("Test: ", loss, '\n')

# # test = tf.sigmoid(x_pred)
# # test = np.array([[1. -1e-7, 1. -1e-7, 1. -1e-7], [0, 0, 0]])
# # result = (tf.math.log(1-test[0,0] + 1e-7) + tf.math.log(1-test[0,1] + 1e-7) + tf.math.log(test[0,2] + 1e-7)) / 3
# # print(result)
# # print(test)
# # print(-tf.math.log(test[0][2]), '\n', -tf.math.log(test[1][1]))

# # test2 = -z * tf.math.log(tf.nn.softmax(x)) - (1-z) * tf.math.log(1- tf.nn.softmax(x))
# # print("Equation: ", test2)


# logits = lambda x: tf.math.log(x/(1.0 - x))

# a = tf.Variable([[1,2,3],[4,5,6.]])
# # print(tf.divide(a,10))
# b = tf.add(a, tf.divide(a,10))
# c = tf.subtract(b,a)
# # print(tf.map_fn(logits, c))



# # print(np.array([list(map(int, box.split(','))) for box in annotation[1]]))

# # a = [1,2,3,5,4]
# # print(a[2:])

# # image = cv2.imread(new_path)
# # cv2.imshow("test", image)
# # if cv2.waitKey() == 'q':
# #     pass

# # print(all_texts_by_line)

# def draw_bbox(image, bboxes, CLASS_DIR=YOLO_COCO_CLASS_DIR, show_label=True, show_confidence=True, Text_colors='', rectangle_colors='', tracking=False):
#     #Initial readings
#     CLASS_NAMES = read_class_names(CLASS_DIR)
#     num_classes = len(CLASS_NAMES)
#     image_h, image_w, _ = image.shape
#     #generate random color for bboxes and labels
#     rectangle_hsv_tuples     = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
#     label_hsv_tuples         = [(1.0 * x / num_classes, 1., 1.) for x in range(int(num_classes/2), num_classes)] 
#     label_hsv_tuples        += [(1.0 * x / num_classes, 1., 1.) for x in range(0, int(num_classes/2))]                                
#     rand_rectangle_colors    = list(map(lambda x: colorsys.hsv_to_rgb(*x), rectangle_hsv_tuples))
#     rand_rectangle_colors    = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), rand_rectangle_colors))
#     rand_text_colors         = list(map(lambda x: colorsys.hsv_to_rgb(*x), label_hsv_tuples))
#     rand_text_colors         = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), rand_text_colors))
#     #draw each bbox and label
#     for bbox in bboxes:
#         coordinates = np.array(bbox[:4], dtype=np.int32)
#         score = bbox[4]
#         class_id = int(bbox[5])
#         #select color
#         bbox_color = rectangle_colors if rectangle_colors != '' else rand_rectangle_colors[class_id]
#         label_color = Text_colors if Text_colors != ''  else rand_text_colors[class_id]
#         #calculate thickness and fontSize
#         bbox_thick = int(0.6 * (image_h + image_w) / 1000)
#         if bbox_thick < 1: bbox_thick = 1
#         fontScale = 0.75 * bbox_thick
#         #draw bbox to image
#         (x1, y1), (x2, y2) = (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3])
#         cv2.rectangle(image, (x1,y1), (x2, y2), bbox_color, bbox_thick * 2)
#         #draw label to image
#         if show_label:
#             score_str = " {:.2f}".format(score) if show_confidence else ""
#             try:
#                 label = "{}".format(CLASS_NAMES[class_id]) + score_str
#             except KeyError:
#                 print("You received KeyError")
#             #draw filled rectangle and add text to this
#             (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=fontScale, thickness=bbox_thick)
#             cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)
#             cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=fontScale, color=label_color, thickness=bbox_thick, lineType=cv2.LINE_AA)    
#     return image


