from YOLOv4_model import *
from YOLOv4_config import *
from YOLOv4_utils import *
import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt


text_by_line = "E:/dataset/TOTAL/test\images/frame_20210424_095010_00681_51.jpg 1242,221,1415,335,0 28,509,146,661,0 1505,129,1735,258,0 760,425,951,550,0 753,314,948,432,0 1064,339,1305,486,0 1688,218,1920,356,0 1888,90,1919,180,0 277,183,317,231,1 319,90,358,133,1 338,276,408,325,2"
# text_by_line = "E:/dataset/TOTAL/test\images\cyclist_223_171_53_30_11002_2020-11-062020-11-06-12-11-26-000382.jpg 543,212,662,263,0 810,382,905,500,0 546,179,622,222,0 91,452,136,536,1 245,255,265,309,1 1140,62,1151,116,1 94,321,121,380,1 949,118,962,144,1 295,272,312,327,1 385,219,401,268,1 311,260,331,306,1 283,275,299,329,1 396,260,418,311,1 450,204,463,244,1 1119,21,1126,46,1 892,114,901,138,1 527,170,538,206,1 7,314,29,368,1 1126,21,1133,46,1 962,101,971,128,1 403,223,419,267,1 418,226,434,271,1 1106,23,1112,48,1 1111,23,1117,48,1 952,100,962,126,1 280,251,296,277,1 293,252,309,277,1 415,209,428,244,1 425,210,440,254,1 459,206,472,246,1 479,181,492,217,1 492,181,502,220,1 511,169,524,208,1 671,394,703,483,2"
RELATIVE_PATH               = "E:/dataset/TOTAL/"
PREFIX_PATH                 = "C:/Users/Claw/VSCode_Gitrepo/Yolov4_Learning/YOLOv4-for-studying/dataset/LG_DATASET"

size = [224, 128]

text = text_by_line.split()
bboxes = []
for t in text:
    if not t.replace(',', '').isnumeric():
        temp_path   = os.path.relpath(t, RELATIVE_PATH)
        temp_path   = os.path.join(PREFIX_PATH, temp_path)
        image_path  = temp_path.replace('\\','/')
    else:
        t = list(map(int, t.split(',')))
        bboxes.append(t)
original_bboxes = np.array(bboxes)
original_image = cv2.imread(image_path)

image, bboxes = image_preprocess(np.copy(original_image), np.array(size), np.copy(original_bboxes))
imagex2, bboxesx2 = image_preprocess(np.copy(original_image), np.array(size)*2, np.copy(original_bboxes))
image_data = image[np.newaxis, ...].astype(np.float32)
imagex2_data = imagex2[np.newaxis,...].astype(np.float32)


# def weight_sharing_origin_to_backbone(dest, src):
#     for i in TEACHER_LAYERS_RANGE:
#         temp_t = i
#         if i >= 11:
#             temp_t = temp_t +1
#         if i >= 49:
#             temp_t = temp_t + 1
#         if i >= 98:
#             temp_t = temp_t + 1
#         if i >= 213:
#             temp_t = temp_t + 1
#         if i >= 328:
#             temp_t = temp_t + 1
#         if dest.layers[temp_t].get_weights() != []:
#             dest.layers[temp_t].set_weights(src.layers[i].get_weights())
#         # print("Finished sharing!")


student_weight = "YOLOv4-for-studying/checkpoints/Num-123_lg_dataset_transfer_224x128/epoch-49_valid-loss-14.35/yolov4_lg_transfer"
teacher_weight = "YOLOv4-for-studying/checkpoints/Num-110_lg_dataset_transfer_448x256/epoch-37_valid-loss-8.60_large-anchor/yolov4_lg_transfer"

#Create YOLO model
student = YOLOv4_Model(CLASSES_PATH=YOLO_CLASS_PATH, training=True)
student.load_weights(student_weight)

_,_,_,_,_,_,mid_st_P2, mid_st_P3, mid_st_P4, out_st_P2, out_st_P3, out_st_P4  = student(image_data, training=True)
student_P2, student_P3, student_P4 = [tf.math.reduce_mean(tf.math.top_k(x,k=20)[0], axis=-1, keepdims=False) for x in [mid_st_P2[0], mid_st_P3[0], mid_st_P4[0]]]
# student_P2, student_P3, student_P4 = [tf.math.reduce_mean(x, axis=-1, keepdims=False) for x in [mid_st_P2[0], mid_st_P3[0], mid_st_P4[0]]]       #channel-wise

teacher = create_YOLOv4_backbone(CLASSES_PATH=YOLO_CLASS_PATH)
teacher.load_weights(teacher_weight)

mid_te_P3, mid_te_P4, mid_te_P5, out_te_P3, out_te_P4, out_te_P5 = teacher(imagex2_data, training=False)
teacher_P3, teacher_P4, teacher_P5 = [tf.math.reduce_mean(tf.math.top_k(x,k=20)[0], axis=-1, keepdims=False) for x in [mid_te_P3[0], mid_te_P4[0], mid_te_P5[0]]]
# teacher_P3, teacher_P4, teacher_P5 = [tf.math.reduce_mean(x, axis=-1, keepdims=False) for x in [mid_te_P3[0], mid_te_P4[0], mid_te_P5[0]]]  #channel-wise

show_featuremap = True


if not show_featuremap:
    pred_bboxes = [pred_bboxes[2*i+1] for i in range(3)]
    #post process for prediction bboxes
    pred_bboxes = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bboxes]
    pred_bboxes = tf.concat(pred_bboxes, axis=0)                                #shape [total_bboxes, 5 + NUM_CLASS]
    pred_bboxes = postprocess_boxes(pred_bboxes, np.copy(original_image), TEST_INPUT_SIZE, 0.35)  #remove invalid and low score bboxes
    pred_bboxes = tf.convert_to_tensor(nms(pred_bboxes, method='nms', iou_threshold=0.5))

    imaget = draw_bbox(original_image, pred_bboxes, YOLO_CLASS_PATH, show_label=False)

    cv2.imshow("Test", imaget)
    if cv2.waitKey()=="q":
        pass

else:
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(student_P2, cmap='magma')
    ax1.set_title("Student P2")

    ax2.imshow(teacher_P3, cmap='magma')
    ax2.set_title("Teacher P3")

    ax3.imshow(student_P3, cmap='magma')
    ax3.set_title("Student P3")
    
    
    
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(student_fmap[2], cmap='magma')
    # ax1.set_title("Student")
    
    # ax2.imshow(teacher_fmap[4], cmap='magma')
    # ax2.set_title("Teacher")

    fig.show()
    print("A")
    if cv2.waitKey()=="q":
        pass
