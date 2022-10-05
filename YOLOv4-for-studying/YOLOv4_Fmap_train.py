#===============================================================#
#                                                               #
#   File name   : YOLOv4_train.py                               #
#   Author      : hxnghia99                                     #
#   Created date: May 24th, 2022                                #
#   GitHub      : https://github.com/hxnghia99/Yolov4_Learning  #
#   Description : YOLOv4 training                               #
#                                                               #
#===============================================================#


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import shutil

import numpy as np
import tensorflow as tf
from YOLOv4_dataset import Dataset
from YOLOv4_model   import *
from YOLOv4_utils   import *
from YOLOv4_config  import *


import logging
tf.get_logger().setLevel(logging.ERROR)


#Compute YOLOv4 loss for each scale using reference code
def compute_loss(i, gt_bboxes, fmap_student=None, fmap_teacher=None):
    batch_size = tf.shape(gt_bboxes)[0]
    #if use featuremap teacher to teach feature map student
    if fmap_teacher != None:
        #global loss
        # gb_loss = tf.norm(fmap_teacher - fmap_student, ord=2, axis=-1)
        gb_loss = tf.square(fmap_teacher - fmap_student)
        gb_loss = tf.reduce_mean(tf.reduce_sum(gb_loss, axis=[1,2,3])) #/ tf.cast((tf.shape(gb_loss)[1]*tf.shape(gb_loss)[2]*tf.shape(gb_loss)[3]), tf.float32))  #each pixel in hxwxc
        #positive object loss
        flag_pos_obj = np.zeros(fmap_teacher.shape)
        num_channels = fmap_teacher.shape[-1]
        list_num_pos_pixel = []
        num_fmap_w_pos_pixel = 0
        for k in range(batch_size):         #each image
            num_pos_pixel = 0
            for j in range(YOLO_MAX_BBOX_PER_SCALE):    #each gt bbox
                if np.multiply.reduce(gt_bboxes[k,j][2:4]) != 0:        #gt_bboxes: xywh
                    gt_bbox = np.concatenate([gt_bboxes[k,j][:2]-gt_bboxes[k,j][2:4]*0.5, gt_bboxes[k,j][:2]+gt_bboxes[k,j][2:4]*0.5], axis=-1).astype(np.int32)
                    xmin, ymin = np.array(gt_bbox[0:2] / YOLO_SCALE_OFFSET[i]).astype(np.int32)
                    xmax, ymax = np.ceil(gt_bbox[2:4] / YOLO_SCALE_OFFSET[i]).astype(np.int32)

                    num_pos_pixel += (ymax-ymin)*(xmax-xmin)*num_channels
                    temp = np.ones([ymax-ymin, xmax-xmin, num_channels])
                    flag_pos_obj[k][ymin:ymax, xmin:xmax, :] = temp
            if num_pos_pixel==0:
                num_pos_pixel=1
            else:
                num_fmap_w_pos_pixel+=1  
            list_num_pos_pixel.append(num_pos_pixel)
        if num_fmap_w_pos_pixel == 0:
            num_fmap_w_pos_pixel=1
        num_fmap_w_pos_pixel = tf.cast(num_fmap_w_pos_pixel, tf.float32)
        list_num_pos_pixel = tf.cast(np.array(list_num_pos_pixel), tf.float32)
        flag_pos_obj = np.array(flag_pos_obj, dtype=np.bool_)
        pos_obj_loss = (fmap_teacher - fmap_student) * tf.cast(flag_pos_obj, tf.float32)
        # pos_obj_loss = (fmap_teacher - fmap_student)[flag_pos_obj]
        # pos_obj_loss = tf.reduce_sum(tf.norm(pos_obj_loss, ord=1, axis=-1))
        pos_obj_loss = tf.reduce_sum(tf.reduce_sum(tf.square(pos_obj_loss), axis=[1, 2, 3])) / num_fmap_w_pos_pixel #/ list_num_pos_pixel) / num_fmap_w_pos_pixel
        # pos_obj_loss = tf.Variable(0.0)
        # pos_obj_loss = tf.divide(pos_obj_loss, tf.cast(batch_size, tf.float32))

    if fmap_teacher!=None:
        return LAMDA_FMAP_LOSS*gb_loss, LAMDA_FMAP_LOSS*pos_obj_loss


#Add neck layers to CSPDarknet53 and create YOLOv4 model
def create_YOLOv4_student(input_channel=3, teacher_ver=False, student_ver=False):
    input_layer = Input([None, None, input_channel])
    # Create CSPDarknet53 network and 3 backbone features at large, medium and small scale
    if MODEL_BRANCH_TYPE[1] == "P5m":
        route_2, route_3, route_4, conv = CSPDarknet53(input_layer, teacher_ver=teacher_ver, student_ver=student_ver) 
    """ PANet bottom up layers """
    if MODEL_BRANCH_TYPE[1] == "P5m":
        #upsampling 1
        if not USE_FTT_P4:
            route_5 = conv                                              #output: 13 x 13 x 512      
            conv = convolutional(conv, (1, 1, 512, 256), teacher_ver=teacher_ver, student_ver=student_ver)                #output: 13 x 13 x 256      #415 +3x2(above+below)->421
            conv = UpSampling2D()(conv)                                 #output: 26 x 26 x 256      #422                                 
            route_4 = convolutional(route_4, (1, 1, 512, 256), teacher_ver=teacher_ver, student_ver=student_ver)          #output: 26 x 26 x 256
            conv = tf.concat([route_4, conv], axis=-1)                  #output: 26 x 26 x 512      #423
        else:
            conv = FTT_module(conv, route_4, 512)   #415 +33->448
        fmap_P4 = conv
        #Compress information of feature maps
        conv = convolutional(conv, (1, 1, 512, 256), teacher_ver=teacher_ver, student_ver=student_ver)                                                #423 +3->426 
        conv = convolutional(conv, (3, 3, 256, 512), teacher_ver=teacher_ver, student_ver=student_ver)                                                #426 +3->429
        conv = convolutional(conv, (1, 1, 512, 256), teacher_ver=teacher_ver, student_ver=student_ver)                                                #429 +3->432
        conv = convolutional(conv, (3, 3, 256, 512), teacher_ver=teacher_ver, student_ver=student_ver)                    #output: 26 x 26 x 512      #432 +3->435
        # fmap_P4 = conv      
        conv = convolutional(conv, (1, 1, 512, 256), teacher_ver=teacher_ver, student_ver=student_ver)                    #output: 26 x 26 x 256      #432 +3->438
    
    if MODEL_BRANCH_TYPE[1] == "P5m":
        #upsampling 2
        if not USE_FTT_P3:
            route_4 = conv                                               #output: 26 x 26 x 256     
            conv = conv = convolutional(conv, (1, 1, 256, 128), teacher_ver=teacher_ver, student_ver=student_ver)          #output: 26 x 26 x 128     #438 +3x2(above+below)->444
            conv = UpSampling2D()(conv)                                  #output: 52 x 52 x 128     #445                 
            route_3 = convolutional(route_3, (1, 1, 256, 128), teacher_ver=teacher_ver, student_ver=student_ver)           #output: 52 x 52 x 128
            conv = tf.concat([route_3, conv], axis=-1)                   #output: 52 x 52 x 256     #446
        else:
            conv = FTT_module(conv, route_3, 256)
        fmap_P3 = conv  
        #Compress information of feature maps
        conv = convolutional(conv, (1, 1, 256, 128), teacher_ver=teacher_ver, student_ver=student_ver)                                                #446 +3->449
        conv = convolutional(conv, (3, 3, 128, 256), teacher_ver=teacher_ver, student_ver=student_ver)                                                #449 +3->452
        conv = convolutional(conv, (1, 1, 256, 128), teacher_ver=teacher_ver, student_ver=student_ver)                                                #452 +3->455
        conv = convolutional(conv, (3, 3, 128, 256), teacher_ver=teacher_ver, student_ver=student_ver)                     #output: 52 x 52 x 256     #455 +3->458
        # fmap_P3 = conv  
        conv = convolutional(conv, (1, 1, 256, 128), teacher_ver=teacher_ver, student_ver=student_ver)                     #output: 52 x 52 x 128     #458 +3->461


    """ Additional upsampling: to resolution P2 """
    if  MODEL_BRANCH_TYPE[1] == "P5m":
        #upsampling 3
        if not USE_FTT_P2:
            route_3 = conv                                                #output: 52 x 52 x 128
            conv = convolutional(conv, (1, 1, 128, 64))                   #output: 52 x 52 x 64
            conv = UpSampling2D()(conv)                                   #output: 104 x 104 x 64                                       
            route_2 = convolutional(route_2, (1, 1, 128, 64))             #output: 104 x 104 x 64
            conv = tf.concat([route_2, conv], axis=-1)                    #output: 104 x 104 x 128
        else:
            conv = FTT_module(conv, route_2, 128)                                                   #461 +33->494
        fmap_P2 = conv    

    output_tensors = [fmap_P2, fmap_P3, fmap_P4, conv]#, test1, test2]
    YOLOv4_student = tf.keras.Model(input_layer, output_tensors)
    return YOLOv4_student





def create_YOLOv4_teacher(input_channel=3, dilation=False, teacher_ver=False, student_ver=False):
    input_layer = Input([None, None, input_channel])
    _, route_3, route_4, conv = CSPDarknet53(input_layer, dilation=dilation, teacher_ver=teacher_ver, student_ver=student_ver)            #26x26x512
    """ PANet bottom up layers """
    if MODEL_BRANCH_TYPE[1] == "P5m":
        #upsampling 1
        fmap_bb_P5 = conv 
        if not USE_FTT_P4:
            conv = convolutional(conv, (1, 1, 512, 256), dilation=dilation, teacher_ver=teacher_ver, student_ver=student_ver)             #output: 26 x 26 x 256
            conv = UpSampling2D()(conv)                                                 #output: 52 x 52 x 256                                       
            route_4 = convolutional(route_4, (1, 1, 512, 256), dilation=dilation, teacher_ver=teacher_ver, student_ver=student_ver)       #output: 52 x 52 x 256
            conv = tf.concat([route_4, conv], axis=-1)                                  #output: 52 x 52 x 512
        else:
            conv = FTT_module(conv, route_4, 512, dilation=dilation)
        # test_P4 = conv
        #Compress information of feature maps
        conv = convolutional(conv, (1, 1, 512, 256), dilation=dilation, teacher_ver=teacher_ver, student_ver=student_ver)
        conv = convolutional(conv, (3, 3, 256, 512), dilation=dilation, teacher_ver=teacher_ver, student_ver=student_ver)
        conv = convolutional(conv, (1, 1, 512, 256), dilation=dilation, teacher_ver=teacher_ver, student_ver=student_ver)
        conv = convolutional(conv, (3, 3, 256, 512), dilation=dilation, teacher_ver=teacher_ver, student_ver=student_ver)                 #output: 52 x 52 x 512
        conv = convolutional(conv, (1, 1, 512, 256), dilation=dilation, teacher_ver=teacher_ver, student_ver=student_ver)                 #output: 52 x 52 x 256  
        fmap_bb_P4 = conv
    
    if MODEL_BRANCH_TYPE[1] == "P5m":
        #upsampling 2
        if not USE_FTT_P3:
            route_4 = conv                                                              #output: 52 x 52 x 256
            conv = conv = convolutional(conv, (1, 1, 256, 128), dilation=dilation, teacher_ver=teacher_ver, student_ver=student_ver)      #output: 52 x 52 x 128
            conv = UpSampling2D()(conv)                                                 #output: 104 x 104 x 128                                                 
            route_3 = convolutional(route_3, (1, 1, 256, 128), dilation=dilation, teacher_ver=teacher_ver, student_ver=student_ver)       #output: 104 x 104 x 128
            conv = tf.concat([route_3, conv], axis=-1)                                  #output: 104 x 104 x 256
        else:
            conv = FTT_module(conv, route_3, 256, dilation=dilation, teacher_ver=teacher_ver, student_ver=student_ver)
        test_P3 = conv
        #Compress information of feature maps
        conv = convolutional(conv, (1, 1, 256, 128), dilation=dilation, teacher_ver=teacher_ver, student_ver=student_ver)
        conv = convolutional(conv, (3, 3, 128, 256), dilation=dilation, teacher_ver=teacher_ver, student_ver=student_ver)
        conv = convolutional(conv, (1, 1, 256, 128), dilation=dilation, teacher_ver=teacher_ver, student_ver=student_ver)
        conv = convolutional(conv, (3, 3, 128, 256), dilation=dilation, teacher_ver=teacher_ver, student_ver=student_ver)                  #output: 104 x 104 x 256
        conv = convolutional(conv, (1, 1, 256, 128), dilation=dilation, teacher_ver=teacher_ver, student_ver=student_ver)                  #output: 104 x 104 x 128       #78 + 14
        fmap_bb_P3 = conv

    output_tensors = [fmap_bb_P3, fmap_bb_P4, fmap_bb_P5, conv]#, test_P3, test_P4]
    YOLOv4_teacher = tf.keras.Model(input_layer, output_tensors)
    return YOLOv4_teacher



def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f'GPUs {gpus}')
    if len(gpus) > 0:
        try: tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError: pass
    #pretrained weights for CSPDarknet53 backbone network
    CSPDarknet_weights = YOLO_V4_COCO_WEIGHTS
    #create training and testing dataset
    trainset = Dataset('train')
    testset = Dataset('test')
    #initial settings for training
    steps_per_epoch = len(trainset)     #num_batches
    validate_steps_per_epoch = len(testset)
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)  #start 1
    warmup_steps = TRAIN_WARMUP_EPOCHS * steps_per_epoch
    total_steps = TRAIN_EPOCHS * steps_per_epoch
    
    def weight_sharing_origin_to_backbone(dest, src):
        for i in TEACHER_LAYERS_RANGE:
            temp_t = i
            if USE_SUPERVISION:
                if i >= 11:
                    temp_t = temp_t +1
                if i >= 49:
                    temp_t = temp_t + 1
                if i >= 98:
                    temp_t = temp_t + 1
                if i >= 213:
                    temp_t = temp_t + 1
                if i >= 328:
                    temp_t = temp_t + 1
            if dest.layers[temp_t].get_weights() != []:
                dest.layers[temp_t].set_weights(src.layers[i].get_weights())


    # yolo_student_layers_range = np.arange(len(yolo_student.layers))  #FTT_P3: 472, FTT_P2: 495
    # ftt_layers_range = np.arange(462, 495)                           #439,         462  
    # yolo_teacher_layers_range = np.setdiff1d(yolo_student_layers_range, ftt_layers_range)   

    #Create Darkent53 model and load pretrained weights
    if not TRAIN_FROM_CHECKPOINT and TRAIN_TRANSFER:
        Darknet = create_YOLOv4_student(student_ver=DISTILLATION_FLAG)
        load_yolov4_weights(Darknet, CSPDarknet_weights) # use darknet weights
        #Create YOLO model
        yolo_student = create_YOLOv4_student(student_ver=DISTILLATION_FLAG) 
        for i, l in enumerate(Darknet.layers):
            layer_weights = l.get_weights()
            if layer_weights != []:
                try:
                    yolo_student.layers[i].set_weights(layer_weights)
                except:
                    print("skipping", yolo_student.layers[i].name)

    elif TRAIN_FROM_CHECKPOINT:
        weight_file = "YOLOv4-for-studying/checkpoints/lg_dataset_transfer_224x128_P5_nFTT_P2/yolov4_lg_transfer"
        yolo_original = YOLOv4_Model(CLASSES_PATH=YOLO_CLASS_PATH, student_ver=DISTILLATION_FLAG)
        yolo_original.load_weights(weight_file)

        #Create YOLO model
        yolo_student = create_YOLOv4_student(student_ver=DISTILLATION_FLAG)

        for i in STUDENT_LAYERS_RANGE:                                 #--> Check layer order
            if yolo_student.layers[i].get_weights() != []:
                yolo_student.layers[i].set_weights(yolo_original.layers[i].get_weights())

    for i in TEACHER_LAYERS_RANGE:                         #--> Check layer order
        yolo_student.layers[i].trainable = False

    if USE_SUPERVISION:
        #yolov4 backbone network
        yolo_teacher = create_YOLOv4_teacher(dilation=BACKBONE_DILATION, teacher_ver=DISTILLATION_FLAG)
        weight_sharing_origin_to_backbone(yolo_teacher, yolo_student)
    
    #Create Adam optimizers
    optimizer = tf.keras.optimizers.Adam()#beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    
    #Create log folder and summary
    if os.path.exists(TRAIN_LOGDIR): shutil.rmtree(TRAIN_LOGDIR)
    training_writer = tf.summary.create_file_writer(TRAIN_LOGDIR+'training/')
    validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR+'validation/')

    

    if DISTILLATION_FLAG:
        #Create training function for each batch
        def train_step(image_data, target):
            if USE_SUPERVISION:
                imagex2_data = image_data[1]
                image_data = image_data[0]

            with tf.GradientTape(persistent=False) as tape:
                fmap_s_P3, fmap_s_P4, fmap_s_P5, _  = yolo_student(image_data, training=True)       #conv+pred: small -> medium -> large : shape [scale, batch size, output size, output size, ...]
                fmap_students = [fmap_s_P3, fmap_s_P4, fmap_s_P5]
                with tape.stop_recording():
                    if USE_SUPERVISION:
                        weight_sharing_origin_to_backbone(yolo_teacher, yolo_student)
                        fmap_t_P3, fmap_t_P4, fmap_t_P5, _ = yolo_teacher(imagex2_data, training=TEACHER_TRAINING_MODE)
                        fmap_teachers = [fmap_t_P3, fmap_t_P4, fmap_t_P5] 

                gb_loss=pos_pixel_loss=0
                num_scales = len(YOLO_SCALE_OFFSET)
                #calculate loss at each scale  
                for i in range(num_scales): 
                    if USE_SUPERVISION and ((i==0 and USE_FTT_P2) or (i==1 and USE_FTT_P3) or (i==2 and USE_FTT_P4)):
                    # if USE_SUPERVISION and ((i==2 and USE_FTT_P2) or (i==2 and USE_FTT_P3) or (i==2 and USE_FTT_P4)):
                        fmap_student = fmap_students[i]
                        fmap_teacher = fmap_teachers[i]
                        loss_items = compute_loss(i, target[i][1], fmap_teacher=fmap_teacher, fmap_student=fmap_student)
        
                        gb_loss += loss_items[0]
                        pos_pixel_loss += loss_items[1]
                #calculate total of loss
                if USE_SUPERVISION:
                    total_loss = gb_loss + pos_pixel_loss

                gradients = tape.gradient(total_loss, yolo_student.trainable_variables)
                optimizer.apply_gradients(zip(gradients, yolo_student.trainable_variables))
                # update learning rate
                if global_steps < warmup_steps:
                    lr = global_steps / warmup_steps * TRAIN_LR_INIT
                else:
                    lr = TRAIN_LR_END + 0.5 * (TRAIN_LR_INIT - TRAIN_LR_END)*(
                        (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))    
                optimizer.lr.assign(lr.numpy())
                #increase global steps 
                global_steps.assign_add(1)

                # writing summary data
                with training_writer.as_default():
                    tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                    tf.summary.scalar("training_loss/total_loss", total_loss, step=global_steps)
                    if USE_SUPERVISION:
                        tf.summary.scalar("training_loss/gb_loss", gb_loss, step=global_steps)
                        tf.summary.scalar("training_loss/pos_pixel_loss", pos_pixel_loss, step=global_steps)
                training_writer.flush()   
                # total_loss += fmap_loss
            # del tape  
            
            if USE_SUPERVISION:
                return global_steps.numpy(), optimizer.lr.numpy(), total_loss.numpy(), gb_loss.numpy(), pos_pixel_loss.numpy()
    else:
        #Create training function for each batch
        def train_step(image_data, target):
            if USE_SUPERVISION:
                # weight_sharing_origin_to_backbone(yolo_teacher, yolo_student)
                fmap_t_P3, fmap_t_P4, fmap_t_P5, _ = yolo_teacher(image_data[1], training=TEACHER_TRAINING_MODE)
                fmap_teachers = [fmap_t_P3, fmap_t_P4, fmap_t_P5] 
                image_data = image_data[0]

            with tf.GradientTape(persistent=False) as tape:
                fmap_s_P3, fmap_s_P4, fmap_s_P5, _  = yolo_student(image_data, training=True)       #conv+pred: small -> medium -> large : shape [scale, batch size, output size, output size, ...]
                fmap_students = [fmap_s_P3, fmap_s_P4, fmap_s_P5]

                gb_loss=pos_pixel_loss=0
                num_scales = len(YOLO_SCALE_OFFSET)
                #calculate loss at each scale  
                for i in range(num_scales): 
                    if USE_SUPERVISION and ((i==0 and USE_FTT_P2) or (i==1 and USE_FTT_P3) or (i==2 and USE_FTT_P4)):
                    # if USE_SUPERVISION and ((i==2 and USE_FTT_P2) or (i==2 and USE_FTT_P3) or (i==2 and USE_FTT_P4)):
                        fmap_student = fmap_students[i]
                        fmap_teacher = fmap_teachers[i]
                        loss_items = compute_loss(i, target[i][1], fmap_teacher=fmap_teacher, fmap_student=fmap_student)
        
                        gb_loss += loss_items[0]
                        pos_pixel_loss += loss_items[1]
                #calculate total of loss
                if USE_SUPERVISION:
                    total_loss = gb_loss + pos_pixel_loss

                gradients = tape.gradient(total_loss, yolo_student.trainable_variables)
                optimizer.apply_gradients(zip(gradients, yolo_student.trainable_variables))
                # update learning rate
                if global_steps < warmup_steps:
                    lr = global_steps / warmup_steps * TRAIN_LR_INIT
                else:
                    lr = TRAIN_LR_END + 0.5 * (TRAIN_LR_INIT - TRAIN_LR_END)*(
                        (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))    
                optimizer.lr.assign(lr.numpy())
                #increase global steps 
                global_steps.assign_add(1)

                # writing summary data
                with training_writer.as_default():
                    tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                    tf.summary.scalar("training_loss/total_loss", total_loss, step=global_steps)
                    if USE_SUPERVISION:
                        tf.summary.scalar("training_loss/gb_loss", gb_loss, step=global_steps)
                        tf.summary.scalar("training_loss/pos_pixel_loss", pos_pixel_loss, step=global_steps)
                training_writer.flush()   
            
            if USE_SUPERVISION:
                return global_steps.numpy(), optimizer.lr.numpy(), total_loss.numpy(), gb_loss.numpy(), pos_pixel_loss.numpy()




    #Create validation function after each epoch
    def validate_step(image_data, target):
        # if USE_SUPERVISION and not FLAG_USE_BACKBONE_EVALUATION:   
        if USE_SUPERVISION:
            # weight_sharing_origin_to_backbone(yolo_teacher, yolo_student)
            fmap_t_P3, fmap_t_P4, fmap_t_P5, _ = yolo_teacher(image_data[1], training=False)
            image_data = image_data[0]
            fmap_teachers = [fmap_t_P3, fmap_t_P4, fmap_t_P5] 

        fmap_s_P3, fmap_s_P4, fmap_s_P5, _  = yolo_student(image_data, training=False)       #conv+pred: small -> medium -> large : shape [scale, batch size, output size, output size, ...]
        fmap_students = [fmap_s_P3, fmap_s_P4, fmap_s_P5]

        gb_loss=pos_pixel_loss=0
        num_scales = len(YOLO_SCALE_OFFSET)
            #calculate loss at each scale  
        for i in range(num_scales): 
            if USE_SUPERVISION and ((i==0 and USE_FTT_P2) or (i==1 and USE_FTT_P3) or (i==2 and USE_FTT_P4)):
            # if USE_SUPERVISION and ((i==2 and USE_FTT_P2) or (i==2 and USE_FTT_P3) or (i==2 and USE_FTT_P4)):
                fmap_student = fmap_students[i]
                fmap_teacher = fmap_teachers[i]
                loss_items = compute_loss(i, target[i][1], fmap_teacher=fmap_teacher, fmap_student=fmap_student)

                gb_loss += loss_items[0]
                pos_pixel_loss += loss_items[1]
        #calculate total of loss
        if USE_SUPERVISION:
            total_loss = gb_loss + pos_pixel_loss
            return total_loss.numpy(), gb_loss.numpy(), pos_pixel_loss.numpy()


    best_val_loss = 100000 # should be large at start
    for epoch in range(TRAIN_EPOCHS):
        #Get a batch of training data to train
        total_train, gb_train, pos_pixel_train = 0, 0, 0
        for image_data, target in trainset:
            results = train_step(image_data, target)            #result = [global steps, learning rate, coor_loss, conf_loss, prob_loss, total_loss]
            current_step = results[0] % steps_per_epoch
            if USE_SUPERVISION:
                sys.stdout.write("\rEpoch ={:2.0f} step= {:5.0f}/{} : lr={:.10f} - gb_loss={:10.4f} - pos_pixel_loss = {:10.4f} - total_loss={:10.4f}"
                    .format(epoch+1, current_step, steps_per_epoch, results[1], results[3], results[4], results[2]))
                total_train += results[2]    
                gb_train += results[3]
                pos_pixel_train += results[4]
            

        # writing training summary data
        with training_writer.as_default():
            tf.summary.scalar("loss/total_val", total_train/steps_per_epoch, step=epoch)
            if USE_SUPERVISION:
                tf.summary.scalar("loss/gb_val", gb_train/steps_per_epoch, step=epoch)
                tf.summary.scalar("loss/pos_pixel_val", pos_pixel_train/steps_per_epoch, step=epoch)
        training_writer.flush()

        # print validate summary data
        print("\n\nSUMMARY of EPOCH = {:2.0f}".format(epoch+1))
        if USE_SUPERVISION:
            print("Training   : total_train_gb_loss:{:10.2f} - total_train_pos_pixel_loss:{:10.2f} - total_train_loss:{:10.2f}".
                format(gb_train/steps_per_epoch, pos_pixel_train/steps_per_epoch, total_train/steps_per_epoch))
        
        #If we do not have testing dataset, we save weights for every epoch
        if len(testset) == 0:
            print("configure TEST options to validate model")
            yolo_student.save_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME))
            continue
        #Validating the model with testing dataset
        num_testset = len(testset)
        total_val, gb_val, pos_pixel_val = 0, 0, 0
        current_step = 0
        for image_data, target in testset:
            results = validate_step(image_data, target)
            # if USE_SUPERVISION:
            #     FLAG_USE_BACKBONE_EVALUATION = False
            sys.stdout.write("\rProcessing: {:5.0f}/{}".format(current_step, validate_steps_per_epoch))
            current_step += 1
            total_val += results[0]
            if USE_SUPERVISION:
                gb_val += results[1]
                pos_pixel_val += results[2]
        # writing validate summary data
        with validate_writer.as_default():
            tf.summary.scalar("loss/total_val", total_val/num_testset, step=epoch)
            if USE_SUPERVISION:
                tf.summary.scalar("loss/gb_val", gb_val/num_testset, step=epoch)
                tf.summary.scalar("loss/pos_pixel_val", pos_pixel_val/num_testset, step=epoch)
        validate_writer.flush()
        if USE_SUPERVISION:
            # print validate summary data 
            print("\rValidation : total_valid_gb_loss:{:10.2f} - total_valid_pos_pixel_loss:{:10.2f} - total_valid_loss:{:10.2f}\n".
                format(gb_val/num_testset, pos_pixel_val/num_testset, total_val/num_testset))
        

        if USE_SUPERVISION:
            detection_loss = total_val

        if TRAIN_SAVE_CHECKPOINT and not TRAIN_SAVE_BEST_ONLY:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME+"_val_loss_{:7.2f}".format(detection_loss/num_testset))
            yolo_student.save_weights(save_directory)
        if TRAIN_SAVE_BEST_ONLY and best_val_loss>detection_loss/num_testset:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME)
            yolo_student.save_weights(save_directory)
            best_val_loss = detection_loss/num_testset
    # if USE_SUPERVISION:
        # FLAG_USE_BACKBONE_EVALUATION = False

if __name__ == '__main__':
    main()
    sys.modules[__name__].__dict__.clear()
