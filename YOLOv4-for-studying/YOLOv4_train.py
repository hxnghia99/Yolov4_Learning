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
from YOLOv4_model   import YOLOv4_Model, create_YOLOv4_backbone
from YOLOv4_loss    import compute_loss
from YOLOv4_utils   import load_yolov4_weights
from YOLOv4_config  import *

import logging
tf.get_logger().setLevel(logging.ERROR)



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
    
    #Create Darkent53 model and load pretrained weights
    if not TRAIN_FROM_CHECKPOINT and TRAIN_TRANSFER:
        Darknet = YOLOv4_Model(CLASSES_PATH=YOLO_COCO_CLASS_PATH, student_ver=DISTILLATION_FLAG)
        load_yolov4_weights(Darknet, CSPDarknet_weights) # use darknet weights
    #Create YOLO model
    yolo = YOLOv4_Model(training=True, CLASSES_PATH=YOLO_CLASS_PATH, student_ver=DISTILLATION_FLAG) 
    


    if not TRAIN_FROM_CHECKPOINT and TRAIN_TRANSFER:
        for i, l in enumerate(Darknet.layers):
            layer_weights = l.get_weights()
            if layer_weights != []:
                try:
                    yolo.layers[i].set_weights(layer_weights)
                except:
                    print("skipping", yolo.layers[i].name)
    elif TRAIN_FROM_CHECKPOINT:
        yolo.load_weights(PREDICTION_WEIGHT_FILE)
        print("Load weight file from checkpoint ... ")

    if USE_SUPERVISION:
        #yolov4 backbone network
        yolo_backbone = create_YOLOv4_backbone(dilation=BACKBONE_DILATION, teacher_ver=DISTILLATION_FLAG)
        FLAG_USE_BACKBONE_EVALUATION = False

    #Create Adam optimizers
    optimizer = tf.keras.optimizers.Adam()#beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    
    #Create log folder and summary
    if os.path.exists(TRAIN_LOGDIR): shutil.rmtree(TRAIN_LOGDIR)
    training_writer = tf.summary.create_file_writer(TRAIN_LOGDIR+'training/')
    validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR+'validation/')
    
    
    def weight_sharing_origin_to_backbone(dest, src):
        for i in TEACHER_LAYERS_RANGE:                                #462 layers + 5 MaxPool2D (due to dilation conv)
            temp_t = i
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
        # print("Finished sharing!")

    if not DISTILLATION_FLAG:
        #Create training function for each batch
        def train_step(image_data, target):
            if USE_SUPERVISION:
                weight_sharing_origin_to_backbone(yolo_backbone, yolo)
                fmap_bb_P3, fmap_bb_P4, fmap_bb_P5, _ = yolo_backbone(image_data[1], training=TEACHER_TRAINING_MODE)
                image_data = image_data[0]
                fmap_backbone = [fmap_bb_P3, fmap_bb_P4, fmap_bb_P5] 

            with tf.GradientTape(persistent=False) as tape:
                pred_result = yolo(image_data, training=True)       #conv+pred: small -> medium -> large : shape [scale, batch size, output size, output size, ...]
                giou_loss=conf_loss=prob_loss=gb_loss=pos_pixel_loss=0
                num_scales = len(YOLO_SCALE_OFFSET)
                #calculate loss at each scale  
                for i in range(num_scales): 
                    if USE_SUPERVISION and ((i==0 and USE_FTT_P2) or (i==1 and USE_FTT_P3) or (i==2 and USE_FTT_P4)):
                    # if USE_SUPERVISION and ((i==2 and USE_FTT_P2) or (i==2 and USE_FTT_P3) or (i==2 and USE_FTT_P4)):
                        conv, pred, fmap_student = pred_result[i*2], pred_result[i*2+1], pred_result[6+i]
                        fmap_teacher = fmap_backbone[i]
                        loss_items = compute_loss(pred, conv, *target[i], i, CLASSES_PATH=YOLO_CLASS_PATH, fmap_teacher=fmap_teacher, fmap_student=fmap_student)
                    else:
                        conv, pred = pred_result[i*2], pred_result[i*2+1]
                        loss_items = compute_loss(pred, conv, *target[i], i, CLASSES_PATH=YOLO_CLASS_PATH)
                    giou_loss += loss_items[0]
                    conf_loss += loss_items[1]
                    prob_loss += loss_items[2]
                    if USE_SUPERVISION and ((i==0 and USE_FTT_P2) or (i==1 and USE_FTT_P3) or (i==2 and USE_FTT_P4)):
                    # if USE_SUPERVISION and ((i==2 and USE_FTT_P2) or (i==2 and USE_FTT_P3) or (i==2 and USE_FTT_P4)):
                        gb_loss += loss_items[3]
                        pos_pixel_loss += loss_items[4]
                #calculate total of loss
                if USE_SUPERVISION:
                    total_loss = giou_loss + conf_loss + prob_loss + gb_loss + pos_pixel_loss
                else:   
                    total_loss = giou_loss + conf_loss + prob_loss 
                
                gradients = tape.gradient(total_loss, yolo.trainable_variables)
                optimizer.apply_gradients(zip(gradients, yolo.trainable_variables))

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
                    tf.summary.scalar("training_loss/giou_loss", giou_loss, step=global_steps)
                    tf.summary.scalar("training_loss/conf_loss", conf_loss, step=global_steps)
                    tf.summary.scalar("training_loss/prob_loss", prob_loss, step=global_steps)
                    if USE_SUPERVISION:
                        tf.summary.scalar("training_loss/gb_loss", gb_loss, step=global_steps)
                        tf.summary.scalar("training_loss/pos_pixel_loss", pos_pixel_loss, step=global_steps)
                training_writer.flush()   

            if USE_SUPERVISION:
                return global_steps.numpy(), optimizer.lr.numpy(), giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy(), gb_loss.numpy(), pos_pixel_loss.numpy()
            else:
                return global_steps.numpy(), optimizer.lr.numpy(), giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()
    else:
        #Create training function for each batch
        def train_step(image_data, target, epoch):
            with tf.GradientTape(persistent=False) as tape:
                pred_result = yolo(image_data[0], training=True)       #conv+pred: small -> medium -> large : shape [scale, batch size, output size, output size, ...]
                
                with tape.stop_recording():
                    if USE_SUPERVISION:
                        weight_sharing_origin_to_backbone(yolo_backbone, yolo)
                        fmap_bb_P3, fmap_bb_P4, fmap_bb_P5, _ = yolo_backbone(image_data[1], training=TEACHER_TRAINING_MODE)
                        fmap_backbone = [fmap_bb_P3, fmap_bb_P4, fmap_bb_P5]
            
                giou_loss=conf_loss=prob_loss=gb_loss=pos_pixel_loss=0
                num_scales = len(YOLO_SCALE_OFFSET)
                #calculate loss at each scale  
                for i in range(num_scales): 
                    if USE_SUPERVISION and ((i==0 and USE_FTT_P2) or (i==1 and USE_FTT_P3) or (i==2 and USE_FTT_P4)):
                    # if USE_SUPERVISION and ((i==2 and USE_FTT_P2) or (i==2 and USE_FTT_P3) or (i==2 and USE_FTT_P4)):
                        conv, pred, fmap_student = pred_result[i*2], pred_result[i*2+1], pred_result[6+i]
                        fmap_teacher = fmap_backbone[i]
                        loss_items = compute_loss(pred, conv, *target[i], i, CLASSES_PATH=YOLO_CLASS_PATH, fmap_teacher=fmap_teacher, fmap_student=fmap_student)
                    else:
                        conv, pred = pred_result[i*2], pred_result[i*2+1]
                        loss_items = compute_loss(pred, conv, *target[i], i, CLASSES_PATH=YOLO_CLASS_PATH)
                    giou_loss += loss_items[0]
                    conf_loss += loss_items[1]
                    prob_loss += loss_items[2]
                    if USE_SUPERVISION and ((i==0 and USE_FTT_P2) or (i==1 and USE_FTT_P3) or (i==2 and USE_FTT_P4)):
                    # if USE_SUPERVISION and ((i==2 and USE_FTT_P2) or (i==2 and USE_FTT_P3) or (i==2 and USE_FTT_P4)):
                        gb_loss += loss_items[3]
                        pos_pixel_loss += loss_items[4]
                #calculate total of loss
                if USE_SUPERVISION:# and epoch <= 25:
                    # total_loss =(giou_loss + conf_loss + prob_loss) * (epoch + 1)/ TRAIN_EPOCHS + (gb_loss + pos_pixel_loss) * (TRAIN_EPOCHS-epoch-1) / TRAIN_EPOCHS
                    total_loss = giou_loss + conf_loss + prob_loss + gb_loss + pos_pixel_loss
                else:   
                    total_loss = giou_loss + conf_loss + prob_loss 
                
                gradients = tape.gradient(total_loss, yolo.trainable_variables)
                optimizer.apply_gradients(zip(gradients, yolo.trainable_variables))
                
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
                    tf.summary.scalar("training_loss/giou_loss", giou_loss, step=global_steps)
                    tf.summary.scalar("training_loss/conf_loss", conf_loss, step=global_steps)
                    tf.summary.scalar("training_loss/prob_loss", prob_loss, step=global_steps)
                    if USE_SUPERVISION:
                        tf.summary.scalar("training_loss/gb_loss", gb_loss, step=global_steps)
                        tf.summary.scalar("training_loss/pos_pixel_loss", pos_pixel_loss, step=global_steps)
                training_writer.flush()   

            if USE_SUPERVISION:
                return global_steps.numpy(), optimizer.lr.numpy(), giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy(), gb_loss.numpy(), pos_pixel_loss.numpy()
            else:
                return global_steps.numpy(), optimizer.lr.numpy(), giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()




    #Create validation function after each epoch
    def validate_step(image_data, target):
        if USE_SUPERVISION and not FLAG_USE_BACKBONE_EVALUATION:
            weight_sharing_origin_to_backbone(yolo_backbone, yolo)
        if USE_SUPERVISION:
            fmap_bb_P3, fmap_bb_P4, fmap_bb_P5, _ = yolo_backbone(image_data[1], training=False)
            image_data = image_data[0]
            fmap_backbone = [fmap_bb_P3, fmap_bb_P4, fmap_bb_P5] 
   
        pred_result = yolo(image_data, training=False)
        giou_loss=conf_loss=prob_loss=gb_loss=pos_pixel_loss=0
        grid = len(YOLO_SCALE_OFFSET)
        #calculate loss at each each
        for i in range(grid):
            if USE_SUPERVISION and ((i==0 and USE_FTT_P2) or (i==1 and USE_FTT_P3) or (i==2 and USE_FTT_P4)):
            # if USE_SUPERVISION and ((i==2 and USE_FTT_P2) or (i==2 and USE_FTT_P3) or (i==2 and USE_FTT_P4)):
                conv, pred, fmap_student = pred_result[i*2], pred_result[i*2+1], pred_result[6+i]
                fmap_teacher = fmap_backbone[i]
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES_PATH=YOLO_CLASS_PATH, fmap_teacher=fmap_teacher, fmap_student=fmap_student)
            else:
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES_PATH=YOLO_CLASS_PATH)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]
            if USE_SUPERVISION and ((i==0 and USE_FTT_P2) or (i==1 and USE_FTT_P3) or (i==2 and USE_FTT_P4)):
            # if USE_SUPERVISION and ((i==2 and USE_FTT_P2) or (i==2 and USE_FTT_P3) or (i==2 and USE_FTT_P4)):
                gb_loss += loss_items[3]
                pos_pixel_loss += loss_items[4]

        #calculate total of loss
        if USE_SUPERVISION:
            total_loss = giou_loss + conf_loss + prob_loss + gb_loss + pos_pixel_loss
            return giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy(), gb_loss.numpy(), pos_pixel_loss.numpy()
        else:   
            total_loss = giou_loss + conf_loss + prob_loss 
            return giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    best_val_loss = float('inf') # should be large at start
    for epoch in range(TRAIN_EPOCHS):
        #Get a batch of training data to train
        giou_train, conf_train, prob_train, total_train, gb_train, pos_pixel_train = 0, 0, 0, 0, 0, 0
        for image_data, target in trainset:
            results = train_step(image_data, target, epoch)            #result = [global steps, learning rate, coor_loss, conf_loss, prob_loss, total_loss]
            current_step = results[0] % steps_per_epoch
            if USE_SUPERVISION:
                sys.stdout.write("\rEpoch ={:2.0f} step= {:5.0f}/{} : lr={:.10f} - giou_loss={:8.4f} - conf_loss={:10.4f} - prob_loss={:8.4f} - total_loss={:10.4f} - fmap_loss={:8.4f}"
                    .format(epoch+1, current_step, steps_per_epoch, results[1], results[2], results[3], results[4], results[5], results[6]+results[7]))
                giou_train += results[2]
                conf_train += results[3]
                prob_train += results[4]
                total_train += results[5]    
                gb_train += results[6]
                pos_pixel_train += results[7]
            else:
                sys.stdout.write("\rEpoch ={:2.0f} step= {:5.0f}/{} : lr={:.10f} - giou_loss={:8.4f} - conf_loss={:10.4f} - prob_loss={:8.4f} - total_loss={:10.4f}"
                    .format(epoch+1, current_step, steps_per_epoch, results[1], results[2], results[3], results[4], results[5]))
                giou_train += results[2]
                conf_train += results[3]
                prob_train += results[4]
                total_train += results[5] 

        # writing training summary data
        with training_writer.as_default():
            tf.summary.scalar("loss/total_val", total_train/steps_per_epoch, step=epoch)
            tf.summary.scalar("loss/giou_val", giou_train/steps_per_epoch, step=epoch)
            tf.summary.scalar("loss/conf_val", conf_train/steps_per_epoch, step=epoch)
            tf.summary.scalar("loss/prob_val", prob_train/steps_per_epoch, step=epoch)
            if USE_SUPERVISION:
                tf.summary.scalar("loss/gb_val", gb_train/steps_per_epoch, step=epoch)
                tf.summary.scalar("loss/pos_pixel_val", pos_pixel_train/steps_per_epoch, step=epoch)
        training_writer.flush()

        # print validate summary data
        print("\n\nSUMMARY of EPOCH = {:2.0f}".format(epoch+1))
        if USE_SUPERVISION:
            print("Training   : giou_train_loss:{:7.2f} - conf_train_loss:{:7.2f} - prob_train_loss:{:7.2f} - total_train_loss:{:7.2f} - total_fmap_loss:{:6.2f}".
                format(giou_train/steps_per_epoch, conf_train/steps_per_epoch, prob_train/steps_per_epoch, total_train/steps_per_epoch, (gb_train+pos_pixel_train)/steps_per_epoch))
        else:
            print("Training   : giou_train_loss:{:7.2f} - conf_train_loss:{:7.2f} - prob_train_loss:{:7.2f} - total_train_loss:{:7.2f}".
                format(giou_train/steps_per_epoch, conf_train/steps_per_epoch, prob_train/steps_per_epoch, total_train/steps_per_epoch))
        
        #If we do not have testing dataset, we save weights for every epoch
        if len(testset) == 0:
            print("configure TEST options to validate model")
            yolo.save_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME))
            continue
        #Validating the model with testing dataset
        num_testset = len(testset)
        giou_val, conf_val, prob_val, total_val, gb_val, pos_pixel_val, detection_loss = 0, 0, 0, 0, 0, 0, 0
        current_step = 0
        for image_data, target in testset:
            results = validate_step(image_data, target)
            if USE_SUPERVISION:
                FLAG_USE_BACKBONE_EVALUATION = True
            sys.stdout.write("\rProcessing: {:5.0f}/{}".format(current_step, validate_steps_per_epoch))
            current_step += 1
            giou_val += results[0]
            conf_val += results[1]
            prob_val += results[2]
            total_val += results[3]
            if USE_SUPERVISION:
                gb_val += results[4]
                pos_pixel_val += results[5]
                detection_loss += results[0] + results[1] + results[2]
        # writing validate summary data
        with validate_writer.as_default():
            tf.summary.scalar("loss/total_val", total_val/num_testset, step=epoch)
            tf.summary.scalar("loss/giou_val", giou_val/num_testset, step=epoch)
            tf.summary.scalar("loss/conf_val", conf_val/num_testset, step=epoch)
            tf.summary.scalar("loss/prob_val", prob_val/num_testset, step=epoch)
            if USE_SUPERVISION:
                tf.summary.scalar("loss/gb_val", gb_val/num_testset, step=epoch)
                tf.summary.scalar("loss/pos_pixel_val", pos_pixel_val/num_testset, step=epoch)
                tf.summary.scalar("loss/detection_loss", detection_loss/num_testset, step=epoch)
        validate_writer.flush()
        if USE_SUPERVISION:
            # print validate summary data 
            print("\rValidation : giou_valid_loss:{:7.2f} - conf_valid_loss:{:7.2f} - prob_valid_loss:{:7.2f} - total_valid_loss:{:7.2f} - total_fmap_loss:{:6.2f}".
                format(giou_val/num_testset, conf_val/num_testset, prob_val/num_testset, total_val/num_testset, (gb_val+pos_pixel_val)/num_testset))
        else:
            # print validate summary data 
            print("\rValidation : giou_valid_loss:{:7.2f} - conf_valid_loss:{:7.2f} - prob_valid_loss:{:7.2f} - total_valid_loss:{:7.2f}".
                format(giou_val/num_testset, conf_val/num_testset, prob_val/num_testset, total_val/num_testset))

        if not USE_SUPERVISION:
            detection_loss = total_val

        if TRAIN_SAVE_CHECKPOINT and not TRAIN_SAVE_BEST_ONLY:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME+"_val_loss_{:7.2f}".format(detection_loss/num_testset))
            yolo.save_weights(save_directory)
        if TRAIN_SAVE_BEST_ONLY and best_val_loss>detection_loss/num_testset:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME)
            yolo.save_weights(save_directory)
            best_val_loss = detection_loss/num_testset
            print("Save best weights at epoch = ", epoch+1, end="\n")
    if USE_SUPERVISION:
        FLAG_USE_BACKBONE_EVALUATION = False

if __name__ == '__main__':
    main()
    sys.modules[__name__].__dict__.clear()
