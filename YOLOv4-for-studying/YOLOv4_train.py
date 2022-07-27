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
from YOLOv4_model   import YOLOv4_Model
from YOLOv4_loss    import compute_loss
from YOLOv4_utils   import load_yolov4_weights
from YOLOv4_config  import *



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
    if TRAIN_TRANSFER:
        Darknet = YOLOv4_Model(CLASSES_PATH=YOLO_COCO_CLASS_PATH)
        load_yolov4_weights(Darknet, CSPDarknet_weights) # use darknet weights
    #Create YOLO model
    yolo = YOLOv4_Model(training=True, CLASSES_PATH=YOLO_CLASS_PATH)
    if TRAIN_TRANSFER:
        for i, l in enumerate(Darknet.layers):
            layer_weights = l.get_weights()
            if layer_weights != []:
                try:
                    yolo.layers[i].set_weights(layer_weights)
                except:
                    print("skipping", yolo.layers[i].name)
    #Create Adam optimizers
    optimizer = tf.keras.optimizers.Adam()
    
    #Create log folder and summary
    if os.path.exists(TRAIN_LOGDIR): shutil.rmtree(TRAIN_LOGDIR)
    training_writer = tf.summary.create_file_writer(TRAIN_LOGDIR+'training/')
    validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR+'validation/')


    #Create training function for each batch
    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = yolo(image_data, training=True)       #conv+pred: small -> medium -> large : shape [scale, batch size, output size, output size, ...]
            giou_loss=conf_loss=prob_loss=0
            num_scales = len(YOLO_SCALE_OFFSET)
            #calculate loss at each scale  
            for i in range(num_scales):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES_PATH=YOLO_CLASS_PATH)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]
            #calculate total of loss
            total_loss = giou_loss + conf_loss + prob_loss
            #backpropagate gradients
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
            training_writer.flush()     

        return global_steps.numpy(), optimizer.lr.numpy(), giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    #Create validation function after each epoch
    def validate_step(image_data, target):
        pred_result = yolo(image_data, training=False)
        giou_loss=conf_loss=prob_loss=0
        grid = len(YOLO_SCALE_OFFSET)
        #calculate loss at each each
        for i in range(grid):
            conv, pred = pred_result[i*2], pred_result[i*2+1]
            loss_items = compute_loss(pred, conv, *target[i], i, CLASSES_PATH=YOLO_CLASS_PATH)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]
        total_loss = giou_loss + conf_loss + prob_loss
        return giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    best_val_loss = 1000 # should be large at start
    #For each epoch, do training and validating
    for epoch in range(TRAIN_EPOCHS):
        #Get a batch of training data to train
        giou_train, conf_train, prob_train, total_train = 0, 0, 0, 0
        for image_data, target in trainset:
            results = train_step(image_data, target)            #result = [global steps, learning rate, coor_loss, conf_loss, prob_loss, total_loss]
            current_step = results[0] % steps_per_epoch
            print("epoch ={:2.0f} step= {:5.0f}/{} : lr={:.10f} - giou_loss={:7.2f} - conf_loss={:7.2f} - prob_loss={:7.2f} - total_loss={:7.2f}"
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
        training_writer.flush()

        # print validate summary data
        print("\n\n TRAINING")
        print("epoch={:2.0f} : giou_train_loss:{:7.2f} - conf_train_loss:{:7.2f} - prob_train_loss:{:7.2f} - total_train_loss:{:7.2f}".
              format(epoch+1, giou_train/steps_per_epoch, conf_train/steps_per_epoch, prob_train/steps_per_epoch, total_train/steps_per_epoch))
        
        
        #If we do not have testing dataset, we save weights for every epoch
        if len(testset) == 0:
            print("configure TEST options to validate model")
            yolo.save_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME))
            continue
        #Validating the model with testing dataset
        num_testset = len(testset)
        giou_val, conf_val, prob_val, total_val = 0, 0, 0, 0
        current_step = 0
        print(" VALIDATION ")
        for image_data, target in testset:
            results = validate_step(image_data, target)
            print("Processing: {:5.0f}/{}".format(current_step, validate_steps_per_epoch))
            current_step += 1
            giou_val += results[0]
            conf_val += results[1]
            prob_val += results[2]
            total_val += results[3]

        # writing validate summary data
        with validate_writer.as_default():
            tf.summary.scalar("loss/total_val", total_val/num_testset, step=epoch)
            tf.summary.scalar("loss/giou_val", giou_val/num_testset, step=epoch)
            tf.summary.scalar("loss/conf_val", conf_val/num_testset, step=epoch)
            tf.summary.scalar("loss/prob_val", prob_val/num_testset, step=epoch)
        validate_writer.flush()

        # print validate summary data 
        print("epoch={:2.0f} : giou_val_loss:{:7.2f} - conf_val_loss:{:7.2f} - prob_val_loss:{:7.2f} - total_val_loss:{:7.2f}\n\n".
              format(epoch+1, giou_val/num_testset, conf_val/num_testset, prob_val/num_testset, total_val/num_testset))

        if TRAIN_SAVE_CHECKPOINT and not TRAIN_SAVE_BEST_ONLY:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME+"_val_loss_{:7.2f}".format(total_val/num_testset))
            yolo.save_weights(save_directory)
        if TRAIN_SAVE_BEST_ONLY and best_val_loss>total_val/num_testset:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME)
            yolo.save_weights(save_directory)
            best_val_loss = total_val/num_testset

if __name__ == '__main__':
    main()
    sys.modules[__name__].__dict__.clear()
