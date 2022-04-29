from yolov1_dataset import *
from yolov1_model import model
from yolov1_lr_scheduler import Custom_LearningRate_Scheduler, lr_scheduler
from yolov1_loss import yolov1_loss

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras

# tf.compat.v1.enable_eager_execution()


mcp_save = ModelCheckpoint('YOLO-v1-for-studying/weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')

model.compile(loss=yolov1_loss, optimizer='adam')

model.fit(  x=My_training_generator,
            # x=training_images, y=training_labels,
            steps_per_epoch= int(len(train_images_path)/batch_size),
            # batch_size=1,
            epochs=135,
            verbose=1,
            workers=4,
            # validation_data=(val_images, val_labels),
            validation_data=My_validation_generator,
            validation_steps=int(len(val_images_path)/batch_size),
            callbacks=[
                Custom_LearningRate_Scheduler(lr_scheduler), 
                mcp_save]
            )
