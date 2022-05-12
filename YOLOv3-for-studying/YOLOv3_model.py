#===============================================================#
#                                                               #
#   File name   : YOLOv3_model.py                               #
#   Author      : hxnghia99                                     #
#   Created date: April 28th, 2022                              #
#   GitHub      : https://github.com/hxnghia99/Yolov4_Learning  #
#   Description : YOLOv3 network architecture                   #
#                                                               #
#===============================================================#

import os
from turtle import down
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, ZeroPadding2D, Input
from tensorflow.keras.regularizers import L2
import numpy as np
from YOLOv3_config import *
from YOLOv3_utils import *


#Configure Batch Normalization layer for 2 trainable parameters
class BatchNormalization(tf.keras.layers.BatchNormalization):
    def call(self, x, training=False):                      # BN has 2 types: frozen state , inference mode
        if not training:                                    # training = False: using frozen mode
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


#Define the fundamental convolutional layer of YOLOv3
#filters_shape contains (filter width, filter height, filter channel, filter num)
def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky'):
    #add zero-padding when downsampling
    if downsample:
        input_layer = ZeroPadding2D(((1, 0), (1, 0)))(input_layer)                  #To create the exact size from computation
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'
    #add the 2D convolutional layer to the input_layer
    conv_layer = Conv2D(  filters       =   filters_shape[-1],          
                    kernel_size         =   filters_shape[0],
                    strides             =   strides,
                    padding             =   padding,
                    use_bias            =   not bn,
                    kernel_regularizer  =   L2(0.0005),                             #L2 regularizer: smaller weights -> simpler model
                    kernel_initializer  =   tf.random_normal_initializer(stddev=0.01),
                    bias_initializer    =   tf.constant_initializer(0.)
                    )(input_layer)
    #add batch normalization layer after convolution layer
    if bn:
        conv_layer = BatchNormalization()(conv_layer)
    #add ReLu activation
    if activate == True:
        if activate_type == "leaky":
            conv_layer = LeakyReLU(alpha=0.1)(conv_layer)
    return conv_layer


#Define the residual block in YOLOv3
def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    short_cut = input_layer
    conv_layer = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
    conv_layer = convolutional(conv_layer , filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type)
    residual_output = short_cut + conv_layer
    return residual_output


#Define Darknet53 network architecture
def Darknet53(input_data):
    input_data = convolutional(input_data, (3, 3,  3,  32))
    input_data = convolutional(input_data, (3, 3, 32,  64), downsample=True)
    #Residual block 1
    for i in range(1):
        input_data = residual_block(input_data,  64,  32, 64)
    input_data = convolutional(input_data, (3, 3,  64, 128), downsample=True)
    #Residual block 2
    for i in range(2):
        input_data = residual_block(input_data, 128,  64, 128)
    input_data = convolutional(input_data, (3, 3, 128, 256), downsample=True)
    #Residual block 3
    for i in range(8):
        input_data = residual_block(input_data, 256, 128, 256)
    #Get feature map at large scale to concatenate later
    fmap_backbone_large = input_data                                    
    input_data = convolutional(input_data, (3, 3, 256, 512), downsample=True)
    #Residual block 4
    for i in range(8):
        input_data = residual_block(input_data, 512, 256, 512)
    #Get feature map at medium scale to concatenate later
    fmap_backbone_medium = input_data
    input_data = convolutional(input_data, (3, 3, 512, 1024), downsample=True)
    #Residual block 5
    for i in range(4):
        input_data = residual_block(input_data, 1024, 512, 1024)
    return fmap_backbone_large, fmap_backbone_medium, input_data


#Upsampling
def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')


#Add detection layers to Darknet53 and create YOLOv3 model
def YOLOv3_detector(input_layer, NUM_CLASS):
    # Create Darknet53 network and 2 backbone features at large and medium scale
    fmap_backbone_large, fmap_backbone_medium, conv = Darknet53(input_layer)   
    # There are 5 subconvolutional layers for small-scaled feature map
    conv = convolutional(conv, (1, 1, 1024,  512))                            
    conv = convolutional(conv, (3, 3,  512, 1024))
    conv = convolutional(conv, (1, 1, 1024,  512))
    conv = convolutional(conv, (3, 3,  512, 1024))
    conv = convolutional(conv, (1, 1, 1024,  512))
    #Make 2 convolutional layers for small-scaled features after subconvolutional layers, result to predict large-sized objects, shape = [None, 13, 13, 255]
    conv_small_scale_branch = convolutional(conv, (3, 3, 512, 1024))
    conv_large_bbox = convolutional(conv_small_scale_branch, (1, 1, 1024, 3*(NUM_CLASS + 5)), activate=False, bn=False)

    #Add 1 convolutional layers after the feature maps behind subconvolutional layers
    conv = convolutional(conv, (1, 1,  512,  256))
    #Upsampling using the nearest neighbor interpolation method
    conv = upsample(conv)                                     
    #Concatenate the upsampling medium feature map and the medium-scaled backbone feature map
    conv = tf.concat([conv, fmap_backbone_medium], axis=-1)
    #There are 5 subconvolutional layers for medium-scaled feature map
    conv = convolutional(conv, (1, 1, 768, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    #Make 2 convolutional layers for small-scaled features after subconvolutional layers, result to predict medium-sized objects, shape = [None, 26, 26, 255]
    conv_medium_scale_branch = convolutional(conv, (3, 3, 256, 512))
    conv_medium_bbox = convolutional(conv_medium_scale_branch, (1, 1, 512, 3*(NUM_CLASS + 5)), activate=False, bn=False)

    #Make convolutional layers the same as above
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = upsample(conv)
    conv = tf.concat([conv, fmap_backbone_large], axis=-1)
    conv = convolutional(conv, (1, 1, 384, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv_large_scale_branch = convolutional(conv, (3, 3, 128, 256))
    # The result is used to predict small-sized objects, shape = [None, 52, 52, 255]
    conv_small_bbox = convolutional(conv_large_scale_branch, (1, 1, 256, 3*(NUM_CLASS +5)), activate=False, bn=False)
        
    return [conv_small_bbox, conv_medium_bbox, conv_large_bbox]


#Define function used to change the output tensor to the information of (bbox, confidence, class)
# i represents for the grid scales: (0,1,2) <--> (large, medium, small)
def decode(conv_output, NUM_CLASS, i=0):
    conv_shape       = tf.shape(conv_output)                # shape [batch_size, output_size, output_size, 255]           
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]
    #Change the output_shape of each scale into [batch_size, output_size, output_size, 3, 85]
    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))
    
    #Split the final dimension into 4 information (offset xy, offset wh, confidence, class probabilities)
    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS), axis=-1)   #shape [batch_size, output_size, output_size, 3, ...]

    # Create the matrix of grid cell indexes
    x_gridcell, y_gridcell = tf.meshgrid(tf.range(output_size), tf.range(output_size))  #create 2 matrices of shape [output_size, output_size]
    xy_gridcell = tf.stack([x_gridcell, y_gridcell], axis=-1)                           #Stack at final axis to create shape [output_size, output_size, 2]
    xy_gridcell = tf.expand_dims(tf.expand_dims(xy_gridcell, axis=2), axis=0)           #Prepare shape [1, output_size, output_size, 1, 2]
    xy_gridcell = tf.tile(xy_gridcell, [batch_size, 1, 1, 3, 1])                        #Create indexes matrix of shape [batch_size, output_size, output_size, 3, 2]
    xy_gridcell = tf.cast(xy_gridcell, tf.float32)

    # Predicted boxes coordinates
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_gridcell) * YOLO_SCALE_OFFSET[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * YOLO_ANCHORS[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    #Predicted box confidence scores
    pred_conf = tf.sigmoid(conv_raw_conf)
    #Predicted box class probabilities 
    pred_prob = tf.sigmoid(conv_raw_prob)

    #Prediction of shape [batch_size, output_size, output_size, 3, 85]
    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def YOLOv3_Model(input_size=416, input_channel=3, training=False, CLASS_DIR = YOLO_COCO_CLASS_DIR):
    #Read coco class names file
    class_names = {}
    with open(CLASS_DIR, 'r') as f:
        for ID, name in enumerate(f):
            class_names[ID] = name.strip('\n')
    NUM_CLASS = len(class_names)
    #Create input layer
    input_layer = Input([input_size, input_size, input_channel])

    conv_tensors = YOLOv3_detector(input_layer, NUM_CLASS)

    output_tensors = []
    for i, conv_tensor in enumerate(conv_tensors):
        pred_tensor = decode(conv_tensor, NUM_CLASS, i)
        if training:                                                            # *SOS* need reading again
            output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)                                      #shape [3, batch_size, output_size, output_size, 3, 85]
    YOLOv3_model = tf.keras.Model(input_layer, output_tensors)
    return YOLOv3_model


if __name__ == '__main__':
    yolo_model = YOLOv3_Model()
    yolo_model.summary()
    



# class YOLOv3_model(tf.keras.Model):
#     def __init__(self, input_size=416, channels=3, training=False, CLASSES=YOLO_COCO_CLASSES):
#         #call parent constructor
#         super().__init__()

#     def call(self, inputs):
#         self.input_layer  = Input([input_size, input_size, channels])
        
        
#         NUM_CLASS = len(read_class_names(CLASSES))
#         output_tensors = []
#         for i, conv_tensor in enumerate(conv_tensors):
#             pred_tensor = decode(conv_tensor, NUM_CLASS, i)
#             if training: output_tensors.append(conv_tensor)
#             output_tensors.append(pred_tensor)


#     #Define a fundamental convolutional layer in YOLOv3
#     def convolutional(self, filters_shape, downsample=False, activate=True, activate_type='leaky', bn=True):
#         #add zero-padding when downsampling
#         if downsample:
#             self.input_layer = ZeroPadding2D(((1, 0), (1, 0)))(self.input_layer)
#             padding = 'valid'
#             strides = 2
#         else:
#             strides = 1
#             padding = 'same'
#         #add the 2D convolutional layer to the input_layer
#         conv = Conv2D(  filters             =   filters_shape[-1],          
#                         kernel_size         =   filters_shape[0],
#                         strides             =   strides,
#                         padding             =   padding,
#                         use_bias            =   not bn,
#                         kernel_regularizer  =   L2(0.0005),
#                         kernel_initializer  =   tf.random_normal_initializer(stddev=0.01),
#                         bias_initializer    =   tf.constant_initializer(0.)
#                       )(input_layer)
#         if bn:
#             conv = BatchNormalization()(conv)
#         if activate == True:
#             if activate_type == "leaky":
#                 conv = LeakyReLU(alpha=0.1)(conv)

#     def add_layers(architecture_config):






