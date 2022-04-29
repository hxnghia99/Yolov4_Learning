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


#Read coco file to extract class
def read_class_names(class_file_name):
    # loads class name from a file
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

#Configure Batch Normalization layer for 2 trainable parameters
class BatchNormalization(tf.keras.layers.BatchNormalization):
    def call(self, x, training=False):                      # BN has 2 type: frozen state , inference mode
        if not training:                                    # training = False: using frozen mode
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


#Define the fundamental convolutional layer of YOLOv3
#filters_shape contains (filter width, filter height, filter channel, filter num)
def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky'):
    #add zero-padding when downsampling
    if downsample:
        input_layer = ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
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
                    kernel_regularizer  =   L2(0.0005),
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

    #Add 1 convolutional layers after the feature maps after subconvolutional layers
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
# i represents for the grid scales
def decode(conv_output, NUM_CLASS, i=0):
    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]
    #Change the output_shape of each scale into [batch_size, output_size, output_size, 3, 85]
    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))
    
    #Split the final dimension into 4 information (offset xy, offset wh, confidence, class probabilities)
    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS), axis=-1)

    # next need Draw the grid. Where output_size is equal to 13, 26 or 52  
    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)
    
    #xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    #xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    #y_grid = tf.cast(xy_grid, tf.float32)

    # Calculate the center position of the prediction box:
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    # Calculate the length and width of the prediction box:
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]

    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_conf = tf.sigmoid(conv_raw_conf) # object box calculates the predicted confidence
    pred_prob = tf.sigmoid(conv_raw_prob) # calculating the predicted probability category box object

    # calculating the predicted probability category box object
    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)









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






