#===============================================================#
#                                                               #
#   File name   : YOLOv4_model.py                               #
#   Author      : hxnghia99                                     #
#   Created date: May 24th, 2022                                #
#   GitHub      : https://github.com/hxnghia99/Yolov4_Learning  #
#   Description : YOLOv4 network architecture                   #
#                                                               #
#===============================================================#



import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, ZeroPadding2D, MaxPool2D, Input, UpSampling2D
from tensorflow.keras.regularizers import L2
from YOLOv4_config import *
from YOLOv4_utils import *


#Configure Batch Normalization layer for 2 trainable parameters
class BatchNormalization(tf.keras.layers.BatchNormalization):
    def call(self, x, training=False):                      # BN has 2 types: training mode , inference mode
        if not training:                                    # training = False: using inference mode
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

#Mish activation function
mish = lambda x: x * tf.math.tanh(tf.math.softplus(x))

#Define the fundamental convolutional layer of YOLOv4
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
        elif activate_type =='mish':
            conv_layer = mish(conv_layer)
    return conv_layer


#Define the residual block in YOLOv4
def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    short_cut = input_layer
    conv_layer = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
    conv_layer = convolutional(conv_layer , filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type)
    residual_output = short_cut + conv_layer
    return residual_output


#Define CSPDarknet53 network architecture
def CSPDarknet53(input_data):
    input_data = convolutional(input_data, (3, 3, 3, 32), activate_type="mish")                     #output: 416 x 416 x 32
    input_data = convolutional(input_data, (3, 3, 32, 64), downsample=True, activate_type='mish')   #output: 208 x 208 x 64
    
    #CSP block 1
    # First branch
    route = input_data
    route = convolutional(input_data, (1, 1, 64, 64), activate_type='mish')                         #output: 208 x 208 x 64
    # Second branch
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type='mish')                    #output: 208 x 208 x 64
    for _ in range(1):
        input_data = residual_block(input_data,  64,  32, 64, activate_type='mish')
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type='mish')                    #output: 208 x 208 x 64
    # Concatenation
    input_data = tf.concat([input_data, route], axis=-1)                                            #outout: 208 x 208 x 128
    input_data = convolutional(input_data, (1, 1, 128, 64), activate_type='mish')                   #output: 208 x 208 x 64
    # Downsampling
    input_data = convolutional(input_data, (3, 3, 64, 128), downsample=True, activate_type='mish')  #output: 104 x 104 x 128
    
    #CSP block 2
    # First branch
    route = input_data
    route = convolutional(input_data, (1, 1, 128, 64), activate_type='mish')                        #output: 104 x 104 x 64
    # Second branch
    input_data = convolutional(input_data, (1, 1, 128, 64), activate_type='mish')                   #output: 104 x 104 x 64
    for _ in range(2):
        input_data = residual_block(input_data, 64,  64, 64, activate_type='mish')
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type='mish')                    #output: 104 x 104 x 64
    # Concatenation
    input_data = tf.concat([input_data, route], axis=-1)                                            #outout: 104 x 104 x 128
    input_data = convolutional(input_data, (1, 1, 128, 128), activate_type='mish')                  #output: 104 x 104 x 128
    # Downsampling
    input_data = convolutional(input_data, (3, 3, 128, 256), downsample=True, activate_type='mish') #output: 52 x 52 x 256

    #CSP block 3
    # First branch
    route = input_data
    route = convolutional(input_data, (1, 1, 256, 128), activate_type='mish')                       #output: 52 x 52 x 128
    # Second branch
    input_data = convolutional(input_data, (1, 1, 256, 128), activate_type='mish')                  #output: 52 x 52 x 128
    for _ in range(8):
        input_data = residual_block(input_data, 128, 128, 128, activate_type='mish')
    input_data = convolutional(input_data, (1, 1, 128, 128), activate_type='mish')                  #output: 52 x 52 x 128
    # Concatenation
    input_data = tf.concat([input_data, route], axis=-1)                                            #outout: 52 x 52 x 256
    input_data = convolutional(input_data, (1, 1, 256, 256), activate_type='mish')                  #output: 52 x 52 x 256
    #Get feature map at large scale to concatenate later
    fmap_backbone_large = input_data     
    # Downsampling
    input_data = convolutional(input_data, (3, 3, 256, 512), downsample=True, activate_type='mish') #output: 26 x 26 x 512
    
    #CSP block 4
    # First branch
    route = input_data
    route = convolutional(input_data, (1, 1, 512, 256), activate_type='mish')                       #output: 26 x 26 x 256
    # Second branch
    input_data = convolutional(input_data, (1, 1, 512, 256), activate_type='mish')                  #output: 26 x 26 x 256
    for _ in range(8):
        input_data = residual_block(input_data, 256, 256, 256, activate_type='mish')
    input_data = convolutional(input_data, (1, 1, 256, 256), activate_type='mish')                  #output: 26 x 26 x 256
    # Concatenation
    input_data = tf.concat([input_data, route], axis=-1)                                            #outout: 26 x 26 x 512
    input_data = convolutional(input_data, (1, 1, 512, 512), activate_type='mish')                  #output: 26 x 26 x 512
    #Get feature map at medium scale to concatenate later
    fmap_backbone_medium = input_data
    # Downsampling
    input_data = convolutional(input_data, (3, 3, 512, 1024), downsample=True, activate_type='mish')#output: 13 x 13 x 1024
    
    #CSP block 5
    # First branch
    route = input_data
    route = convolutional(input_data, (1, 1, 1024, 512), activate_type='mish')                      #output: 13 x 13 x 512
    # Second branch
    input_data = convolutional(input_data, (1, 1, 1024, 512), activate_type='mish')                 #output: 13 x 13 x 512
    for _ in range(4):
        input_data = residual_block(input_data, 512, 512, 512, activate_type='mish')
    input_data = convolutional(input_data, (1, 1, 512, 512), activate_type='mish')                  #output: 13 x 13 x 512
    # Concatenation
    input_data = tf.concat([input_data, route], axis=-1)                                            #output: 13 x 13 x 1024
    input_data = convolutional(input_data, (1, 1, 1024, 1024), activate_type='mish')                #output: 13 x 13 x 1024

    #Compress information of feature map
    input_data = convolutional(input_data, (1, 1, 1024, 512))                                       #output: 13 x 13 x 512
    input_data = convolutional(input_data, (3, 3, 512, 1024))                                       #output: 13 x 13 x 1024
    input_data = convolutional(input_data, (1, 1, 1024, 512))                                       #output: 13 x 13 x 512
    #SPP block
    max_pooling_1 = MaxPool2D(pool_size=13, padding='SAME', strides=1)(input_data)
    max_pooling_2 = MaxPool2D(pool_size=9, padding='SAME', strides=1)(input_data)
    max_pooling_3 = MaxPool2D(pool_size=5, padding='SAME', strides=1)(input_data)
    input_data = tf.concat([max_pooling_1, max_pooling_2, max_pooling_3, input_data], axis=-1)      #output: 13 x 13 x 2048
    input_data = convolutional(input_data, (1, 1, 2048, 512))                                       
    input_data = convolutional(input_data, (3, 3, 512, 1024))
    input_data = convolutional(input_data, (1, 1, 1024, 512))                                       #output: 13 x 13 x 512

    return fmap_backbone_large, fmap_backbone_medium, input_data


#Upsampling
def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')


#Add neck layers to CSPDarknet53 and create YOLOv4 model
def YOLOv4_detector(input_layer, NUM_CLASS):
    # Create CSPDarknet53 network and 3 backbone features at large, medium and small scale
    fmap_backbone_large, fmap_backbone_medium, conv = CSPDarknet53(input_layer)   
    
    # *** PANet bottom up layers ***
    #upsampling 1
    PAN_route_1 = conv                                                              #output: 13 x 13 x 512
    conv = convolutional(conv, (1, 1, 512, 256))                                    #output: 13 x 13 x 256
    conv = UpSampling2D()(conv)                                                     #output: 26 x 26 x 256                                       
    fmap_backbone_medium = convolutional(fmap_backbone_medium, (1, 1, 512, 256))    #output: 26 x 26 x 256
    conv = tf.concat([fmap_backbone_medium, conv], axis=-1)                         #output: 26 x 26 x 512
    
    #Compress information of feature maps
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))                                    #output: 26 x 26 x 256  
    
    #upsampling 2
    PAN_route_2 = conv                                                              #output: 26 x 26 x 256
    conv = conv = convolutional(conv, (1, 1, 256, 128))                             #output: 26 x 26 x 128
    conv = UpSampling2D()(conv)                                                     #output: 52 x 52 x 128                                                     
    fmap_backbone_large = convolutional(fmap_backbone_large, (1, 1, 256, 128))      #output: 52 x 52 x 128
    conv = tf.concat([fmap_backbone_large, conv], axis=-1)                          #output: 52 x 52 x 256

    #Compress information of feature maps
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))                                    #output: 52 x 52 x 128

    # *** PANet top down layers ***
    #Small bbox convolutional output
    output_route_1 = conv
    output_route_1 = convolutional(output_route_1, (3, 3, 128, 256))                #output: 52 x 52 x 256
    conv_sbbox = convolutional(output_route_1, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
                                                                                    #output: 52 x 52 x 3*(NUM_CLASS+5)
    #Downsampling
    conv = convolutional(conv, (3, 3, 128, 256), downsample=True)                   #output: 26 x 26 x 256
    conv = tf.concat([conv, PAN_route_2], axis=-1)                                  #output: 26 x 26 x 512

    #Compress information of feature maps
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))                                    #output: 26 x 26 x 256

    #Medium bbox convolutional output
    output_route_2 = conv
    output_route_2 = convolutional(output_route_2, (3, 3, 256, 512))                #output: 26 x 26 x 512
    conv_mbbox = convolutional(output_route_2, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
                                                                                    #output: 26 x 26 x 3*(NUM_CLASS+5)
    #Downsampling
    conv = convolutional(conv, (3, 3, 256, 512), downsample=True)                   #output: 13 x 13 x 512
    conv = tf.concat([conv, PAN_route_1], axis=-1)                                  #output: 13 x 13 x 1024

    #Compress information of feature maps
    conv = convolutional(conv, (1, 1, 1024, 512))
    conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))
    conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))                                   #output: 13 x 13 x 512
    
    #Small bbox convolutional output
    conv = convolutional(conv, (3, 3, 512, 1024))                                   #output: 13 x 13 x 1024
    conv_lbbox = convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
                                                                                    #output: 13 x 13 x 3*(NUM_CLASS+5)
    return [conv_sbbox, conv_mbbox, conv_lbbox]


#Define function used to change the output tensor to the information of (bbox, confidence, class)
# i represents for the grid scales: (0,1,2) <--> (large, medium, small)
def decode(conv_output, NUM_CLASS, i=0):
    conv_shape       = tf.shape(conv_output)                # shape [batch_size, output_size, output_size, 255]           
    batch_size       = conv_shape[0]
    output_size_h      = conv_shape[1]
    output_size_w      = conv_shape[2]
    #Change the output_shape of each scale into [batch_size, output_size_h, output_size_w, 3, 85]
    conv_output = tf.reshape(conv_output, (batch_size, output_size_h, output_size_w, 3, 5 + NUM_CLASS))
    
    #Split the final dimension into 4 information (offset xy, offset wh, confidence, class probabilities)
    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS), axis=-1)
    #shape [batch_size, output_size_h, output_size_w, 3, ...]

    # Create the matrix of grid cell indexes
    x_gridcell, y_gridcell = tf.meshgrid(tf.range(output_size_w), tf.range(output_size_h))  #create 2 matrices of shape [output_size_h, output_size_w]
    xy_gridcell = tf.stack([x_gridcell, y_gridcell], axis=-1)                           #Stack at final axis to create shape [output_size_h, output_size_w, 2]
    xy_gridcell = tf.expand_dims(tf.expand_dims(xy_gridcell, axis=2), axis=0)           #Prepare shape [1, output_size_h, output_size_w, 1, 2]
    xy_gridcell = tf.tile(xy_gridcell, [batch_size, 1, 1, 3, 1])                        #Create indexes matrix of shape [batch_size, output_size_h, output_size_w, 3, 2]
    xy_gridcell = tf.cast(xy_gridcell, tf.float32)

    # Predicted boxes coordinates
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_gridcell) * YOLO_SCALE_OFFSET[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * YOLO_ANCHORS[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    #Predicted box confidence scores
    pred_conf = tf.sigmoid(conv_raw_conf)
    #Predicted box class probabilities 
    pred_prob = tf.sigmoid(conv_raw_prob)

    #Prediction of shape [batch_size, output_size_h, output_size_w, 3, 85]
    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def YOLOv4_Model(input_size=YOLO_INPUT_SIZE, input_channel=3, training=False, CLASSES_PATH=YOLO_COCO_CLASS_PATH):
    #Read coco class names file
    class_names = {}
    with open(CLASSES_PATH, 'r') as data:
        for ID, name in enumerate(data):
            class_names[ID] = name.strip('\n')
    NUM_CLASS = len(class_names)
    #Create input layer
    input_layer = Input([None, None, input_channel])

    conv_tensors = YOLOv4_detector(input_layer, NUM_CLASS)

    output_tensors = []
    for i, conv_tensor in enumerate(conv_tensors):                              #small -> medium -> large
        pred_tensor = decode(conv_tensor, NUM_CLASS, i)
        if training:                                                           
            output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)                                      #shape [3 or 6, batch_size, output_size, output_size, 3, 85]
    YOLOv4_model = tf.keras.Model(input_layer, output_tensors)
    return YOLOv4_model


if __name__ == '__main__':
    yolo_model = YOLOv4_Model()
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






