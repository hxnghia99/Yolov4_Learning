#===============================================================#
#                                                               #
#   File name   : YOLOv4_model.py                               #
#   Author      : hxnghia99                                     #
#   Created date: May 24th, 2022                                #
#   GitHub      : https://github.com/hxnghia99/Yolov4_Learning  #
#   Description : YOLOv4 network architecture                   #
#                                                               #
#===============================================================#



from statistics import mode
from sympy import Mod
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, ZeroPadding2D, MaxPool2D, Input, UpSampling2D, Conv2DTranspose
from tensorflow.keras.regularizers import L2
from YOLOv4_config import *

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



#filters_shape contains (filter width, filter height, filter channel, filter num)
def deconvolutional(input_layer, filters_shape, upsample=False, activate=True, bn=True, activate_type='leaky'):
    #add zero-padding when downsampling
    if upsample:
        # input_layer = ZeroPadding2D(((1, 0), (1, 0)))(input_layer)                  #To create the exact size from computation
        padding = 'same'
        strides = 2
    else:
        strides = 1
        padding = 'same'
    #add the 2D deconvolutional layer to the input_layer
    conv_layer = Conv2DTranspose(  filters       =   filters_shape[-1],          
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

    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 32, 64), downsample=True, activate_type='mish')   #output: 208 x 208 x 64                 #2

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
    input_data = convolutional(input_data, (1, 1, 128, 64), activate_type='mish')                   #output: 208 x 208 x 64                 #8
    # Downsampling
    route_2 = input_data
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
    input_data = convolutional(input_data, (1, 1, 128, 128), activate_type='mish')                  #output: 104 x 104 x 128            #17
    # Downsampling
    route_3 = input_data
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
    input_data = convolutional(input_data, (1, 1, 256, 256), activate_type='mish')                  #output: 52 x 52 x 256             #38
    

    if MODEL_BRANCH_TYPE[1] == "P3":
        """ High resolution P3 """
        #Compress information of feature map
        input_data = convolutional(input_data, (1, 1, 256, 128))                                        #output: 52 x 52 x 128
        input_data = convolutional(input_data, (3, 3, 128, 256))                                        #output: 52 x 52 x 256
        input_data = convolutional(input_data, (1, 1, 256, 128))                                        #output: 52 x 52 x 128
        #SPP block
        max_pooling_1 = MaxPool2D(pool_size=11, padding='SAME', strides=1)(input_data)
        max_pooling_2 = MaxPool2D(pool_size=7, padding='SAME', strides=1)(input_data)
        max_pooling_3 = MaxPool2D(pool_size=4, padding='SAME', strides=1)(input_data)
        input_data = tf.concat([max_pooling_1, max_pooling_2, max_pooling_3, input_data], axis=-1)      #output: 52 x 52 x 512
        input_data = convolutional(input_data, (1, 1, 512, 128))                                       
        input_data = convolutional(input_data, (3, 3, 128, 256))
        input_data = convolutional(input_data, (1, 1, 256, 128))                                        #output: 52 x 52 x 128

        return route_1, route_2, route_3, input_data
    
    
    #Get feature map at large scale to concatenate later
    route_4 = input_data     
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
    input_data = convolutional(input_data, (1, 1, 512, 512), activate_type='mish')                  #output: 26 x 26 x 512             #59
    
    if MODEL_BRANCH_TYPE[1] == "P4":
        """ High resolution P4 """
        #Compress information of feature map
        input_data = convolutional(input_data, (1, 1, 512, 256))                                       #output: 13 x 13 x 512
        input_data = convolutional(input_data, (3, 3, 256, 512))                                       #output: 13 x 13 x 1024
        input_data = convolutional(input_data, (1, 1, 512, 256))                                       #output: 13 x 13 x 512
        #SPP block
        max_pooling_1 = MaxPool2D(pool_size=13, padding='SAME', strides=1)(input_data)
        max_pooling_2 = MaxPool2D(pool_size=9, padding='SAME', strides=1)(input_data)
        max_pooling_3 = MaxPool2D(pool_size=5, padding='SAME', strides=1)(input_data)
        input_data = tf.concat([max_pooling_1, max_pooling_2, max_pooling_3, input_data], axis=-1)      #output: 13 x 13 x 2048
        input_data = convolutional(input_data, (1, 1, 1024, 256))                                       
        input_data = convolutional(input_data, (3, 3, 256, 512))
        input_data = convolutional(input_data, (1, 1, 512, 256))                                       #output: 13 x 13 x 512

        return route_1, route_2, route_3, route_4, input_data
    
    if MODEL_BRANCH_TYPE[1] == "P5" or MODEL_BRANCH_TYPE[1] == "P5n":
        #Get feature map at medium scale to concatenate later
        route_5 = input_data
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
        input_data = convolutional(input_data, (1, 1, 1024, 1024), activate_type='mish')                #output: 13 x 13 x 1024         #72

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
        input_data = convolutional(input_data, (1, 1, 1024, 512))                                       #output: 13 x 13 x 512          #78

        if MODEL_BRANCH_TYPE[1] == "P5":
            return route_1, route_2, route_3, route_4, route_5, input_data
        elif MODEL_BRANCH_TYPE[1] == "P5n":
            return route_4, route_5, input_data

        


#Add neck layers to CSPDarknet53 and create YOLOv4 model
def YOLOv4_detector(input_layer, NUM_CLASS):
    # Create CSPDarknet53 network and 3 backbone features at large, medium and small scale
    if MODEL_BRANCH_TYPE[1] == "P5n":
        route_4, route_5, conv = CSPDarknet53(input_layer)
    elif MODEL_BRANCH_TYPE[1] == "P5":
        route_1, route_2, route_3, route_4, route_5, conv = CSPDarknet53(input_layer)
    elif MODEL_BRANCH_TYPE[1] == "P4":
        route_1, route_2, route_3, route_4, conv = CSPDarknet53(input_layer)   
    elif MODEL_BRANCH_TYPE[1] == "P3":
        route_1, route_2, route_3, conv = CSPDarknet53(input_layer)   

    """ PANet bottom up layers """
    if MODEL_BRANCH_TYPE[1] == "P5" or MODEL_BRANCH_TYPE[1] == "P5n":
        #upsampling 1
        route_6 = conv                                                              #output: 13 x 13 x 512
        conv = convolutional(conv, (1, 1, 512, 256))                                    #output: 13 x 13 x 256
        conv = UpSampling2D()(conv)                                                     #output: 26 x 26 x 256                                       
        route_5 = convolutional(route_5, (1, 1, 512, 256))    #output: 26 x 26 x 256
        conv = tf.concat([route_5, conv], axis=-1)                         #output: 26 x 26 x 512
        
        #Compress information of feature maps
        conv = convolutional(conv, (1, 1, 512, 256))
        conv = convolutional(conv, (3, 3, 256, 512))
        conv = convolutional(conv, (1, 1, 512, 256))
        conv = convolutional(conv, (3, 3, 256, 512))
        conv = convolutional(conv, (1, 1, 512, 256))                                    #output: 26 x 26 x 256  
    
    if MODEL_BRANCH_TYPE[1] == "P5" or MODEL_BRANCH_TYPE[1] == "P4" or MODEL_BRANCH_TYPE[1] == "P5n":
        #upsampling 2
        route_5 = conv                                                              #output: 26 x 26 x 256
        conv = conv = convolutional(conv, (1, 1, 256, 128))                             #output: 26 x 26 x 128
        conv = UpSampling2D()(conv)                                                     #output: 52 x 52 x 128                                                     
        route_4 = convolutional(route_4, (1, 1, 256, 128))      #output: 52 x 52 x 128
        conv = tf.concat([route_4, conv], axis=-1)                          #output: 52 x 52 x 256

        #Compress information of feature maps
        conv = convolutional(conv, (1, 1, 256, 128))
        conv = convolutional(conv, (3, 3, 128, 256))
        conv = convolutional(conv, (1, 1, 256, 128))
        conv = convolutional(conv, (3, 3, 128, 256))
        conv = convolutional(conv, (1, 1, 256, 128))                                    #output: 52 x 52 x 128

    """ Additional upsampling: 3 times to original image size """
    if MODEL_BRANCH_TYPE[1] == "P5" or MODEL_BRANCH_TYPE[1] == "P4" or MODEL_BRANCH_TYPE[1] == "P3":
        #upsampling 3
        # route_4 = conv                                                                #output: 52 x 52 x 128
        conv = convolutional(conv, (1, 1, 128, 64))                                     #output: 52 x 52 x 64
        conv = UpSampling2D()(conv)                                                     #output: 104 x 104 x 64                                       
        route_3 = convolutional(route_3, (1, 1, 128, 64))                               #output: 104 x 104 x 64
        conv = tf.concat([route_3, conv], axis=-1)                                      #output: 104 x 104 x 128
        #Compress information of feature maps
        conv = convolutional(conv, (1, 1, 128, 64))
        conv = convolutional(conv, (3, 3, 64, 128))
        conv = convolutional(conv, (1, 1, 128, 64))
        conv = convolutional(conv, (3, 3, 64, 128))
        conv = convolutional(conv, (1, 1, 128, 64))                                     #output: 104 x 104 x 64


        #upsampling 4
        route_3 = conv                                                                  #output: 104 x 104 x 64
        conv = convolutional(conv, (1, 1, 64, 32))                                      #output: 104 x 104 x 32
        conv = UpSampling2D()(conv)                                                     #output: 208 x 208 x 32                                       
        route_2 = convolutional(route_2, (1, 1, 64, 32))                                #output: 208 x 208 x 32         
        conv = tf.concat([route_2, conv], axis=-1)                                      #output: 208 x 208 x 64     
        #Compress information of feature maps
        conv = convolutional(conv, (1, 1, 64, 32))
        conv = convolutional(conv, (3, 3, 32, 64))
        conv = convolutional(conv, (1, 1, 64, 32))
        conv = convolutional(conv, (3, 3, 32, 64))
        conv = convolutional(conv, (1, 1, 64, 32))                                      #output: 208 x 208 x 32  


        #upsampling 5
        route_2 = conv                                                                  #output: 208 x 208 x 32
        conv = convolutional(conv, (1, 1, 32, 16))                                      #output: 208 x 208 x 16
        conv = UpSampling2D()(conv)                                                     #output: 416 x 416 x 16 
        route_1 = convolutional(route_1, (1, 1, 32, 16))                                #output: 416 x 416 x 16         
        conv = tf.concat([route_1, conv], axis=-1)                                      #output: 416 x 416 x 32   
        #Compress information of feature maps
        conv = convolutional(conv, (1, 1, 32, 16))
        conv = convolutional(conv, (3, 3, 16, 32))
        conv = convolutional(conv, (1, 1, 32, 16))
        conv = convolutional(conv, (3, 3, 16, 32))
        conv = convolutional(conv, (1, 1, 32, 16))                                      #output: 416 x 416 x 16


    if MODEL_BRANCH_TYPE[0] == "P3n":
        route_4 = conv
        conv = convolutional(conv, (3, 3, 128, 256))
        conv_sbbox = convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        conv = convolutional(route_4, (3, 3, 128, 256), downsample=True)
        conv = tf.concat([conv, route_5], axis=-1)

        conv = convolutional(conv, (1, 1, 512, 256))
        conv = convolutional(conv, (3, 3, 256, 512))
        conv = convolutional(conv, (1, 1, 512, 256))
        conv = convolutional(conv, (3, 3, 256, 512))
        conv = convolutional(conv, (1, 1, 512, 256))

        route_5 = conv
        conv = convolutional(conv, (3, 3, 256, 512))
        conv_mbbox = convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        conv = convolutional(route_5, (3, 3, 256, 512), downsample=True)
        conv = tf.concat([conv, route_6], axis=-1)

        conv = convolutional(conv, (1, 1, 1024, 512))
        conv = convolutional(conv, (3, 3, 512, 1024))
        conv = convolutional(conv, (1, 1, 1024, 512))
        conv = convolutional(conv, (3, 3, 512, 1024))
        conv = convolutional(conv, (1, 1, 1024, 512))

        conv = convolutional(conv, (3, 3, 512, 1024))
        conv_lbbox = convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        return [conv_sbbox, conv_mbbox, conv_lbbox]



    if MODEL_BRANCH_TYPE[0] == "P0":
        """PANet top down layers: P0 - P2"""
        #Small bbox convolutional output
        route_1 = conv
        route_1 = convolutional(route_1, (3, 3, 16, 32))                                #output: 416 x 416 x 32 
        conv_sbbox = convolutional(route_1, (1, 1, 32, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
                                                                                        #output: 416 x 416 x 3*(NUM_CLASS+5)
        #Downsampling 1
        conv = convolutional(conv, (3, 3, 16, 32), downsample=True)                     #output: 208 x 208 x 32
        conv = tf.concat([conv, route_2], axis=-1)                                      #output: 208 x 208 x 64

        #Compress information of feature maps
        conv = convolutional(conv, (1, 1, 64, 32))
        conv = convolutional(conv, (3, 3, 32, 64))
        conv = convolutional(conv, (1, 1, 64, 32))
        conv = convolutional(conv, (3, 3, 32, 64))
        conv = convolutional(conv, (1, 1, 64, 32))                                      #output: 208 x 208 x 32

        #Medium bbox convolutional output
        route_2 = conv
        route_2 = convolutional(route_2, (3, 3, 32, 64))                                #output: 208 x 208 x 64
        conv_mbbox = convolutional(route_2, (1, 1, 64, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
                                                                                        #output: 208 x 208 x 3*(NUM_CLASS+5)
        #Downsampling 2
        conv = convolutional(conv, (3, 3, 32, 64), downsample=True)                     #output: 104 x 104 x 64
        conv = tf.concat([conv, route_3], axis=-1)                                      #output: 104 x 104 x 128

        #Compress information of feature maps
        conv = convolutional(conv, (1, 1, 128, 64))
        conv = convolutional(conv, (3, 3, 64, 128))
        conv = convolutional(conv, (1, 1, 128, 64))
        conv = convolutional(conv, (3, 3, 64, 128))
        conv = convolutional(conv, (1, 1, 128, 64))                                     #output: 104 x 104 x 64
        
        #Small bbox convolutional output
        conv = convolutional(conv, (3, 3, 64, 128))                                     #output: 104 x 104 x 128
        conv_lbbox = convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
                                                                                        #output: 104 x 104 x 3*(NUM_CLASS+5)
        return [conv_sbbox, conv_mbbox, conv_lbbox]

    elif MODEL_BRANCH_TYPE[0] == "P(-1)":
        #upsampling 6
        route_1 = conv                                                                  #output: 208 x 208 x 32
        conv = convolutional(conv, (1, 1, 16, 16))                                      #output: 208 x 208 x 16
        conv = UpSampling2D()(conv)                                                     #output: 416 x 416 x 16   
        #Compress information of feature maps
        conv = convolutional(conv, (1, 1, 16, 8))
        conv = convolutional(conv, (3, 3, 8, 16))
        conv = convolutional(conv, (1, 1, 16, 8))
        conv = convolutional(conv, (3, 3, 8, 16))
        conv = convolutional(conv, (1, 1, 16, 8))                                      #output: 416 x 416 x 16

        """PANet top down layers : P(-1) - P1 """
        #Small bbox convolutional output
        route_0 = conv
        route_0 = convolutional(route_0, (3, 3, 8, 16))                                #output: 416 x 416 x 32 
        conv_sbbox = convolutional(route_0, (1, 1, 16, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
                                                                                        #output: 416 x 416 x 3*(NUM_CLASS+5)
        #Downsampling 1
        conv = convolutional(conv, (3, 3, 8, 16), downsample=True)                     #output: 208 x 208 x 32
        conv = tf.concat([conv, route_1], axis=-1)                                      #output: 208 x 208 x 64

        #Compress information of feature maps
        conv = convolutional(conv, (1, 1, 32, 16))
        conv = convolutional(conv, (3, 3, 16, 32))
        conv = convolutional(conv, (1, 1, 32, 16))
        conv = convolutional(conv, (3, 3, 16, 32))
        conv = convolutional(conv, (1, 1, 32, 16))                                      #output: 208 x 208 x 32

        #Medium bbox convolutional output
        route_1 = conv
        route_1 = convolutional(route_1, (3, 3, 16, 32))                                #output: 208 x 208 x 64
        conv_mbbox = convolutional(route_1, (1, 1, 32, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
                                                                                        #output: 208 x 208 x 3*(NUM_CLASS+5)
        #Downsampling 2
        conv = convolutional(conv, (3, 3, 16, 32), downsample=True)                     #output: 104 x 104 x 64
        conv = tf.concat([conv, route_2], axis=-1)                                      #output: 104 x 104 x 128

        #Compress information of feature maps
        conv = convolutional(conv, (1, 1, 64, 32))
        conv = convolutional(conv, (3, 3, 32, 64))
        conv = convolutional(conv, (1, 1, 64, 32))
        conv = convolutional(conv, (3, 3, 32, 64))
        conv = convolutional(conv, (1, 1, 64, 32))                                     #output: 104 x 104 x 64

        #Small bbox convolutional output
        conv = convolutional(conv, (3, 3, 32, 64))                                     #output: 104 x 104 x 128
        conv_lbbox = convolutional(conv, (1, 1, 64, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
                                                                                        #output: 104 x 104 x 3*(NUM_CLASS+5)
        return [conv_sbbox, conv_mbbox, conv_lbbox]








# def cspdarknet53(input_data):
#     input_data = convolutional(input_data, (3, 3,  3,  32), activate_type="mish")
#     input_data = convolutional(input_data, (3, 3, 32,  64), downsample=True, activate_type="mish")                #2

#     route = input_data
#     route = convolutional(route, (1, 1, 64, 64), activate_type="mish")
#     input_data = convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
#     for i in range(1):
#         input_data = residual_block(input_data,  64,  32, 64, activate_type="mish")
#     input_data = convolutional(input_data, (1, 1, 64, 64), activate_type="mish")

#     input_data = tf.concat([input_data, route], axis=-1)
#     input_data = convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
#     input_data = convolutional(input_data, (3, 3, 64, 128), downsample=True, activate_type="mish")
#     route = input_data
#     route = convolutional(route, (1, 1, 128, 64), activate_type="mish")
#     input_data = convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
#     for i in range(2):
#         input_data = residual_block(input_data, 64,  64, 64, activate_type="mish")
#     input_data = convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
#     input_data = tf.concat([input_data, route], axis=-1)

#     input_data = convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
#     input_data = convolutional(input_data, (3, 3, 128, 256), downsample=True, activate_type="mish")
#     route = input_data
#     route = convolutional(route, (1, 1, 256, 128), activate_type="mish")
#     input_data = convolutional(input_data, (1, 1, 256, 128), activate_type="mish")
#     for i in range(8):
#         input_data = residual_block(input_data, 128, 128, 128, activate_type="mish")
#     input_data = convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
#     input_data = tf.concat([input_data, route], axis=-1)

#     input_data = convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
#     route_1 = input_data
#     input_data = convolutional(input_data, (3, 3, 256, 512), downsample=True, activate_type="mish")
#     route = input_data
#     route = convolutional(route, (1, 1, 512, 256), activate_type="mish")
#     input_data = convolutional(input_data, (1, 1, 512, 256), activate_type="mish")
#     for i in range(8):
#         input_data = residual_block(input_data, 256, 256, 256, activate_type="mish")
#     input_data = convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
#     input_data = tf.concat([input_data, route], axis=-1)

#     input_data = convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
#     route_2 = input_data
#     input_data = convolutional(input_data, (3, 3, 512, 1024), downsample=True, activate_type="mish")
#     route = input_data
#     route = convolutional(route, (1, 1, 1024, 512), activate_type="mish")
#     input_data = convolutional(input_data, (1, 1, 1024, 512), activate_type="mish")
#     for i in range(4):
#         input_data = residual_block(input_data, 512, 512, 512, activate_type="mish")
#     input_data = convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
#     input_data = tf.concat([input_data, route], axis=-1)

#     input_data = convolutional(input_data, (1, 1, 1024, 1024), activate_type="mish")
#     input_data = convolutional(input_data, (1, 1, 1024, 512))
#     input_data = convolutional(input_data, (3, 3, 512, 1024))
#     input_data = convolutional(input_data, (1, 1, 1024, 512))

#     max_pooling_1 = tf.keras.layers.MaxPool2D(pool_size=13, padding='SAME', strides=1)(input_data)
#     max_pooling_2 = tf.keras.layers.MaxPool2D(pool_size=9, padding='SAME', strides=1)(input_data)
#     max_pooling_3 = tf.keras.layers.MaxPool2D(pool_size=5, padding='SAME', strides=1)(input_data)
#     input_data = tf.concat([max_pooling_1, max_pooling_2, max_pooling_3, input_data], axis=-1)

#     input_data = convolutional(input_data, (1, 1, 2048, 512))
#     input_data = convolutional(input_data, (3, 3, 512, 1024))
#     input_data = convolutional(input_data, (1, 1, 1024, 512))

#     return route_1, route_2, input_data


# def YOLOv4_detector(input_layer, NUM_CLASS):
#     route_1, route_2, conv = cspdarknet53(input_layer)

#     route = conv
#     conv = convolutional(conv, (1, 1, 512, 256))
#     conv = UpSampling2D()(conv)
#     route_2 = convolutional(route_2, (1, 1, 512, 256))
#     conv = tf.concat([route_2, conv], axis=-1)

#     conv = convolutional(conv, (1, 1, 512, 256))
#     conv = convolutional(conv, (3, 3, 256, 512))
#     conv = convolutional(conv, (1, 1, 512, 256))
#     conv = convolutional(conv, (3, 3, 256, 512))
#     conv = convolutional(conv, (1, 1, 512, 256))

#     route_2 = conv
#     conv = convolutional(conv, (1, 1, 256, 128))
#     conv = UpSampling2D()(conv)
#     route_1 = convolutional(route_1, (1, 1, 256, 128))
#     conv = tf.concat([route_1, conv], axis=-1)

#     conv = convolutional(conv, (1, 1, 256, 128))
#     conv = convolutional(conv, (3, 3, 128, 256))
#     conv = convolutional(conv, (1, 1, 256, 128))
#     conv = convolutional(conv, (3, 3, 128, 256))
#     conv = convolutional(conv, (1, 1, 256, 128))

#     route_1 = conv
#     conv = convolutional(conv, (3, 3, 128, 256))
#     conv_sbbox = convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

#     conv = convolutional(route_1, (3, 3, 128, 256), downsample=True)
#     conv = tf.concat([conv, route_2], axis=-1)

#     conv = convolutional(conv, (1, 1, 512, 256))
#     conv = convolutional(conv, (3, 3, 256, 512))
#     conv = convolutional(conv, (1, 1, 512, 256))
#     conv = convolutional(conv, (3, 3, 256, 512))
#     conv = convolutional(conv, (1, 1, 512, 256))

#     route_2 = conv
#     conv = convolutional(conv, (3, 3, 256, 512))
#     conv_mbbox = convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

#     conv = convolutional(route_2, (3, 3, 256, 512), downsample=True)
#     conv = tf.concat([conv, route], axis=-1)

#     conv = convolutional(conv, (1, 1, 1024, 512))
#     conv = convolutional(conv, (3, 3, 512, 1024))
#     conv = convolutional(conv, (1, 1, 1024, 512))
#     conv = convolutional(conv, (3, 3, 512, 1024))
#     conv = convolutional(conv, (1, 1, 1024, 512))

#     conv = convolutional(conv, (3, 3, 512, 1024))
#     conv_lbbox = convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

#     return [conv_sbbox, conv_mbbox, conv_lbbox]










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


def YOLOv4_Model(input_channel=3, training=False, CLASSES_PATH=YOLO_COCO_CLASS_PATH):
    #Read coco class names file
    class_names = {}
    with open(CLASSES_PATH, 'r') as data:
        for ID, name in enumerate(data):
            class_names[ID] = name.strip('\n')
    NUM_CLASS = len(class_names)
    #Create input layer
    input_layer = Input([128, 224, input_channel])

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






