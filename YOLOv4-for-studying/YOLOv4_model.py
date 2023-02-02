#===============================================================#
#                                                               #
#   File name   : YOLOv4_model.py                               #
#   Author      : hxnghia99                                     #
#   Created date: May 24th, 2022                                #
#   GitHub      : https://github.com/hxnghia99/Yolov4_Learning  #
#   Description : YOLOv4 network architecture                   #
#                                                               #
#===============================================================#


from YOLOv4_utils import *

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, ZeroPadding2D, MaxPool2D, Input, UpSampling2D, Conv2DTranspose, ReLU
from tensorflow.keras.regularizers import L2
from YOLOv4_config import *
from keras_flops import get_flops

# #Configure Batch Normalization layer for 2 trainable parameters
# class BatchNormalization(tf.keras.layers.BatchNormalization):
#     def call(self, x, training=False):                      # BN has 2 types: training mode , inference mode
#         if not training:                                    # training = False: using inference mode
#             training = tf.constant(False)
#         training = tf.logical_and(training, self.trainable)
#         return super().call(x, training)


k = CHANNEL_TIMES_K

#Mish activation function
mish = lambda x: x * tf.math.tanh(tf.math.softplus(x))

#Define the fundamental convolutional layer of YOLOv4
#filters_shape contains (filter width, filter height, filter channel, filter num)
def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky', dilation=False, dilation_rate=2):
    #add zero-padding when downsampling
    if downsample and not dilation:
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
                    kernel_regularizer  =   L2(0.0001),                             #L2 regularizer: smaller weights -> simpler model
                    kernel_initializer  =   tf.random_normal_initializer(stddev=0.01),
                    bias_initializer    =   tf.constant_initializer(0.),
                    dilation_rate       =   1 if not dilation else dilation_rate)(input_layer)
    #add batch normalization layer after convolution layer
    if bn:
        conv_layer = BatchNormalization()(conv_layer)
    #add ReLu activation
    if activate == True:
        if activate_type == "leaky":
            conv_layer = LeakyReLU(alpha=0.1)(conv_layer)
        elif activate_type =='mish':
            conv_layer = mish(conv_layer)
    if dilation and downsample:
        conv_layer = MaxPool2D(pool_size=(2,2), strides=2, padding='same')(conv_layer)
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
def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky', dilation=False):
    short_cut = input_layer
    conv_layer = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type, dilation= dilation)
    conv_layer = convolutional(conv_layer , filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type, dilation= dilation)
    residual_output = short_cut + conv_layer
    return residual_output


#Define CSPDarknet53 network architecture
def CSPDarknet53(input_data, dilation=False):
    input_data = convolutional(input_data, (3, 3, 3, 32*k), activate_type="mish", dilation=dilation)                     #output: 416 x 416 x 32  # 0(input) +5 -> 5 
    route_0 = input_data
    input_data = convolutional(input_data, (3, 3, 32*k, 64*k), downsample=True, activate_type='mish', dilation=dilation)   #output: 208 x 208 x 64  # 5 +1(ZeroPad)+5->11           #2

    #CSP block 1
    # First branch
    route = input_data
    route = convolutional(route, (1, 1, 64*k, 64*k), activate_type='mish', dilation=dilation)                         #output: 208 x 208 x 64  # 11 +5->16
    # Second branch
    input_data = convolutional(input_data, (1, 1, 64*k, 64*k), activate_type='mish', dilation=dilation)                    #output: 208 x 208 x 64  
    for _ in range(1):
        input_data = residual_block(input_data,  64*k,  32*k, 64*k, activate_type='mish', dilation=dilation)                      # 16 +5x2+1(add)->27
    input_data = convolutional(input_data, (1, 1, 64*k, 64*k), activate_type='mish', dilation=dilation)                    #output: 208 x 208 x 64  #27 +5x2(below+above)->37
    # Concatenation
    input_data = tf.concat([input_data, route], axis=-1)                                            #outout: 208 x 208 x 128                    #38 
    input_data = convolutional(input_data, (1, 1, 128*k, 64*k), activate_type='mish', dilation=dilation)                   #output: 208 x 208 x 64          #38 +5->43              #8
    # Downsampling
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 64*k, 128*k), downsample=True, activate_type='mish', dilation=dilation)  #output: 104 x 104 x 128     #43 +1(ZeroPad)+5->49
    
    #CSP block 2
    # First branch
    route = input_data
    route = convolutional(route, (1, 1, 128*k, 64*k), activate_type='mish', dilation=dilation)                        #output: 104 x 104 x 64      #49 +5->54
    # Second branch
    input_data = convolutional(input_data, (1, 1, 128*k, 64*k), activate_type='mish', dilation=dilation)                   #output: 104 x 104 x 64
    for _ in range(2):
        input_data = residual_block(input_data, 64*k,  64*k, 64*k, activate_type='mish', dilation=dilation)                                               #54 +5x2+1(add)+5x2+1(add) ->76
    input_data = convolutional(input_data, (1, 1, 64*k, 64*k), activate_type='mish', dilation=dilation)                    #output: 104 x 104 x 64      #76 +5x2(below+above)->86
    # Concatenation
    input_data = tf.concat([input_data, route], axis=-1)                                            #outout: 104 x 104 x 128                        #87
    input_data = convolutional(input_data, (1, 1, 128*k, 128*k), activate_type='mish', dilation=dilation)                  #output: 104 x 104 x 128     #87 +5->92                  #17
    # Downsampling
    route_2 = input_data
    input_data = convolutional(input_data, (3, 3, 128*k, 256*k), downsample=True, activate_type='mish', dilation=dilation) #output: 52 x 52 x 256       #92 +1(ZeroPad)+5->98

    #CSP block 3
    # First branch
    route = input_data
    route = convolutional(route, (1, 1, 256*k, 128*k), activate_type='mish', dilation=dilation)                       #output: 52 x 52 x 128       #98 +5->103
    # Second branch
    input_data = convolutional(input_data, (1, 1, 256*k, 128*k), activate_type='mish', dilation=dilation)                  #output: 52 x 52 x 128
    for _ in range(8):
        input_data = residual_block(input_data, 128*k, 128*k, 128*k, activate_type='mish', dilation=dilation)                                             #103 +[+5x2+1(add)]x8->191
    input_data = convolutional(input_data, (1, 1, 128*k, 128*k), activate_type='mish', dilation=dilation)                  #output: 52 x 52 x 128       #191 +5x2(below+above)->201
    # Concatenation
    input_data = tf.concat([input_data, route], axis=-1)                                            #outout: 52 x 52 x 256                          #202
    input_data = convolutional(input_data, (1, 1, 256*k, 256*k), activate_type='mish', dilation=dilation)                  #output: 52 x 52 x 256       #202 +5->207                #38
    

    if MODEL_BRANCH_TYPE[1] == "P3":
        """ High resolution P3 """
        #Compress information of feature map
        input_data = convolutional(input_data, (1, 1, 256*k, 128*k), dilation=dilation)                                        #output: 52 x 52 x 128
        input_data = convolutional(input_data, (3, 3, 128*k, 256*k), dilation=dilation)                                        #output: 52 x 52 x 256
        input_data = convolutional(input_data, (1, 1, 256*k, 128*k), dilation=dilation)                                        #output: 52 x 52 x 128
        #SPP block
        max_pooling_1 = MaxPool2D(pool_size=11, padding='SAME', strides=1)(input_data)
        max_pooling_2 = MaxPool2D(pool_size=7, padding='SAME', strides=1)(input_data)
        max_pooling_3 = MaxPool2D(pool_size=4, padding='SAME', strides=1)(input_data)
        input_data = tf.concat([max_pooling_1, max_pooling_2, max_pooling_3, input_data], axis=-1)      #output: 52 x 52 x 512
        input_data = convolutional(input_data, (1, 1, 512*k, 128*k), dilation=dilation)                                       
        input_data = convolutional(input_data, (3, 3, 128*k, 256*k), dilation=dilation)
        input_data = convolutional(input_data, (1, 1, 256*k, 128*k), dilation=dilation)                                        #output: 52 x 52 x 128

        return route_0, route_1, route_2, input_data
    
    
    #Get feature map at large scale to concatenate later
    route_3 = input_data     
    # Downsampling
    input_data = convolutional(input_data, (3, 3, 256*k, 512*k), downsample=True, activate_type='mish', dilation=dilation) #output: 26 x 26 x 512       #207 +1(ZeroPad)+5->213
    
    #CSP block 4
    # First branch
    route = input_data
    route = convolutional(route, (1, 1, 512*k, 256*k), activate_type='mish', dilation=dilation)                       #output: 26 x 26 x 256       #213 +5->218
    # Second branch
    input_data = convolutional(input_data, (1, 1, 512*k, 256*k), activate_type='mish', dilation=dilation)                  #output: 26 x 26 x 256
    for _ in range(8):
        input_data = residual_block(input_data, 256*k, 256*k, 256*k, activate_type='mish', dilation=dilation)                                             #218 +[+5x2+1]x8->306
    input_data = convolutional(input_data, (1, 1, 256*k, 256*k), activate_type='mish', dilation=dilation)                  #output: 26 x 26 x 256       #306 +5x2(below+above)->316
    # Concatenation
    input_data = tf.concat([input_data, route], axis=-1)                                            #outout: 26 x 26 x 512                          #317
    input_data = convolutional(input_data, (1, 1, 512*k, 512*k), activate_type='mish', dilation=dilation)                  #output: 26 x 26 x 512       #317 +5->322                #59
    
    if MODEL_BRANCH_TYPE[1] == "P4":
        """ High resolution P4 """
        #Compress information of feature map
        input_data = convolutional(input_data, (1, 1, 512*k, 256*k), dilation=dilation)                                       #output: 13 x 13 x 512
        input_data = convolutional(input_data, (3, 3, 256*k, 512*k), dilation=dilation)                                       #output: 13 x 13 x 1024
        input_data = convolutional(input_data, (1, 1, 512*k, 256*k), dilation=dilation)                                       #output: 13 x 13 x 512
        #SPP block
        max_pooling_1 = MaxPool2D(pool_size=13, padding='SAME', strides=1)(input_data)
        max_pooling_2 = MaxPool2D(pool_size=9, padding='SAME', strides=1)(input_data)
        max_pooling_3 = MaxPool2D(pool_size=5, padding='SAME', strides=1)(input_data)
        input_data = tf.concat([max_pooling_1, max_pooling_2, max_pooling_3, input_data], axis=-1)      #output: 13 x 13 x 2048
        input_data = convolutional(input_data, (1, 1, 1024*k, 256*k), dilation=dilation)                                       
        input_data = convolutional(input_data, (3, 3, 256*k, 512*k), dilation=dilation)
        input_data = convolutional(input_data, (1, 1, 512*k, 256*k), dilation=dilation)                                       #output: 13 x 13 x 512

        return route_0, route_1, route_2, route_3, input_data
    
    if MODEL_BRANCH_TYPE[1] == "P5" or MODEL_BRANCH_TYPE[1] == "P5n" or MODEL_BRANCH_TYPE[1] == "P5m":
        #Get feature map at medium scale to concatenate later
        route_4 = input_data                                                                            #output: 26 x 16 x 512
        # Downsampling
        input_data = convolutional(input_data, (3, 3, 512*k, 1024*k), downsample=True, activate_type='mish', dilation=dilation)#output: 13 x 13 x 1024      #322 +1(ZeroPad)+5->328

        #CSP block 5
        # First branch
        route = input_data
        route = convolutional(route, (1, 1, 1024*k, 512*k), activate_type='mish', dilation=dilation)                      #output: 13 x 13 x 512       #328 +5->333
        # Second branch
        input_data = convolutional(input_data, (1, 1, 1024*k, 512*k), activate_type='mish', dilation=dilation)                 #output: 13 x 13 x 512
        for _ in range(4):
            input_data = residual_block(input_data, 512*k, 512*k, 512*k, activate_type='mish', dilation=dilation)                                             #333 +[+5x2+1]x4->377
        input_data = convolutional(input_data, (1, 1, 512*k, 512*k), activate_type='mish', dilation=dilation)                  #output: 13 x 13 x 512       #377 +5x2(below+above)->387
        # Concatenation
        input_data = tf.concat([input_data, route], axis=-1)                                            #output: 13 x 13 x 1024                         #388
        input_data = convolutional(input_data, (1, 1, 1024*k, 1024*k), activate_type='mish', dilation=dilation)                #output: 13 x 13 x 1024      #388 +5->393            #72
        route_5 = input_data
        #Compress information of feature map
        input_data = convolutional(input_data, (1, 1, 1024*k, 512*k), dilation=dilation)                                       #output: 13 x 13 x 512       #393 +3->396
        input_data = convolutional(input_data, (3, 3, 512*k, 1024*k), dilation=dilation)                                       #output: 13 x 13 x 1024      #396 +3->399
        input_data = convolutional(input_data, (1, 1, 1024*k, 512*k), dilation=dilation)                                       #output: 13 x 13 x 512       #399 +3->402
        #SPP block
        if not dilation:
            max_pooling_1 = MaxPool2D(pool_size=13, padding='SAME', strides=1)(input_data)                                                                  #403
            max_pooling_2 = MaxPool2D(pool_size=9, padding='SAME', strides=1)(input_data)                                                                   #404
            max_pooling_3 = MaxPool2D(pool_size=5, padding='SAME', strides=1)(input_data)    
        else:
            max_pooling_1 = MaxPool2D(pool_size=25, padding='SAME', strides=1)(input_data)                                                                  #403
            max_pooling_2 = MaxPool2D(pool_size=17, padding='SAME', strides=1)(input_data)                                                                   #404
            max_pooling_3 = MaxPool2D(pool_size=9, padding='SAME', strides=1)(input_data)                                                                 #405
        input_data = tf.concat([max_pooling_1, max_pooling_2, max_pooling_3, input_data], axis=-1)      #output: 13 x 13 x 2048                         #406
        input_data = convolutional(input_data, (1, 1, 2048*k, 512*k), dilation=dilation)                                                                    #406 +3->409
        input_data = convolutional(input_data, (3, 3, 512*k, 1024*k), dilation=dilation)                                                                    #409 +3->412
        input_data = convolutional(input_data, (1, 1, 1024*k, 512*k), dilation=dilation)                                       #output: 13 x 13 x 512       #412 +3->415   #78

        if MODEL_BRANCH_TYPE[1] == "P5":
            return route_0, route_1, route_2, route_3, route_4, input_data
        elif MODEL_BRANCH_TYPE[1] == "P5n":
            return route_2, route_3, route_4, input_data
        elif MODEL_BRANCH_TYPE[1] == "P5m":
            return route_1, route_2, route_3, route_4, input_data


#Implementation of feature texture transfer (FTT model)
def FTT_module(p_lr, p_hr, p_hrx2=None, num_channels=None, dilation=False, num_res=1 if USE_FTT_DEVELOPING_VERSION else 2):       #(p_lr, p_hr, c) = (p5, p4, 512), (p4, p3, 256), (p3, p2, 128)
    #Extract detailed information from LR
    def content_extractor(conv, num_channels, iterations=num_res):
        for _ in range(iterations):
            if not USE_SDCAB_BLOCK_IN_FTT:
                shortcut = conv
                conv = convolutional(conv, (1,1, num_channels, num_channels), dilation=dilation)
                conv = convolutional(conv, (3,3, num_channels, num_channels), dilation=dilation)
                conv = shortcut + conv
            else:
                conv = SDCAB_block(conv, num_channels)
        return conv
    #Extract context information from HR
    def texture_extractor(conv, num_channels, iterations=num_res):
        for _ in range(iterations):
            if not USE_SDCAB_BLOCK_IN_FTT:
                shortcut = conv
                conv = convolutional(conv, (1,1, num_channels, num_channels), dilation=dilation)                    
                conv = convolutional(conv, (3,3, num_channels, num_channels), dilation=dilation)
                conv = shortcut + conv
            else:
                conv = SDCAB_block(conv, num_channels)
        conv = convolutional(conv, (1,1, num_channels, int(num_channels/2)), dilation=dilation)
        return conv
    #start the module
    p_lr = convolutional(p_lr, (1, 1, num_channels, num_channels*4), dilation=dilation) # x4 channels --> num_channels = 4c                     #0 +3->3 (461 +3->464)
    p_lr = content_extractor(p_lr, num_channels*4)                                                                                              #3 +[+3+2+1(add)]x2->15
    p_lr = tf.nn.depth_to_space(p_lr, 2)           #pixel shufflement --> num_channels = c                                                      #16

    # if USE_FTT_DEVELOPING_VERSION:
    #     p_hr = convolutional(p_hr, (1,1, num_channels, num_channels))
    p_hr = tf.concat([p_lr, p_hr], axis=-1) #(104x104x256)                                                                                                    #17
    p_hr = texture_extractor(p_hr, num_channels*2)      #num_channels = c                                                                       #17 +[+3+2+1(add)]x2+3->32
    
    #element-wise sum
    result = p_lr + p_hr                                                                                                                        #33                                    
    
    if USE_FTT_DEVELOPING_VERSION and p_hrx2 != None:
        p_hr = convolutional(result, (1, 1, num_channels, num_channels*2), dilation=dilation) #increase channel x2
        p_hr = content_extractor(p_hr, num_channels*2)  
        p_hr = tf.nn.depth_to_space(p_hr, 2)            #increase resolution x2, channel /4 (208x208x64)
        
        # p_hrx2 = convolutional(p_hrx2, (1,1, int(num_channels/2), int(num_channels/2)))
        p_hrx2 = tf.concat([p_hr, p_hrx2], axis=-1)       # (208x208x128)
        for _ in range(num_res):
            if not USE_SDCAB_BLOCK_IN_FTT:
                shortcut = p_hrx2
                p_hrx2 = convolutional(p_hrx2, (1,1, num_channels, num_channels), dilation=dilation)                    
                p_hrx2 = convolutional(p_hrx2, (3,3, num_channels, num_channels), dilation=dilation)
                p_hrx2 = shortcut + p_hrx2
            else:
                p_hrx2 = SDCAB_block(p_hrx2, num_channels)
        p_hrx2 = convolutional(p_hrx2, (3,3, num_channels, num_channels), downsample=True) #(104x104x128) #14
        # p_hrx2 = convolutional(p_hrx2, (1,1, num_channels, num_channels))
        result = result + p_hrx2
    return result


def spatial_attention_module(conv, kernel_size=1):
    route_mean = tf.math.reduce_mean(conv, axis=-1, keepdims=True)
    route_max = tf.math.reduce_max(conv, axis=-1, keepdims=True)
    concat = tf.concat([route_mean, route_max], axis=-1)                #get concatenation of mean and max pooling
    concat = convolutional(concat, (kernel_size, kernel_size, 2, 1), bn=True, activate=False) #do convolution with bn and without activation
    concat = tf.math.sigmoid(concat)                                    #use sigmoid as activation function
    return conv * concat


def Resblocks(conv, num_channels, dilation=False, num_rb=1):
    for _ in range(num_rb):
        if not USE_SDCAB_BLOCK_IN_FTT:
            shortcut = conv
            conv = convolutional(conv, (1,1, num_channels, num_channels/2), dilation=dilation)
            conv = convolutional(conv, (3,3, num_channels/2, num_channels), dilation=dilation)
            conv = shortcut + conv
        else:
            conv = SDCAB_block(conv)
    return conv

def SDCAB_block(conv, num_channels):
    shortcut = conv
    conv_1 = convolutional(conv,(3,3,num_channels, num_channels), activate=False, bn=False, dilation=True)
    conv_2 = convolutional(conv,(3,3,num_channels, num_channels), activate=False, bn=False, dilation=True, dilation_rate=3)
    conv = convolutional(conv, (3,3, num_channels, num_channels), activate=False, bn=False)
    conv = tf.concat([conv, conv_1, conv_2], axis=-1)
    conv = convolutional(conv, (1,1,num_channels*3, num_channels), activate=True, bn=True)
    conv = conv + shortcut
    return conv


#Add neck layers to CSPDarknet53 and create YOLOv4 model
def YOLOv4_detector(input_layer, NUM_CLASS, dilation=False, dilation_bb=False, Modified_model=False):
    # Create CSPDarknet53 network and 3 backbone features at large, medium and small scale
    if MODEL_BRANCH_TYPE[1] == "P5n":
        route_2, route_3, route_4, conv = CSPDarknet53(input_layer)
    elif MODEL_BRANCH_TYPE[1] == "P5":
        route_0, route_1, route_2, route_3, route_4, conv = CSPDarknet53(input_layer)
    elif MODEL_BRANCH_TYPE[1] == "P4":
        route_0, route_1, route_2, route_3, conv = CSPDarknet53(input_layer)   
    elif MODEL_BRANCH_TYPE[1] == "P3":
        route_0, route_1, route_2, conv = CSPDarknet53(input_layer)   
    elif MODEL_BRANCH_TYPE[1] == "P5m":
        route_1, route_2, route_3, route_4, conv = CSPDarknet53(input_layer, dilation=dilation_bb)
        # backbone_P2, backbone_P3, backbone_P4, backbone_P5 = route_2, route_3, route_4, conv
    
    """ PANet bottom up layers """
    if MODEL_BRANCH_TYPE[1] == "P5" or MODEL_BRANCH_TYPE[1] == "P5n" or MODEL_BRANCH_TYPE[1] == "P5m":
        #upsampling 1
        if not USE_FTT_P4:
            route_5 = conv                                              #output: 13 x 13 x 512      
            # fmap_t5 = route_5
            conv = convolutional(conv, (1, 1, 512*k, 256*k), dilation=dilation)                #output: 13 x 13 x 256      #415 +3x2(above+below)->421
            conv = UpSampling2D()(conv)                                 #output: 26 x 26 x 256      #422                                 
            route_4 = convolutional(route_4, (1, 1, 512*k, 256*k), dilation=dilation)          #output: 26 x 26 x 256
            conv = tf.concat([route_4, conv], axis=-1)                  #output: 26 x 26 x 512      #423
        else:
            conv = FTT_module(conv, route_4, num_channels=512*k, dilation=dilation)    #512 + 256 -> 512                                                                                   #415 +33->448
        fmap_P4 = conv
        #Compress information of feature maps
        conv = convolutional(conv, (1, 1, 512*k, 256*k), dilation=dilation)                                                #423 +3->426 
        conv = convolutional(conv, (3, 3, 256*k, 512*k), dilation=dilation)                                                #426 +3->429
        conv = convolutional(conv, (1, 1, 512*k, 256*k), dilation=dilation)                                                #429 +3->432
        if not USE_FTT_DEVELOPING_VERSION:
            conv = convolutional(conv, (3, 3, 256*k, 512*k), dilation=dilation)                    #output: 26 x 26 x 512      #432 +3->435               
            conv = convolutional(conv, (1, 1, 512*k, 256*k), dilation=dilation)                    #output: 26 x 26 x 256      #435 +3->438
        # fmap_P4 = conv
        
    if MODEL_BRANCH_TYPE[1] == "P5" or MODEL_BRANCH_TYPE[1] == "P4" or MODEL_BRANCH_TYPE[1] == "P5n" or MODEL_BRANCH_TYPE[1] == "P5m":
        #upsampling 2
        if not USE_FTT_P3:
            route_4 = conv                                               #output: 26 x 26 x 256     
            conv = conv = convolutional(conv, (1, 1, 256*k, 128*k), dilation=dilation)          #output: 26 x 26 x 128     #438 +3x2(above+below)->444
            conv = UpSampling2D()(conv)                                  #output: 52 x 52 x 128     #445                 
            route_3 = convolutional(route_3, (1, 1, 256*k, 128*k), dilation=dilation)           #output: 52 x 52 x 128
            # fmap_t3 = conv
            conv = tf.concat([route_3, conv], axis=-1)                   #output: 52 x 52 x 256     #446
        else:
            conv = FTT_module(conv, route_3, num_channels=256*k, dilation=dilation)
        fmap_P3 = conv  
        #Compress information of feature maps
        conv = convolutional(conv, (1, 1, 256*k, 128*k), dilation=dilation)                                                #446 +3->449
        conv = convolutional(conv, (3, 3, 128*k, 256*k), dilation=dilation)                                                #449 +3->452
        conv = convolutional(conv, (1, 1, 256*k, 128*k), dilation=dilation)                                                #452 +3->455
        if not USE_FTT_DEVELOPING_VERSION:
            conv = convolutional(conv, (3, 3, 128*k, 256*k), dilation=dilation)                     #output: 52 x 52 x 256     #455 +3->458
            conv = convolutional(conv, (1, 1, 256*k, 128*k), dilation=dilation)                     #output: 52 x 52 x 128     #458 +3->461
        # fmap_P3 = conv
        # fmap_t3 = conv

    """ Additional upsampling: to resolution P2 """
    if  MODEL_BRANCH_TYPE[1] == "P5m":
        #upsampling 3
        if not USE_FTT_P2:
            route_3 = conv                                                #output: 52 x 52 x 128
            conv = convolutional(conv, (1, 1, 128*k, 64*k), dilation=dilation)                   #output: 52 x 52 x 64
            conv = UpSampling2D()(conv)                                   #output: 104 x 104 x 64                                       
            route_2 = convolutional(route_2, (1, 1, 128*k, 64*k), dilation=dilation)             #output: 104 x 104 x 64
            conv = tf.concat([route_2, conv], axis=-1)                    #output: 104 x 104 x 128
        else:
            conv = FTT_module(conv, route_2,route_1, 128*k, dilation=dilation)                                                   #461 +33->494        #if 3-FTT: 544
        fmap_P2 = conv    
        #Compress information of feature maps
        conv = convolutional(conv, (1, 1, 128*k, 64*k), dilation=dilation) 
        conv = convolutional(conv, (3, 3, 64*k, 128*k), dilation=dilation)
        conv = convolutional(conv, (1, 1, 128*k, 64*k), dilation=dilation)
        if not USE_FTT_DEVELOPING_VERSION:    
            conv = convolutional(conv, (3, 3, 64*k, 128*k), dilation=dilation)
            conv = convolutional(conv, (1, 1, 128*k, 64*k), dilation=dilation)                       #output: 104 x 104 x 64
        # fmap_P2 = conv


    """ Additional upsampling: 3 times to original image size """
    if MODEL_BRANCH_TYPE[1] == "P5" or MODEL_BRANCH_TYPE[1] == "P4" or MODEL_BRANCH_TYPE[1] == "P3":
        #upsampling 3
        # route_3 = conv                                                                #output: 52 x 52 x 128
        conv = convolutional(conv, (1, 1, 128, 64))                                     #output: 52 x 52 x 64
        conv = UpSampling2D()(conv)                                                     #output: 104 x 104 x 64                                       
        route_2 = convolutional(route_2, (1, 1, 128, 64))                               #output: 104 x 104 x 64
        conv = tf.concat([route_2, conv], axis=-1)                                      #output: 104 x 104 x 128
        #Compress information of feature maps
        conv = convolutional(conv, (1, 1, 128, 64))
        conv = convolutional(conv, (3, 3, 64, 128))
        conv = convolutional(conv, (1, 1, 128, 64))
        conv = convolutional(conv, (3, 3, 64, 128))
        conv = convolutional(conv, (1, 1, 128, 64))                                     #output: 104 x 104 x 64

        #upsampling 4
        route_2 = conv                                                                  #output: 104 x 104 x 64
        conv = convolutional(conv, (1, 1, 64, 32))                                      #output: 104 x 104 x 32
        conv = UpSampling2D()(conv)                                                     #output: 208 x 208 x 32                                       
        route_1 = convolutional(route_1, (1, 1, 64, 32))                                #output: 208 x 208 x 32         
        conv = tf.concat([route_1, conv], axis=-1)                                      #output: 208 x 208 x 64     
        #Compress information of feature maps
        conv = convolutional(conv, (1, 1, 64, 32))
        conv = convolutional(conv, (3, 3, 32, 64))
        conv = convolutional(conv, (1, 1, 64, 32))
        conv = convolutional(conv, (3, 3, 32, 64))
        conv = convolutional(conv, (1, 1, 64, 32))                                      #output: 208 x 208 x 32  

        #upsampling 5
        route_1 = conv                                                                  #output: 208 x 208 x 32
        conv = convolutional(conv, (1, 1, 32, 16))                                      #output: 208 x 208 x 16
        conv = UpSampling2D()(conv)                                                     #output: 416 x 416 x 16 
        route_0 = convolutional(route_0, (1, 1, 32, 16))                                #output: 416 x 416 x 16         
        conv = tf.concat([route_0, conv], axis=-1)                                      #output: 416 x 416 x 32   
        #Compress information of feature maps
        conv = convolutional(conv, (1, 1, 32, 16))
        conv = convolutional(conv, (3, 3, 16, 32))
        conv = convolutional(conv, (1, 1, 32, 16))
        conv = convolutional(conv, (3, 3, 16, 32))
        conv = convolutional(conv, (1, 1, 32, 16))                                      #output: 416 x 416 x 16


    if MODEL_BRANCH_TYPE[0] == "P3n":
        route_3 = conv
        conv = convolutional(conv, (3, 3, 128*k, 256*k))
        # conv_sbbox = convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
        conv_sbbox = convolutional(conv, (1, 1, 256*k, (ANCHORS_PER_GRID_CELL_SMALL if USE_5_ANCHORS_SMALL_SCALE else ANCHORS_PER_GRID_CELL) * (NUM_CLASS + 5)), activate=False, bn=False)

        conv = convolutional(route_3, (3, 3, 128*k, 256*k), downsample=True)
        conv = tf.concat([conv, route_4], axis=-1)

        conv = convolutional(conv, (1, 1, 512*k, 256*k))
        conv = convolutional(conv, (3, 3, 256*k, 512*k))
        conv = convolutional(conv, (1, 1, 512*k, 256*k))
        conv = convolutional(conv, (3, 3, 256*k, 512*k))
        conv = convolutional(conv, (1, 1, 512*k, 256*k))

        route_4 = conv
        conv = convolutional(conv, (3, 3, 256*k, 512*k))
        conv_mbbox = convolutional(conv, (1, 1, 512*k, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        conv = convolutional(route_4, (3, 3, 256*k, 512*k), downsample=True)
        conv = tf.concat([conv, route_5], axis=-1)

        conv = convolutional(conv, (1, 1, 1024*k, 512*k))
        conv = convolutional(conv, (3, 3, 512*k, 1024*k))
        conv = convolutional(conv, (1, 1, 1024*k, 512*k))
        conv = convolutional(conv, (3, 3, 512*k, 1024*k))
        conv = convolutional(conv, (1, 1, 1024*k, 512*k))

        conv = convolutional(conv, (3, 3, 512*k, 1024*k))
        conv_lbbox = convolutional(conv, (1, 1, 1024*k, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        return [conv_sbbox, conv_mbbox, conv_lbbox]


    if MODEL_BRANCH_TYPE[0] == "P0":
        """PANet top down layers: P0 - P2"""
        #Small bbox convolutional output
        route_0 = conv
        conv = convolutional(conv, (3, 3, 16, 32))                                #output: 416 x 416 x 32 
        conv_sbbox = convolutional(conv, (1, 1, 32, 3 * (NUM_CLASS + 5)), activate=False, bn=False)     #output: 416 x 416 x 3*(NUM_CLASS+5)
        #Downsampling 1
        conv = convolutional(route_0, (3, 3, 16, 32), downsample=True)                     #output: 208 x 208 x 32
        conv = tf.concat([conv, route_1], axis=-1)                                      #output: 208 x 208 x 64
        #Compress information of feature maps
        conv = convolutional(conv, (1, 1, 64, 32))
        conv = convolutional(conv, (3, 3, 32, 64))
        conv = convolutional(conv, (1, 1, 64, 32))
        conv = convolutional(conv, (3, 3, 32, 64))
        conv = convolutional(conv, (1, 1, 64, 32))                                      #output: 208 x 208 x 32
        #Medium bbox convolutional output
        route_1 = conv
        conv = convolutional(conv, (3, 3, 32, 64))                                #output: 208 x 208 x 64
        conv_mbbox = convolutional(conv, (1, 1, 64, 3 * (NUM_CLASS + 5)), activate=False, bn=False)     #output: 208 x 208 x 3*(NUM_CLASS+5)
        #Downsampling 2
        conv = convolutional(route_1, (3, 3, 32, 64), downsample=True)                     #output: 104 x 104 x 64
        conv = tf.concat([conv, route_2], axis=-1)                                      #output: 104 x 104 x 128
        #Compress information of feature maps
        conv = convolutional(conv, (1, 1, 128, 64))
        conv = convolutional(conv, (3, 3, 64, 128))
        conv = convolutional(conv, (1, 1, 128, 64))
        conv = convolutional(conv, (3, 3, 64, 128))
        conv = convolutional(conv, (1, 1, 128, 64))                                     #output: 104 x 104 x 64
        #Small bbox convolutional output
        conv = convolutional(conv, (3, 3, 64, 128))                                     #output: 104 x 104 x 128
        conv_lbbox = convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)        #output: 104 x 104 x 3*(NUM_CLASS+5)
        
        return [conv_sbbox, conv_mbbox, conv_lbbox]

    elif MODEL_BRANCH_TYPE[0] == "P(-1)":
        #upsampling 6
        route_0 = conv                                                                  #output: 208 x 208 x 32
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
        route_m1 = conv
        conv = convolutional(conv, (3, 3, 8, 16))                                #output: 416 x 416 x 32 
        conv_sbbox = convolutional(conv, (1, 1, 16, 3 * (NUM_CLASS + 5)), activate=False, bn=False)     #output: 416 x 416 x 3*(NUM_CLASS+5)
        #Downsampling 1
        conv = convolutional(route_m1, (3, 3, 8, 16), downsample=True)                     #output: 208 x 208 x 32
        conv = tf.concat([conv, route_0], axis=-1)                                      #output: 208 x 208 x 64
        #Compress information of feature maps
        conv = convolutional(conv, (1, 1, 32, 16))
        conv = convolutional(conv, (3, 3, 16, 32))
        conv = convolutional(conv, (1, 1, 32, 16))
        conv = convolutional(conv, (3, 3, 16, 32))
        conv = convolutional(conv, (1, 1, 32, 16))                                      #output: 208 x 208 x 32
        #Medium bbox convolutional output
        route_0 = conv
        conv = convolutional(conv, (3, 3, 16, 32))                                #output: 208 x 208 x 64
        conv_mbbox = convolutional(conv, (1, 1, 32, 3 * (NUM_CLASS + 5)), activate=False, bn=False)     #output: 208 x 208 x 3*(NUM_CLASS+5)
        #Downsampling 2
        conv = convolutional(route_0, (3, 3, 16, 32), downsample=True)                     #output: 104 x 104 x 64
        conv = tf.concat([conv, route_1], axis=-1)                                      #output: 104 x 104 x 128
        #Compress information of feature maps
        conv = convolutional(conv, (1, 1, 64, 32))
        conv = convolutional(conv, (3, 3, 32, 64))
        conv = convolutional(conv, (1, 1, 64, 32))
        conv = convolutional(conv, (3, 3, 32, 64))
        conv = convolutional(conv, (1, 1, 64, 32))                                     #output: 104 x 104 x 64
        #Small bbox convolutional output
        conv = convolutional(conv, (3, 3, 32, 64))                                     #output: 104 x 104 x 128
        conv_lbbox = convolutional(conv, (1, 1, 64, 3 * (NUM_CLASS + 5)), activate=False, bn=False)     #output: 104 x 104 x 3*(NUM_CLASS+5)
        
        return [conv_sbbox, conv_mbbox, conv_lbbox]

    elif MODEL_BRANCH_TYPE[0] == "P2":
        if Modified_model:
            route_2 = conv
            conv = convolutional(conv, (3,3,64,128), dilation=dilation)
            # conv = convolutional(conv, (3, 3, 128, 128), dilation=dilation)
            # conv = spatial_attention_module(conv, kernel_size=5)
            conv = Resblocks(conv, 128, num_rb=1)
            conv = convolutional(conv, (3,3,128,256), dilation=dilation)
            conv = spatial_attention_module(conv, kernel_size=5)
            # conv = Resblocks(conv, 256, num_rb=2)
            fmap_P2 = conv
            conv_sbbox = convolutional(conv, (1, 1, 256, (ANCHORS_PER_GRID_CELL_SMALL if USE_5_ANCHORS_SMALL_SCALE else ANCHORS_PER_GRID_CELL) * (NUM_CLASS + 5)), activate=False, bn=False, dilation=dilation)
            
            conv = convolutional(route_2, (3, 3, 64, 128), downsample=True, dilation=dilation)
            conv = tf.concat([conv, route_3], axis=-1)

            conv = convolutional(conv, (1, 1, 256, 128), dilation=dilation)
            conv = convolutional(conv, (3, 3, 128, 256), dilation=dilation)
            conv = convolutional(conv, (1, 1, 256, 128), dilation=dilation)
            if not USE_FTT_DEVELOPING_VERSION:
                conv = convolutional(conv, (3, 3, 128, 256), dilation=dilation)
                conv = convolutional(conv, (1, 1, 256, 128), dilation=dilation)

            route_3 = conv
            # conv = route_3
            conv = convolutional(conv, (3, 3, 128, 256))
            # conv = spatial_attention_module(conv, kernel_size=5)
            conv = Resblocks(conv, 256, num_rb=1)
            conv = convolutional(conv, (3, 3, 256, 512))
            conv = spatial_attention_module(conv, kernel_size=5)
            # conv = Resblocks(conv, 512, num_rb=2)
            fmap_P3 = conv
            conv_mbbox = convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False, dilation=dilation)

            conv = convolutional(route_3, (3, 3, 128, 256), downsample=True, dilation=dilation)
            conv = tf.concat([conv, route_4], axis=-1)

            conv = convolutional(conv, (1, 1, 512, 256), dilation=dilation)
            conv = convolutional(conv, (3, 3, 256, 512), dilation=dilation)
            conv = convolutional(conv, (1, 1, 512, 256), dilation=dilation)
            if not USE_FTT_DEVELOPING_VERSION:
                conv = convolutional(conv, (3, 3, 256, 512), dilation=dilation)
                conv = convolutional(conv, (1, 1, 512, 256), dilation=dilation)

            route_4 = conv
            conv = convolutional(conv, (3, 3, 256, 512))
            # conv = spatial_attention_module(conv, kernel_size=5)
            conv = Resblocks(conv, 512, num_rb=1)
            conv = convolutional(conv, (3, 3, 512, 1024))
            conv = spatial_attention_module(conv, kernel_size=5)
            # conv = Resblocks(conv, 1024, num_rb=2)
            fmap_P4 = conv
            conv_lbbox = convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False, dilation=dilation)

            if USE_SUPERVISION:
                return [conv_sbbox, conv_mbbox, conv_lbbox], [fmap_P2, fmap_P3, fmap_P4]
            else:
                return [conv_sbbox, conv_mbbox, conv_lbbox]

        else:
            route_2 = conv
            fmap_P2 = conv
            if USE_SUPERVISION and USE_ADAPTATION_LAYER:
                fmap_P2 = convolutional(conv, (1, 1, 64*k, 128), dilation=dilation)   #adaptation layer
            conv = convolutional(conv, (3, 3, 64*k, 128*k), dilation=dilation)
            conv_sbbox = convolutional(conv, (1, 1, 128*k, (ANCHORS_PER_GRID_CELL_SMALL if USE_5_ANCHORS_SMALL_SCALE else ANCHORS_PER_GRID_CELL) * (NUM_CLASS + 5)), activate=False, bn=False, dilation=dilation)
            
            conv = convolutional(route_2, (3, 3, 64*k, 128*k), downsample=True, dilation=dilation)
            conv = tf.concat([conv, route_3], axis=-1)

            conv = convolutional(conv, (1, 1, 256*k, 128*k), dilation=dilation)
            conv = convolutional(conv, (3, 3, 128*k, 256*k), dilation=dilation)
            conv = convolutional(conv, (1, 1, 256*k, 128*k), dilation=dilation)
            if not USE_FTT_DEVELOPING_VERSION:
                conv = convolutional(conv, (3, 3, 128*k, 256*k), dilation=dilation)
                conv = convolutional(conv, (1, 1, 256*k, 128*k), dilation=dilation)

            route_3 = conv
            fmap_P3 = conv
            if USE_SUPERVISION and USE_ADAPTATION_LAYER:
                fmap_P3 = convolutional(conv, (1, 1, 128*k, 256))     #adaptation layer
            conv = convolutional(conv, (3, 3, 128*k, 256*k))
            conv_mbbox = convolutional(conv, (1, 1, 256*k, 3 * (NUM_CLASS + 5)), activate=False, bn=False, dilation=dilation)

            conv = convolutional(route_3, (3, 3, 128*k, 256*k), downsample=True, dilation=dilation)
            conv = tf.concat([conv, route_4], axis=-1)

            conv = convolutional(conv, (1, 1, 512*k, 256*k), dilation=dilation)
            conv = convolutional(conv, (3, 3, 256*k, 512*k), dilation=dilation)
            conv = convolutional(conv, (1, 1, 512*k, 256*k), dilation=dilation)
            if not USE_FTT_DEVELOPING_VERSION:
                conv = convolutional(conv, (3, 3, 256*k, 512*k), dilation=dilation)
                conv = convolutional(conv, (1, 1, 512*k, 256*k), dilation=dilation)

            fmap_P4 = conv
            if USE_SUPERVISION and USE_ADAPTATION_LAYER:
                fmap_P4 = convolutional(conv, (1, 1, 256*k, 512))     #adaptation layer
            conv = convolutional(conv, (3, 3, 256*k, 512*k))
            conv_lbbox = convolutional(conv, (1, 1, 512*k, 3 * (NUM_CLASS + 5)), activate=False, bn=False, dilation=dilation)

            if USE_SUPERVISION:
                return [conv_sbbox, conv_mbbox, conv_lbbox], [fmap_P2, fmap_P3, fmap_P4]#, fmap_t3, fmap_t4,fmap_t5, backbone_P2, backbone_P3, backbone_P4, backbone_P5]
            else:
                return [conv_sbbox, conv_mbbox, conv_lbbox]



#New CSPDarket-52: remove the first downsampling layer and reduce all the number of channels after that by 2
def CSPDarknet52_YOLOv4_detector(input_data, NUM_CLASS, dilation=False):
    input_data = convolutional(input_data, (3, 3, 3, 32), activate_type="mish", dilation=dilation)      
    # input_data = convolutional(input_data, (3, 3, 32, 32), activate_type="mish", dilation=dilation)
    #CSP block 1
    # First branch
    route = input_data
    route = convolutional(route, (1, 1, 32, 32), activate_type='mish', dilation=dilation)                         
    # Second branch
    input_data = convolutional(input_data, (1, 1, 32, 32), activate_type='mish', dilation=dilation)                   
    for _ in range(1):
        input_data = residual_block(input_data,  32,  16, 32, activate_type='mish', dilation=dilation)                      
    input_data = convolutional(input_data, (1, 1, 32, 32), activate_type='mish', dilation=dilation)                    
    # Concatenation
    input_data = tf.concat([input_data, route], axis=-1)                                            
    input_data = convolutional(input_data, (1, 1, 64, 32), activate_type='mish', dilation=dilation)                  
    # Downsampling
    route_0 = input_data
    input_data = convolutional(input_data, (3, 3, 32, 64), downsample=True, activate_type='mish', dilation=dilation) 
    
    #CSP block 2
    # First branch
    route = input_data
    route = convolutional(route, (1, 1, 64, 32), activate_type='mish', dilation=dilation)                       
    # Second branch
    input_data = convolutional(input_data, (1, 1, 64, 32), activate_type='mish', dilation=dilation)                   
    for _ in range(2):
        input_data = residual_block(input_data, 32, 32, 32, activate_type='mish', dilation=dilation)                                              
    input_data = convolutional(input_data, (1, 1, 32, 32), activate_type='mish', dilation=dilation)                    
    # Concatenation
    input_data = tf.concat([input_data, route], axis=-1)                                            
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type='mish', dilation=dilation) 
    # Downsampling
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 64, 128), downsample=True, activate_type='mish', dilation=dilation) 

    #CSP block 3
    # First branch
    route = input_data
    route = convolutional(route, (1, 1, 128, 64), activate_type='mish', dilation=dilation)                   
    # Second branch
    input_data = convolutional(input_data, (1, 1, 128, 64), activate_type='mish', dilation=dilation)                 
    for _ in range(8):
        input_data = residual_block(input_data, 64, 64, 64, activate_type='mish', dilation=dilation)                                           
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type='mish', dilation=dilation)                 
    # Concatenation
    input_data = tf.concat([input_data, route], axis=-1)                                            
    input_data = convolutional(input_data, (1, 1, 128, 128), activate_type='mish', dilation=dilation)                  
    # Downsampling
    route_2 = input_data
    input_data = convolutional(input_data, (3, 3, 128, 256), downsample=True, activate_type='mish', dilation=dilation) 
    
    #CSP block 4
    # First branch
    route = input_data
    route = convolutional(route, (1, 1, 256, 128), activate_type='mish', dilation=dilation)                       
    # Second branch
    input_data = convolutional(input_data, (1, 1, 256, 128), activate_type='mish', dilation=dilation)                  
    for _ in range(8):
        input_data = residual_block(input_data, 128, 128, 128, activate_type='mish', dilation=dilation)                                             
    input_data = convolutional(input_data, (1, 1, 128, 128), activate_type='mish', dilation=dilation)                  
    # Concatenation
    input_data = tf.concat([input_data, route], axis=-1)                                            
    input_data = convolutional(input_data, (1, 1, 256, 256), activate_type='mish', dilation=dilation)                                                                                
    # Downsampling
    route_3 = input_data
    input_data = convolutional(input_data, (3, 3, 256, 512), downsample=True, activate_type='mish', dilation=dilation)

    #CSP block 5
    # First branch
    route = input_data
    route = convolutional(route, (1, 1, 512, 256), activate_type='mish', dilation=dilation)  
    # Second branch
    input_data = convolutional(input_data, (1, 1, 512, 256), activate_type='mish', dilation=dilation)      
    for _ in range(4):
        input_data = residual_block(input_data, 256, 256, 256, activate_type='mish', dilation=dilation)             
    input_data = convolutional(input_data, (1, 1, 256, 256), activate_type='mish', dilation=dilation)   
    # Concatenation
    input_data = tf.concat([input_data, route], axis=-1)                                               
    input_data = convolutional(input_data, (1, 1, 512, 512), activate_type='mish', dilation=dilation) 
    
    #SPP block
    input_data = convolutional(input_data, (1, 1, 512, 256), dilation=dilation)                
    input_data = convolutional(input_data, (3, 3, 256, 512), dilation=dilation)             
    input_data = convolutional(input_data, (1, 1, 512, 256), dilation=dilation)                 
    max_pooling_1 = MaxPool2D(pool_size=13, padding='SAME', strides=1)(input_data)                                              
    max_pooling_2 = MaxPool2D(pool_size=9, padding='SAME', strides=1)(input_data)                                                    
    max_pooling_3 = MaxPool2D(pool_size=5, padding='SAME', strides=1)(input_data)                                   
    input_data = tf.concat([max_pooling_1, max_pooling_2, max_pooling_3, input_data], axis=-1)           
    input_data = convolutional(input_data, (1, 1, 1024, 256), dilation=dilation)                                  
    input_data = convolutional(input_data, (3, 3, 256, 512), dilation=dilation)                      
    input_data = convolutional(input_data, (1, 1, 512, 256), dilation=dilation)                              
    conv = input_data

    """ PANet bottom up layers """
    route_4 = conv
    fmap_P4 = conv                                      #channels: 256                                             
    #upsampling 1
    conv = convolutional(conv, (1, 1, 256, 128))               
    conv = UpSampling2D()(conv)                                                               
    route_3 = convolutional(route_3, (1, 1, 256, 128))        
    conv = tf.concat([route_3, conv], axis=-1)        
    #Compress information of feature maps
    conv = convolutional(conv, (1, 1, 256, 128))              
    conv = convolutional(conv, (3, 3, 128, 256))                         
    conv = convolutional(conv, (1, 1, 256, 128))              
    if not USE_FTT_DEVELOPING_VERSION:
        conv = convolutional(conv, (3, 3, 128, 256))             
        conv = convolutional(conv, (1, 1, 256, 128))    
    route_3 = conv    
    fmap_P3 = conv                                      #channels: 128
    #upsampling 2
    if not USE_FTT_P2:                                          
        conv = conv = convolutional(conv, (1, 1, 128, 64))     
        conv = UpSampling2D()(conv)                                      
        route_2 = convolutional(route_2, (1, 1, 128, 64))    
        conv = tf.concat([route_2, conv], axis=-1)  
    else:
        conv = FTT_module(conv, route_2, route_1 if USE_FTT_DEVELOPING_VERSION else None, num_channels=128)
    fmap_P2 = conv                                      #channels: 128
    #Compress information of feature maps
    conv = convolutional(conv, (1, 1, 128, 64))                                  
    conv = convolutional(conv, (3, 3, 64, 128))                                     
    conv = convolutional(conv, (1, 1, 128, 64))                                        
    if not USE_FTT_DEVELOPING_VERSION:
        conv = convolutional(conv, (3, 3, 64, 128))                 
        conv = convolutional(conv, (1, 1, 128, 64))                
    route_2 = conv
    fmap_P2_h = conv
    if USE_SUPERVISION and USE_ADAPTATION_LAYER:
        fmap_P2_h = convolutional(conv, (1, 1, 64, 128))

    conv = convolutional(conv, (3, 3, 64, 128))
    conv_sbbox = convolutional(conv, (1, 1, 128, (ANCHORS_PER_GRID_CELL_SMALL if USE_5_ANCHORS_SMALL_SCALE else ANCHORS_PER_GRID_CELL) * (NUM_CLASS + 5)), activate=False, bn=False)
    #Downsampling 1
    conv = convolutional(route_2, (3, 3, 64, 128), downsample=True)
    conv = tf.concat([conv, route_3], axis=-1)
    #Compress information of feature maps
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    if not USE_FTT_DEVELOPING_VERSION:
        conv = convolutional(conv, (3, 3, 128, 256))
        conv = convolutional(conv, (1, 1, 256, 128))
    route_3 = conv
    fmap_P3_h = conv
    if USE_SUPERVISION and USE_ADAPTATION_LAYER:
        fmap_P3_h = convolutional(conv, (1, 1, 128, 256))

    conv = convolutional(conv, (3, 3, 128, 256))
    conv_mbbox = convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
    #Downsampling 2
    conv = convolutional(route_3, (3, 3, 128, 256), downsample=True)
    conv = tf.concat([conv, route_4], axis=-1)
    #Compress information of feature maps
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    if not USE_FTT_DEVELOPING_VERSION:
        conv = convolutional(conv, (3, 3, 256, 512))
        conv = convolutional(conv, (1, 1, 512, 256))
    fmap_P4_h = conv
    if USE_SUPERVISION and USE_ADAPTATION_LAYER:
        fmap_P4_h = convolutional(conv, (1, 1, 256, 512))

    conv = convolutional(conv, (3, 3, 256, 512))
    conv_lbbox = convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    if USE_SUPERVISION:
        # return [conv_sbbox, conv_mbbox, conv_lbbox], [fmap_P2, fmap_P3, fmap_P4]
        return [conv_sbbox, conv_mbbox, conv_lbbox], [fmap_P2_h, fmap_P3_h, fmap_P4_h]
    else:
        return [conv_sbbox, conv_mbbox, conv_lbbox]


#Define function used to change the output tensor to the information of (bbox, confidence, class)
# i represents for the grid scales: (0,1,2) <--> (large, medium, small)
def decode(conv_output, NUM_CLASS, i=0, YOLO_SCALE_OFFSET=YOLO_SCALE_OFFSET, YOLO_ANCHORS=YOLO_ANCHORS, TEACHER_MODE=True):
    conv_shape       = tf.shape(conv_output)                # shape [batch_size, output_size, output_size, 255]           
    batch_size       = conv_shape[0]
    output_size_h      = conv_shape[1]
    output_size_w      = conv_shape[2]
    #Change the output_shape of each scale into [batch_size, output_size_h, output_size_w, 3, 85]
    conv_output = tf.reshape(conv_output, (batch_size, output_size_h, output_size_w, ANCHORS_PER_GRID_CELL_SMALL if (i==0 and USE_5_ANCHORS_SMALL_SCALE) else ANCHORS_PER_GRID_CELL, 5 + NUM_CLASS))
    
    #Split the final dimension into 4 information (offset xy, offset wh, confidence, class probabilities)
    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS), axis=-1)
    #shape [batch_size, output_size_h, output_size_w, 3, ...]

    # Create the matrix of grid cell indexes
    x_gridcell, y_gridcell = tf.meshgrid(tf.range(output_size_w), tf.range(output_size_h))  #create 2 matrices of shape [output_size_h, output_size_w]
    xy_gridcell = tf.stack([x_gridcell, y_gridcell], axis=-1)                           #Stack at final axis to create shape [output_size_h, output_size_w, 2]
    xy_gridcell = tf.expand_dims(tf.expand_dims(xy_gridcell, axis=2), axis=0)           #Prepare shape [1, output_size_h, output_size_w, 1, 2]
    xy_gridcell = tf.tile(xy_gridcell, [batch_size, 1, 1, ANCHORS_PER_GRID_CELL_SMALL if (i==0 and USE_5_ANCHORS_SMALL_SCALE) else ANCHORS_PER_GRID_CELL, 1])                        #Create indexes matrix of shape [batch_size, output_size_h, output_size_w, 3, 2]
    xy_gridcell = tf.cast(xy_gridcell, tf.float32)

    # Predicted boxes coordinates
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_gridcell) * YOLO_SCALE_OFFSET[i]      #new: 2 * sigmoid - 0.5 (make pre-predict_xy smaller)
    if TEACHER_MODE:
        pred_wh = (tf.exp(conv_raw_dwdh) * YOLO_ANCHORS[i])
    else:
        pred_wh = ((2 * tf.sigmoid(conv_raw_dwdh))**2 * YOLO_ANCHORS[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    #Predicted box confidence scores
    pred_conf = tf.sigmoid(conv_raw_conf)
    #Predicted box class probabilities 
    pred_prob = tf.sigmoid(conv_raw_prob)

    #Prediction of shape [batch_size, output_size_h, output_size_w, 3, 85]
    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def YOLOv4_Model(input_channel=3, training=False, CLASSES_PATH=YOLO_COCO_CLASS_PATH, dilation=False, dilation_bb=False, Modified_model=False):
    #Read coco class names file
    class_names = {}
    with open(CLASSES_PATH, 'r') as data:
        for ID, name in enumerate(data):
            class_names[ID] = name.strip('\n')
    NUM_CLASS = len(class_names)
    #Create input layer
    input_layer = Input([YOLO_INPUT_SIZE[1] if TEST_FLOPS else None, YOLO_INPUT_SIZE[0] if TEST_FLOPS else None, input_channel])
    if USE_SUPERVISION:
        if USE_NEW_BACKBONE: 
            conv_tensors, student_fmaps_mid = CSPDarknet52_YOLOv4_detector(input_layer, NUM_CLASS, dilation=dilation)
        else:
            conv_tensors, student_fmaps_mid = YOLOv4_detector(input_layer, NUM_CLASS, dilation=dilation, dilation_bb=dilation_bb, Modified_model=Modified_model)
    else:
        if USE_NEW_BACKBONE:
            conv_tensors = CSPDarknet52_YOLOv4_detector(input_layer, NUM_CLASS, dilation=dilation)
        else:
            conv_tensors = YOLOv4_detector(input_layer, NUM_CLASS, dilation=dilation, dilation_bb=dilation_bb, Modified_model=Modified_model)

    output_tensors = []
    student_fmaps = []
    for i, conv_tensor in enumerate(conv_tensors):                              #small bboxes -> medium -> large
        pred_tensor = decode(conv_tensor, NUM_CLASS, i)
        if training:                                                           
            output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)                                      #shape [3 or 6, batch_size, output_size, output_size, 3, 85]
        student_fmaps.append(conv_tensor)
    if training and USE_SUPERVISION:
        for temp in student_fmaps_mid:
            output_tensors.append(temp)
        for temp in student_fmaps:
            output_tensors.append(temp)
    YOLOv4_model = tf.keras.Model(input_layer, output_tensors)
    return YOLOv4_model





def cspdarknet53_teacher(input_data):
    input_data = convolutional(input_data, (3, 3,  3,  32), activate_type="mish")
    input_data = convolutional(input_data, (3, 3, 32,  64), downsample=True, activate_type="mish")                #2
    #CSP1
    route = input_data
    route = convolutional(route, (1, 1, 64, 64), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    for i in range(1):
        input_data = residual_block(input_data,  64,  32, 64, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)
    input_data = convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    input_data = convolutional(input_data, (3, 3, 64, 128), downsample=True, activate_type="mish")
    #CSP2
    route = input_data
    route = convolutional(route, (1, 1, 128, 64), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    for i in range(2):
        input_data = residual_block(input_data, 64,  64, 64, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)
    input_data = convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = convolutional(input_data, (3, 3, 128, 256), downsample=True, activate_type="mish")
    #CSP3
    route = input_data
    route = convolutional(route, (1, 1, 256, 128), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 256, 128), activate_type="mish")
    for i in range(8):
        input_data = residual_block(input_data, 128, 128, 128, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)
    input_data = convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 256, 512), downsample=True, activate_type="mish")
    #CSP4
    route = input_data
    route = convolutional(route, (1, 1, 512, 256), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 512, 256), activate_type="mish")
    for i in range(8):
        input_data = residual_block(input_data, 256, 256, 256, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)
    input_data = convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    route_2 = input_data
    input_data = convolutional(input_data, (3, 3, 512, 1024), downsample=True, activate_type="mish")
    #CSP5
    route = input_data
    route = convolutional(route, (1, 1, 1024, 512), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 1024, 512), activate_type="mish")
    for i in range(4):
        input_data = residual_block(input_data, 512, 512, 512, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = convolutional(input_data, (1, 1, 1024, 1024), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 1024, 512))
    input_data = convolutional(input_data, (3, 3, 512, 1024))
    input_data = convolutional(input_data, (1, 1, 1024, 512))

    max_pooling_1 = tf.keras.layers.MaxPool2D(pool_size=13, padding='SAME', strides=1)(input_data)
    max_pooling_2 = tf.keras.layers.MaxPool2D(pool_size=9, padding='SAME', strides=1)(input_data)
    max_pooling_3 = tf.keras.layers.MaxPool2D(pool_size=5, padding='SAME', strides=1)(input_data)
    input_data = tf.concat([max_pooling_1, max_pooling_2, max_pooling_3, input_data], axis=-1)

    input_data = convolutional(input_data, (1, 1, 2048, 512))
    input_data = convolutional(input_data, (3, 3, 512, 1024))
    input_data = convolutional(input_data, (1, 1, 1024, 512))

    return route_1, route_2, input_data


def create_YOLOv4_backbone(input_channel=3, dilation_bb=False, CLASSES_PATH=None):
    class_names = {}
    with open(CLASSES_PATH, 'r') as data:
        for ID, name in enumerate(data):
            class_names[ID] = name.strip('\n')
    NUM_CLASS = len(class_names)

    input_layer = Input([None, None, input_channel])
    route_3, route_4, conv = cspdarknet53_teacher(input_layer)            #26x26x512                      #415
    # fmap_bb_P3 = route_3
    # fmap_bb_P4 = route_4
    """ PANet bottom up layers """
    fmap_bb_P5 = conv
    route_5 = conv
    conv = convolutional(conv, (1, 1, 512, 256))             #output: 26 x 26 x 256
    conv = UpSampling2D()(conv)                                                 #output: 52 x 52 x 256                                       
    route_4 = convolutional(route_4, (1, 1, 512, 256))       #output: 52 x 52 x 256
    conv = tf.concat([route_4, conv], axis=-1)                                  #output: 52 x 52 x 512          #415 +8->423
                                        #415 +33->448
        
    #Compress information of feature maps
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))                 #output: 52 x 52 x 512
    conv = convolutional(conv, (1, 1, 512, 256))                 #output: 52 x 52 x 256          #423 +15->438
    fmap_bb_P4 = conv
    
    
    route_4 = conv                                                              #output: 52 x 52 x 256
    conv = conv = convolutional(conv, (1, 1, 256, 128))      #output: 52 x 52 x 128
    conv = UpSampling2D()(conv)                                                 #output: 104 x 104 x 128                                                 
    route_3 = convolutional(route_3, (1, 1, 256, 128))       #output: 104 x 104 x 128
    conv = tf.concat([route_3, conv], axis=-1)                                  #output: 104 x 104 x 256        #438 +8->446
                                            #438 +33->471

    #Compress information of feature maps
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))                  #output: 104 x 104 x 256
    conv = convolutional(conv, (1, 1, 256, 128))                  #output: 104 x 104 x 128       #78 + 14        #446 +15->461
    fmap_bb_P3 = conv


    route_3 = conv
    conv = convolutional(conv, (3, 3, 128, 256))
    # fmap_bb_P3 = conv
    conv_sbbox = convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
    conv_sbbox_decoded = decode(conv_sbbox, NUM_CLASS, i=0)

    conv = convolutional(route_3, (3, 3, 128, 256), downsample=True)
    conv = tf.concat([conv, route_4], axis=-1)

    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))

    fmap_bb_P4= conv
    route_4 = conv
    conv = convolutional(conv, (3, 3, 256, 512))
    # fmap_bb_P4= conv
    conv_mbbox = convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
    conv_mbbox_decoded = decode(conv_mbbox, NUM_CLASS, i=1)


    conv = convolutional(route_4, (3, 3, 256, 512), downsample=True)
    conv = tf.concat([conv, route_5], axis=-1)

    conv = convolutional(conv, (1, 1, 1024, 512))
    conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))
    conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))
    
    fmap_bb_P5 = conv
    conv = convolutional(conv, (3, 3, 512, 1024))
    # fmap_bb_P5 = conv
    conv_lbbox = convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
    conv_lbbox_decoded = decode(conv_lbbox, NUM_CLASS, i=2)

    output_tensors = [fmap_bb_P3, fmap_bb_P4, fmap_bb_P5, conv_sbbox, conv_mbbox, conv_lbbox]
    YOLOv4_backbone = tf.keras.Model(input_layer, output_tensors)
    return YOLOv4_backbone

if __name__ == '__main__':
    yolo_model = YOLOv4_Model(training=True, Modified_model=False, dilation_bb=False)
    flops = get_flops(yolo_model, batch_size=1)  
    print(f"FLOPS: {flops / 10 ** 9:.04} G")
    # yolo_model.summary()

    backbone = create_YOLOv4_backbone(CLASSES_PATH=YOLO_CLASS_PATH, dilation_bb=False)
    print(len(backbone.weights))

    # print(tf.shape(backbone.weights[400]))
    # print(tf.shape(yolo_model.weights[400]))


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






