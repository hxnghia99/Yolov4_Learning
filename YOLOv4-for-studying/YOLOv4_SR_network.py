import numpy as np
from tensorflow.keras.layers import (Input, BatchNormalization, Add, Lambda,
                                     Dense, Conv2D, LeakyReLU, PReLU)
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
from tensorflow.nn import depth_to_space



def edsr(num_filters: int = 64, num_res_blocks: int = 16):
    """
    Creates an EDSR model.
    
    Parameters
    ----------
    num_filters: int
        Number of filters per convolution layer.
        Default=64

    num_res_blocks: int
        Number of residual blocks in the model
        Default=16 

    Returns
    -------
        EDSR Model object.
    """
    DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255
    normalize = lambda x: (x - DIV2K_RGB_MEAN) / 127.5
    denormalize = lambda x: x * 127.5 + DIV2K_RGB_MEAN
    pixel_shuffle = lambda x: depth_to_space(x, 2)

    def residual_block(layer_input, filters, block_number):
        """Residual block described in paper"""
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu', name=f"conv_res_{block_number}_1")(layer_input)
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same', name=f"conv_res_{block_number}_2")(d)
        d = Add(name=f"add_res_{block_number}")([d, layer_input])
        return d

    def upsample_block(layer_input, i) :
        u = Conv2D(num_filters*4, kernel_size=3, strides=1, padding='same', name=f"conv_up_{i}")(layer_input)
        u = Lambda(pixel_shuffle, name=f"pix_shuf_{i}")(u)
        return u

    # ==================
    # Model Construction
    # ==================

    x_in = Input(shape=(None, None, 3), name="LR Batch")
    x = Lambda(normalize, name="normalize_input")(x_in)

    x = r = Conv2D(num_filters, 3, padding='same', name="Conv_ip")(x)
    for i in range(num_res_blocks):
        r = residual_block(r, num_filters, i)

    c2 = Conv2D(num_filters, 3, padding='same', name="conv_out")(r)
    c2 = Add(name="add_out")([x, c2])

    u1 = upsample_block(c2, 1)
    # u2 = upsample_block(u1, 2)
    c3 = Conv2D(3, 3, padding='same', name="conv_final")(u1)

    x_out = Lambda(denormalize, name="denormalize_output")(c3)
    return Model(x_in, x_out, name="EDSR")
