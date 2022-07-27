import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, ZeroPadding2D, MaxPool2D, Input, UpSampling2D
from tensorflow.keras.regularizers import L2
import cv2
import colorsys
import random
import shutil
import sys
import tensorflow_hub as hub




a = tf.Variable([1,2,3])
print(tf.shape(a))