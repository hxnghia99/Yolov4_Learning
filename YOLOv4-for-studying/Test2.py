import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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




input = Input([100, 100, 3])
conv1 = Conv2D(filters=10, kernel_size=3, padding='same')(input)
conv2 = BatchNormalization()(conv1)
conv2 = LeakyReLU()(conv2)
conv2 = Conv2D(filters=20, kernel_size=3, padding='same', strides=2)(conv2)

model1 = tf.keras.Model(input, conv1)
model2 = tf.keras.Model(input, conv2)

# test = tf.fill((1, 100,100,3), 2)

# test1 = model1(test)
# test2 = model2(test)

# print(test1)
# print("\n")
# print(test2)
# print("\n")

model2.trainable_variables[0].assign(tf.fill(tf.shape(model2.trainable_variables[0]), 10.0))

for i in range(2):
    model1.trainable_variables[i].assign(model2.trainable_variables[i])



print(model1.trainable_variables[0])
print(model1.trainable_variables[1])
print("\n")
print(model2.trainable_variables[0])
print(model2.trainable_variables[1])

print(len(model1.trainable_variables))
print(len(model2.trainable_variables))
