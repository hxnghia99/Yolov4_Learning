from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow as tf
import numpy as np

test = np.zeros([1,56,32,128])

def resnet_50_model():
    resnet50 = ResNet50(input_shape=(256,448,3), include_top=False)
    resnet = tf.keras.models.Model(resnet50.layers[45].input, resnet50.layers[174].output)
    return resnet


def vgg_19_model():
    vgg = VGG19(input_shape=(128,224,3), include_top=False)
    vgg = tf.keras.models.Model(vgg.layers[6].output, vgg.layers[20].output)
    return vgg

# a = vgg_19_model()
# # print(a.summary())


b = resnet_50_model()
print(b.summary())