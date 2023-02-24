from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, GlobalAvgPool2D, Input, BatchNormalization, Activation, Add
import tensorflow as tf
from tensorflow.keras import layers


def ResNet(stack_fn,
           use_bias,
           model_name='resnet',
           input_shape=[224,224,3]):
    bn_axis=3
    img_input = layers.Input(input_shape)

    x = layers.ZeroPadding2D(
        padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
    x = layers.Activation('relu', name='conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)

    # Create model.
    model = tf.keras.models.Model(img_input, x, name=model_name)

    model.load_weights("YOLOv4-for-studying/model_data/resnet34_imagenet_notop.h5")

    return model

def block3(x,
          filters,
          kernel_size=3,
          stride=1,
          conv_shortcut=True,
          name=None):

  bn_axis = 3
  if conv_shortcut:
    shortcut = layers.Conv2D(
        filters,
        1,
        strides=stride,
        use_bias=False,
        name=name + '_0_conv')(x)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
  else:
    shortcut = x

  x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_1_pad')(x)
  x = layers.Conv2D(
      filters, kernel_size=kernel_size, strides=stride, use_bias=False, name=name + '_1_conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
  x = layers.Activation('relu', name=name + '_1_relu')(x)

  x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
  x = layers.Conv2D(
      filters, kernel_size=kernel_size, use_bias=False, name=name + '_2_conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)

  x = layers.Add(name=name + '_add')([shortcut, x])
  x = layers.Activation('relu', name=name + '_out')(x)
  return x

def stack3(x, filters, blocks, stride1=2, conv_shortcut=True, name=None):
  x = block3(x, filters, stride=stride1, conv_shortcut=conv_shortcut, name=name + '_block1')
  for i in range(2, blocks + 1):
    x = block3(
        x,
        filters,
        conv_shortcut=False,
        name=name + '_block' + str(i))
  return x

def ResNet34(input_shape=None):
  def stack_fn(x):
    x = stack3(x, 64, 3, stride1=1, conv_shortcut=False, name='conv2')
    x = stack3(x, 128, 4, name='conv3')
    x = stack3(x, 256, 6, name='conv4')
    return stack3(x, 512, 3, name='conv5')
  return ResNet(stack_fn, use_bias=False, model_name='resnet34', input_shape=input_shape)




# resnet34 = ResNet34((256,448,3))
# print(resnet34.summary())
# print("A")