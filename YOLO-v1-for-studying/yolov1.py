# import tensorflow as tf
# import os

# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# print(len(tf.config.list_physical_devices('GPU')))


# from ctypes import sizeof
# import time
# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# from torch import conv2d

# time0 = time.time()
# print(tf.__version__)

# #dummy data
# train_images = np.random.rand(60000, 28, 28)
# train_labels = np.random.randint(0,10,60000)
# test_images = np.random.rand(10000, 28, 28)
# test_labels = np.random.randint(0,10,10000)

# print(len(train_images))
# print(len(train_labels))
# print(len(test_images))
# print(len(test_labels))

# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(10, activation='softmax')
#     ])
# model.compile(optimizer='adam', 
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(train_images, train_labels, epochs=5)

# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# print('\nTest accuracy:', test_acc)
# print("time: ", time.time() - time0)


import tensorflow as tf



(img_h, img_w) = (448, 448)


#Create YOLOv1 model with initial configuration
architecture_config = [
    #Block1
    #tuple = (filters, kernel_size, strides, padding, activation, $$$input_shape)
    (64, (7,7), (2,2), 'same', 'relu', (img_h, img_w, 3)),
    'Max',      #Maxpooling strides=(2,2)
    #Block2
    (192, (3,3), (1,1), 'same', 'relu'),
    'Max',      #Maxpooling strides=(2,2)
    #Block3
    (128, (1,1), (1,1), 'same', 'relu'),
    (256, (3,3), (1,1), 'same', 'relu'),
    (256, (1,1), (1,1), 'same', 'relu'),
    (512, (3,3), (1,1), 'same', 'relu'),
    'Max',      #Maxpooling strides=(2,2)
    #Block4
    #[1st layer tuple, 2nd layer tuple, repeat times]
    [(256, (1,1), (1,1), 'same', 'relu'), (512, (3,3), (1,1), 'same', 'relu'), 4],
    (512, (1,1), (1,1), 'same', 'relu'),
    (1024, (3,3), (1,1), 'same', 'relu'),
    'Max',
    #Block5
    #[1st layer tuple, 2nd layer tuple, repeat times]
    [(512, (1,1), (1,1), 'same', 'relu'), (1024, (3,3), (1,1), 'same', 'relu'), 2],
    (1024, (3,3), (1,1), 'same', 'relu'),
    (1024, (3,3), (2,2), 'same', 'relu'),
    #Block6
    (1024, (3,3), (1,1), 'same', 'relu'),
    (1024, (3,3), (1,1), 'same', 'relu'),
    #Block7
    'Flatten', 
    'Dense1',      #Fully connected layer with output = 4096
    #Block8
    'Dropout',
    'Dense2',
    'Reshape'
]


model = tf.keras.Sequential()
for x in architecture_config:
    if type(x) == tuple and len(x)==6:
        model.add(tf.keras.layers.Conv2D(filters=x[0], kernel_size=x[1], strides=x[2], padding=x[3], activation=x[4], input_shape=x[5]))
    elif type(x) == tuple:
        model.add(tf.keras.layers.Conv2D(filters=x[0], kernel_size=x[1], strides=x[2], padding=x[3], activation=x[4]))
    elif type(x) == str and x == 'Max':
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
    elif type(x) == list:
        conv1 = x[0]
        conv2 = x[1]
        repeat_times = x[2]
        for _ in range(repeat_times):
            model.add(tf.keras.layers.Conv2D(filters=conv1[0], kernel_size=conv1[1], strides=conv1[2], padding=conv1[3], activation=conv1[4]))
            model.add(tf.keras.layers.Conv2D(filters=conv2[0], kernel_size=conv2[1], strides=conv2[2], padding=conv2[3], activation=conv2[4]))
    elif type(x) == str and x == 'Flatten':
        model.add(tf.keras.layers.Flatten())
    elif type(x) == str and x == 'Dense1':
        model.add(tf.keras.layers.Dense(4096))
    elif type(x) == str and x == 'Dropout':
        model.add(tf.keras.layers.Dropout(0.5))
    elif type(x) == str and x == 'Dense2':
        model.add(tf.keras.layers.Dense(1470))
    elif type(x) == str and x == 'Reshape':
        model.add(tf.keras.layers.Reshape((7,7,30)))
# print(model.summary())





