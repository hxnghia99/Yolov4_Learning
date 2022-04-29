from tkinter import Y
import tensorflow as tf
import tensorflow.keras.backend as K


#Parameters configuration
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


# #YOLOv2
# architecture_config = [
#     #Block1
#     #tuple = (filters, kernel_size, strides, padding, activation, $$$input_shape)
#     (32, (3,3), (1,1), 'same', 'relu', (416, 416, 3)),                              #1
#     'Batch',
#     'Max',      #Maxpooling strides=(2,2)
#     #Block2
#     (64, (3,3), (1,1), 'same', 'relu'),                                             #2
#     'Batch',
#     'Max',      #Maxpooling strides=(2,2)
#     #Block3
#     (128, (3,3), (1,1), 'same', 'relu'),                                            #3
#     'Batch',
#     (64, (1,1), (1,1), 'same', 'relu'),                                             #4
#     'Batch',
#     (128, (3,3), (1,1), 'same', 'relu'),                                            #5
#     'Batch',
#     'Max',      #Maxpooling strides=(2,2)
#     #Block4
#     #[1st layer tuple, 2nd layer tuple, repeat times]
#     # [(256, (1,1), (1,1), 'same', 'relu'), (512, (3,3), (1,1), 'same', 'relu'), 4],
#     (256, (3,3), (1,1), 'same', 'relu'),                                            #6
#     'Batch',    
#     (128, (1,1), (1,1), 'same', 'relu'),                                            #7
#     'Batch',
#     (256, (3,3), (1,1), 'same', 'relu'),                                            #8
#     'Batch',
#     'Max',
#     #Block5
#     #[1st layer tuple, 2nd layer tuple, repeat times]
#     [(512, (3,3), (1,1), 'same', 'relu'), (256, (1,1), (1,1), 'same', 'relu'), 2],  #9,10,11,12
#     (512, (3,3), (1,1), 'same', 'relu'),                                            #13                 #passthrough layer from here to #20
#     'Batch',
#     'Max',
#     #Block6
#     [(1024, (3,3), (1,1), 'same', 'relu'), (512, (1,1), (1,1), 'same', 'relu'), 2], #14,15,16,17
#     (1024, (3,3), (1,1), 'same', 'relu'),                                           #18
#     'Batch',
#     (1024, (3,3), (1,1), 'same', 'relu'),                                           #19
#     'Batch',    
#     (1024, (3,3), (1,1), 'same', 'relu'),                                           #20
#     'Batch',
#     (1024, (3,3), (1,1), 'same', 'relu'),                                           #21
#     'Batch',
#     (125, (1,1), (1,1), 'same', 'relu'),                                            #22
#     'Batch',
#     #Passthrough layer
# ]


class Yolo_Reshape(tf.keras.layers.Layer):
  def __init__(self, target_shape):
    super(Yolo_Reshape, self).__init__()
    self.target_shape = tuple(target_shape)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'target_shape': self.target_shape
    })
    return config

  def call(self, input):
    # grids 7x7
    S = [self.target_shape[0], self.target_shape[1]]
    # classes
    C = 20
    # no of bounding boxes per grid
    B = 2

    idx1 = S[0] * S[1] * C              #7x7x20
    idx2 = idx1 + S[0] * S[1] * B       #idx1 + 7x7x2
    
    # class probabilities
    class_probs = K.reshape(input[:, :idx1], (K.shape(input)[0],) + tuple([S[0], S[1], C]))     #batch x 7 x 7 x 20
    class_probs = K.softmax(class_probs)

    #confidence
    confs = K.reshape(input[:, idx1:idx2], (K.shape(input)[0],) + tuple([S[0], S[1], B]))       #batch x 7 x 7 x 2
    confs = K.sigmoid(confs)

    # boxes
    boxes = K.reshape(input[:, idx2:], (K.shape(input)[0],) + tuple([S[0], S[1], B * 4]))       #batch x 7 x 7 x 8
    boxes = K.sigmoid(boxes)

    outputs = K.concatenate([class_probs, confs, boxes])
    return outputs




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
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Conv2D(filters=conv2[0], kernel_size=conv2[1], strides=conv2[2], padding=conv2[3], activation=conv2[4]))
            model.add(tf.keras.layers.BatchNormalization())
    elif type(x) == str and x == 'Flatten':
        model.add(tf.keras.layers.Flatten())
    elif type(x) == str and x == 'Dense1':
        model.add(tf.keras.layers.Dense(4096))
    elif type(x) == str and x == 'Dropout':
        model.add(tf.keras.layers.Dropout(0.5))
    elif type(x) == str and x == 'Dense2':
        model.add(tf.keras.layers.Dense(1470, activation='sigmoid'))
    elif type(x) == str and x == 'Reshape':
        # model.add(tf.keras.layers.Reshape((7,7,30)))
        model.add(Yolo_Reshape(target_shape=(7,7,30)))
    elif type(x) == str and x == 'Avg':
        model.add(tf.keras.layers.GlobalAveragePooling2D())
    elif type(x) == str and x == 'Softmax':
        model.add(tf.keras.layers.Softmax())
    elif type(x) == str and x == 'Batch':
        model.add(tf.keras.layers.BatchNormalization())


if __name__ == "__main__":
    print(model.summary())



