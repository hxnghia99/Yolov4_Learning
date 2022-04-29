import tensorflow as tf
import tensorflow.keras.backend as K

a = tf.Variable([[[[100,200]],[[400,500]]]])
print(a.shape)

b = tf.Variable([[[[1,2],[3,4]],[[5,6],[7,8]]],[[[9,10],[11,12]],[[13,14],[15,16]]]])
print(b.shape)

print(b[...,1])


# c = tf.Variable([[[[1,2],[4,5]],[[7,8],[10,10]]],[[[10,20],[40,50]],[[70,80],[100,100]]]])

# c = K.expand_dims(a,2)
# # print(c)
# print(c.shape)



# print((b+c))
# print((b+c).shape)

