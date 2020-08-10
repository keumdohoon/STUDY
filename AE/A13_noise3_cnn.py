#20200805 컬러 추가해주는 기능
#cifar10으로 autoencoder 구성할 것

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose
import numpy as np
import tensorflow as tf


import tensorflow as tf

def autoencoder():
    model = Sequential()
    model.add(Conv2D(filters = 128, kernel_size=(3,3),
                                    padding = 'valid', input_shape=(32,32,3),
                                    activation = 'relu')),
    model.add(Conv2DTranspose(filters = 3, kernel_size = (3,3),
                                                padding = 'valid', activation = 'sigmoid'))
    return model


#데이터
train_set, test_set = tf.keras.datasets.cifar10.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 3))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2],3))
x_train = x_train /255.
x_test = x_test /255.

print(x_train.shape)
print(x_test.shape)

model = autoencoder()
model.summary()


# model.compile(optimizer='adam', loss='mse', metrics=['acc'])  # loss = 0.01
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])    # loss = 0.09

-model.fit(x_train, x_train, epochs = 1)


