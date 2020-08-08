#20200804
#a08 복붙
#CNN으로 오토인코더를 구성하시오.

import tensorflow as tf

def autoencoder():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters = 128, kernel_size=(3,3),
                                    padding = 'valid', input_shape=(28,28,1),
                                    activation = 'relu')),
    model.add(tf.keras.layers.Conv2DTranspose(filters = 1, kernel_size = (3,3),
                                                padding = 'valid', activation = 'sigmoid'))
    return model


#데이터
train_set, test_set = tf.keras.datasets.mnist.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2],1))
x_train = x_train /255.
x_test = x_test /255.

print(x_train.shape)
print(x_test.shape)

model = autoencoder()
model.summary()


# model.compile(optimizer='adam', loss='mse', metrics=['acc'])  # loss = 0.01
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])    # loss = 0.09

model.fit(x_train, x_train, epochs = 10)


