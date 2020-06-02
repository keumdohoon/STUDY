import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
plt.imshow(x_train[2])
plt.show()


print(x_train[0])
print('y_train[0] :', y_train[0])
 #y_train[0] : 9
print(x_train.shape)
 #(60000, 28, 28)
print(x_test.shape)
 #(10000, 28, 28)
print(y_train.shape)
 # (60000,)
print(y_test.shape)
 #(10000,)

# 데이터 전처리
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)
 #(60000, 10)

x_train = x_train.reshape(60000, 784).astype('float32')/ 255
x_test = x_test.reshape(10000, 784).astype('float32')/ 255

#2. 모델
model= Sequential()
model.add(Dense(12, activation='relu', input_shape=(784,)))
model.add(Dense(12, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(24, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()
#3. 훈련
model.compile(loss = 'categorical_crossentropy', optimizer= 'adam', metrics= ['accuracy'])
model.fit(x_train, y_train, epochs = 10, batch_size = 30, validation_split=0.2)

#4, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size= 30)
print('loss: ', loss)
print('acc: ', acc)

# loss:  0.43370584832131864
# acc:  0.8468999862670898