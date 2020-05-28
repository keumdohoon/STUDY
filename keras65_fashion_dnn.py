import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
plt.imshow(x_train[2])
plt.show()


print(x_train[2])
print('y_train[0] :', y_train[2])
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

x_train = x_train.reshape(60000, 784,).astype('float32')/ 255
x_test = x_test.reshape(10000, 784,).astype('float32')/ 255

#2. 모델
input1 = Input(shape= (784,))
dense1_1 = Dense(12)(input1)
dense1_2 = Dense(24)(dense1_1)
dense1_2 = Dense(24)(dense1_2)
dense1_2 = Dense(24)(dense1_2)
drop1 = Dropout(0.2)(dense1_2)
dense1_2 = Dense(24)(drop1)
dense1_2 = Dense(24)(dense1_2)
dense1_2 = Dense(24)(dense1_2)
drop2 = Dropout(0.2)(dense1_2)

output1_2 = Dense(32)(drop2)
output1_2 = Dense(16)(output1_2)
output1_2 = Dense(8)(output1_2)
output1_2 = Dense(4)(output1_2)
output1_3 = Dense(10)(output1_2)

model = Model(inputs = input1,
 outputs = output1_3)

#3. 훈련
model.compile(loss = 'categorical_crossentropy', optimizer= 'adam', metrics= ['accuracy'])
model.fit(x_train, y_train, epochs = 10, batch_size = 30)

#4, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size= 30)
print('loss: ', loss)
print('acc: ', acc)

