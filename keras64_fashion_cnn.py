import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, Dropout, Flatten, Dense
from keras.datasets import fashion_mnist
from keras.models import Model
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
plt.imshow(x_train[0])
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
print(y_test.shape)
 #(10000, 10)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/ 255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/ 255

#2. 모델
input1 = Input(shape= (28,28,1))
Conv2d1 = Conv2D(filters = 20, kernel_size= 9, padding= 'same', activation='elu')(input1)

Conv2d2 = Conv2D(filters= 25, kernel_size= 3, padding= 'same', activation= 'elu')(Conv2d1)
Conv2d3 = Conv2D(filters= 15, kernel_size= 2, padding= 'same', activation= 'elu')(Conv2d2)
drop1 = Dropout(0.2)(Conv2d3)

Conv2d4 = Conv2D(filters= 10, kernel_size= 3, padding = 'same', activation= 'elu')(drop1)
Conv2d5 = Conv2D(filters= 20, kernel_size= 2, padding = 'same', activation= 'elu')(Conv2d4)
drop2= Dropout(0.2)(Conv2d5)

output1 = (Flatten())(drop2)
output2 = Dense(10, activation='softmax')(output1)
model = Model(inputs= input1, outputs= output2)
model.summary()

print(x_train)
print(y_train)

#3. 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ['acc'])
model.fit(x_train, y_train, epochs = 10)

#4, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 30)
print("loss :,  ", loss)
print("acc : ", acc)
#############양호#################
#####acc:0.8884000182151794#######
##################################