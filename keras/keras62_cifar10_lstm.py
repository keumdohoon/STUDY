from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train[0])
print('y_train[0] :', y_train[0]) #y_train[0] : [6]

print(x_train.shape)              #(50000, 32, 32, 3)
print(x_train.shape)              #(50000, 32, 32, 3)
print(y_train.shape)              #(50000, 1)
print(y_test.shape)               #(10000, 1)

plt.imshow(x_train[0])
plt.show()

#데이터 전처리
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)#(50000, 10)

x_train = x_train.reshape(50000, 32, 96).astype('float32') / 255
x_test = x_test.reshape(10000, 32, 96).astype('float32') / 255


#2. 모델구성

input1 = Input(shape=(32,96))
dense1 = LSTM(6)(input1)
dense2 = Dense(36)(dense1)
dense3 = Dense(126)(dense2)
dense4 = Dense(36)(dense3)

output1 = Dense(6)(dense4)
output2 = Dense(10)(output1)
model = Model(inputs = input1, outputs = output2)
model.summary()




#3. 실행
model.compile(optimizer='adam', loss = 'mse', metrics= ['acc'])
model.fit(x_train, y_train, epochs=1, batch_size=32)



#4. 예측
loss,acc = model.evaluate(x_test, y_test, batch_size=30)

print("loss : {loss}", loss)
print("acc : {acc}", acc)






