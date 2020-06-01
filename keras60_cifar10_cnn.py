from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train[0])
print('y_train[0] :', y_train[0]) #y_train[0] : [6]

# print(x_train.shape)              #(50000, 32, 32, 3)
# print(x_train.shape)              #(10000, 32, 32, 3)
# print(y_train.shape)              #(50000, 1)
# print(y_test.shape)               #(10000, 1)

plt.imshow(x_train[0])
plt.show()

#데이터 전처리
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)#(50000, 10)

x_train = x_train.reshape(50000, 32,32,3).astype('float32') / 255
x_test = x_test.reshape(10000, 32,32,3).astype('float32') / 255

#2. 모델
input1= Input(shape= (32,32,3))
Conv2d1 = Conv2D(filters= 15, kernel_size=3, padding="same", activation="relu")(input1)

Conv2d2= Conv2D(filters= 15, kernel_size=3, padding="same", activation="relu")(Conv2d1) 
drop1 = Dropout(0.1)(Conv2d2)

Conv2d3= Conv2D(filters= 22, kernel_size=3, padding="same", activation="relu")(drop1) 
Conv2d4= Conv2D(filters= 21, kernel_size=3, padding="same", activation="relu")(Conv2d3) 
drop2 = Dropout(0.3)(Conv2d4)

Conv2d5= Conv2D(filters= 21, kernel_size=3, padding="same", activation="relu")(drop2) 
drop3 = Dropout(0.3)(Conv2d5)
Conv2d6= Conv2D(filters= 12, kernel_size=3, padding="same", activation="relu")(drop3) 
drop4 = Dropout(0.3)(Conv2d6)

Conv2d7= Conv2D(filters= 18, kernel_size=3, padding="same", activation="relu")(drop4) 

drop5 = Dropout(0.1)(Conv2d7)

Conv2d8= Conv2D(filters= 12, kernel_size=3, padding="same", activation="relu")(drop5) 
drop6 = Dropout(0.1)(Conv2d8)

output1 = (Flatten())(drop6)
output2 = Dense(10, activation = 'softmax')(output1)
model = Model(inputs=input1, outputs=output2)
model.summary()
print(x_train)
print(y_train)


#3. 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 50)


#acc75프로로 잡아라
#4. 예측
loss,acc = model.evaluate(x_test,y_test, batch_size=30)

print("loss : {loss}", loss)
print("acc : {acc}", acc)

early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')











