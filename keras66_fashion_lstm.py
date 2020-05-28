import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, Dropout


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


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
 # (10000)

plt.imshow(x_train[0])
 #plt.show()

#데이터 전처리 OneHot
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)#(60000, 10)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


#2. 모델구성
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, Dropout


model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(28,28)))

model.add(Dense(36, activation='relu'))
model.add(Dense(50, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()




#3. 실행
model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics= ['acc'])
model.fit(x_train, y_train, epochs=20, batch_size=52)



#4. 예측
loss,acc = model.evaluate(x_test, y_test, batch_size=52)

print("loss : {loss}", loss)
print("acc : {acc}", acc)


#acc=0.85