#keras34 

from numpy import array
from keras.layers import Dense, LSTM, Input
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, GRU


# 실습 : LSTM 레이어를 5개이랑 엮어서 Dense 결과를 이겨내시오!!!
#1. 데이터
x = array([[1, 2, 3],  [2, 3, 4], [3, 4, 5], [4, 5, 6],
           [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
           [9, 10, 11],[11, 12, 13],[20, 30, 40],[30, 40,50],[40,50,60]])

y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])#스칼라 4개짜리의 하나의 벡터

x_predict = array([55, 65, 75])




print("x.shape", x.shape) #x.shape (13, 3)
print("y.shape", y.shape) #y.shape (13,)#스칼라가 4개라는 뜻이다. 
#x = x.reshape(13,3,1)

x = x.reshape(x.shape[0], x.shape[1], 1)#x.shape 0에는 13가 들어가고 1에는 3이 들어간다. 
print("x.shape", x.shape)#(13,3,1)





#2. 모델구성
input1 = Input(shape=(3, 1))
dense1 = LSTM(6000, return_sequences=True)(input1)
dense2 = LSTM(4000, return_sequences=True)(dense1)
dense3 = LSTM(7000, return_sequences=True)(dense2)
dense4 = LSTM(1000, return_sequences=True)(dense3)
dense5 = LSTM(10000, return_sequences=True)(dense4)

dense6 = LSTM(2000)(dense5)

output1 = Dense(1000)(dense6)
output1 = Dense(1000)(output1)
output1 = Dense(1000)(output1)
output1 = Dense(1000)(output1)
output1 = Dense(1000)(output1)
output1 = Dense(200)(output1)

output2 = Dense(1, name='finalone')(output1)
model = Model(inputs = input1, outputs = output2)
model.summary()


#3. 실행
model.compile(optimizer='adam', loss = 'mse')
#early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x, y, epochs=1000, batch_size=32) #callbacks=[early_stopping])


#4. 예측

x_predict = x_predict.reshape(1,3,1)


# y_pred = model.predict(x_pred)
# print("y_pred : ", y_pred)
print("x_predict:",x_predict)
y_predict = model.predict(x_predict)
print("y_predict:", y_predict)
