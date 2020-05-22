#keras34 
import numpy as np
from numpy import array
from keras.layers import Dense, LSTM, Input
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, GRU
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = array([[1, 2, 3],  [2, 3, 4], [3, 4, 5], [4, 5, 6],
           [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
           [9, 10, 11],[11, 12, 13],[20, 30, 40],[30, 40,50],[40,50,60]])

y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])#스칼라 4개짜리의 하나의 벡터

x_predict = array([55, 65, 75])




print("x.shape", x.shape) #x.shape (13, 3)
print("y.shape", y.shape) #y.shape (13,)#스칼라가 4개라는 뜻이다. 
#x = x.reshape(13,3,1)

#x = x.reshape(x.shape[0], x.shape[1], 1)#x.shape 0에는 13가 들어가고 1에는 3이 들어간다. 
#print("x.shape", x.shape)#(13,3,1)




model = Sequential()
#2. 모델구성
model.add(Dense(900, input_dim=(3)))
model.add(Dense(700))
model.add(Dense(500))
model.add(Dense(300))
model.add(Dense(100))
model.add(Dense(700))
model.add(Dense(500))
model.add(Dense(300))
model.add(Dense(100))
model.add(Dense(1))

model.summary()
#덴스 모델에서는 LSTM에서의 3차원정보를 2차원으로 바꾸어주지 못한다. 







#3. 실행
model.compile(optimizer='adam', loss = 'mse')
#early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x, y, epochs=1000, batch_size=32) #callbacks=[early_stopping])


#4. 예측

x_predict = x_predict.reshape(1,3)
# y_pred = model.predict(x_pred)
# print("y_pred : ", y_pred)
print("x_predict:",x_predict)
y_predict = model.predict(x_predict)
print("y_predict:", y_predict)
