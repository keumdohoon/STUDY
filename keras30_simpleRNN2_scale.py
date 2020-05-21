from numpy import array
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.callbacks import EarlyStopping
#1. 데이터
x = array([[1, 2, 3],  [2, 3, 4], [3, 4, 5], [4, 5, 6],
           [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
           [9, 10, 11],[10, 11, 12],[20, 30, 40],[30, 40,50],[40,50,60]])

y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])#스칼라 4개짜리의 하나의 벡터

x_predict = array([50, 60, 70])




print("x.shape", x.shape) #x.shape (13, 3)
print("y.shape", y.shape) #y.shape (13,)#스칼라가 4개라는 뜻이다. 
#x = x.reshape(13,3,1)

x = x.reshape(x.shape[0], x.shape[1], 1)#x.shape 0에는 13가 들어가고 1에는 3이 들어간다. 
print("x.shape", x.shape)#(13,3,1)

             



#2. 모델구성
model = Sequential()

#model.add(LSTM(10, activation='relu', input_shape=(3,1)))#원래는 (13,3,1)인데 행을 무시해서 없다. 
model.add(SimpleRNN(985, input_shape=(3,1)))#원래는 (13,3,1)인데 행을 무시해서 없다. 
model.add(Dense(430))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(1))

model.summary()

#3. 실행
model.compile(optimizer='adam', loss='mse')

from keras.callbacks import EarlyStopping
#early_stoppong = EarlyStopping(monitor='loss', patience=2, mode='auto')
#model.fit(x, y, epochs=100, batch_size=1, validation_split=0.25, verbose=1, 
#          callbacks=[early_stoppong])




model.fit(x, y, epochs=800, batch_size=32, verbose=0)


x_predict = x_predict.reshape(1,3,1)

#4. 예측
print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)
'''
Total params: 1,502,576
Trainable params: 1,502,576
Non-trainable params: 0

[[77.130745]]
[[77.552124]]
[[77.59611]]
'''