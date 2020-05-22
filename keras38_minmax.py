#keras34 

from numpy import array
from keras.layers import Dense, LSTM, Input
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler



#1. 데이터
x = array([[1, 2, 3],  [2, 3, 4], [3, 4, 5], [4, 5, 6],
           [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
           [9, 10, 11],[11, 12, 13],[2000, 3000, 4000],[3000, 4000,5000],[4000,5000,6000], [100,200,300]])
#14,3
y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 5000, 6000, 7000, 400])#(14)

x_predict = array([55, 65, 75])
x_predict = x_predict.reshape(1,3)
#scaler= MinMaxScaler()#이거를 스칼라라고 이름지어준다.
scaler = StandardScaler()
scaler.fit(x)#전처리에서 fit은 실행한다는 뜻이다. 
x= scaler.transform(x)#fit한 결과는 트랜스폼 하라는 뜻이다.        
x_predict = scaler.transform(x_predict)



print("x.shape", x.shape) #(14, 3, 1)
print("y.shape", y.shape) #y.shape (14,) 
x = x.reshape(x.shape[0], x.shape[1], 1)#x.shape 0에는 13가 들어가고 1에는 3이 들어간다. 
print("x.shape", x.shape)#(13,3,1)

#X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
#X_scaled = X_std * (max - min) + min

  



#2. 모델구성
input1 = Input(shape=(3, 1))
dense1 = LSTM(80, return_sequences=True)(input1)
dense2 = LSTM(40, return_sequences=False)(dense1)
#위에 레이어의 아웃풋은 2차원으로 해줬다 그런데 밑에 레이어는 3차원으로 나와야하기 때문에 valueerror이 뜨게 되는 것이다. deminesion3차를 기대했는데 받은건 2차원이다.

#덴스 모델에서는 LSTM에서의 3차원정보를 2차원으로 바꾸어주지 못한다. 
output1 = Dense(30)(dense2)
output1 = Dense(20)(output1)
output1 = Dense(15)(output1)
output1 = Dense(50)(output1)
output1 = Dense(35)(output1)
output1 = Dense(20)(output1)

output2 = Dense(1, name='finalone')(output1)
model = Model(inputs = input1, outputs = output2)
model.summary()

#3. 실행
model.compile(optimizer='adam', loss = 'mse')
#early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x, y, epochs=110, batch_size=32) #callbacks=[early_stopping])


#4. 예측
x_predict = x_predict.reshape(1, 3, 1)



# y_pred = model.predict(x_pred)
# print("y_pred : ", y_pred)
print("x_predict:",x_predict)
y_predict = model.predict(x_predict)
print("y_predict:", y_predict)
