#keras34 

from numpy import array
from keras.layers import Dense, LSTM, Input
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, GRU

#1. 데이터
x = array([[1, 2, 3],  [2, 3, 4], [3, 4, 5], [4, 5, 6],
           [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
           [9, 10, 11],[11, 12, 13],[20, 30, 40],[30, 40,50],[40,50,60]])

y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])#스칼라 4개짜리의 하나의 벡터

x_predict = array([50, 60, 70])




print("x.shape", x.shape) #x.shape (13, 3)
print("y.shape", y.shape) #y.shape (13,)#스칼라가 4개라는 뜻이다. 
#x = x.reshape(13,3,1)

x = x.reshape(x.shape[0], x.shape[1], 1)#x.shape 0에는 13가 들어가고 1에는 3이 들어간다. 
print("x.shape", x.shape)#(13,3,1)





#2. 모델구성
input1 = Input(shape=(3, 1))
dense1 = LSTM(800, return_sequences=True)(input1)
dense2 = LSTM(400, return_sequences=False)(dense1)
#위에 레이어의 아웃풋은 2차원으로 해줬다 그런데 및에는 3차원으로 나와야하기 때문에 valueerror이 뜨게 되는 것이다. deminesion3차를 기대했는데 받은건 2차원이다datetime A combination of a date and a time. Attributes: ()

#덴스 모델에서는 LSTM에서의 3차원정보를 2차원으로 바꾸어주지 못한다. 
output1 = Dense(300)(dense2)
output1 = Dense(200)(output1)
output1 = Dense(150)(output1)
output1 = Dense(50)(output1)
output1 = Dense(35)(output1)
output1 = Dense(20)(output1)

output2 = Dense(1, name='finalone')(output1)
model = Model(inputs = input1, outputs = output2)
model.summary()
#인풋 Shape가 3,1 이다 행의 갯수는 총 13개이다. 행은 무시니까 타임 스텝스 3이랑 피쳐 1, 인풋딤1, 인풋 랭스가 3, 피쳐의 갯수와 
#(N,3,1)이 첫번째 input_shape이다. 하지만 n은 포함하지 않는다. 
# 리턴 시퀀스 10개  는 아웃풋 노드의 갯수이다. 다음 레이어의 노드의 갯수는 그 전 레이어의 피쳐가 차지하게된다.
#인풋은 항상 인풋의 피쳐 위치로 가게 된다. 
#리턴 시퀀스는 차원이 계속 유지된다. 
#서머리로 갔을때 아웃풋 노드의 갯수는 피쳐와 같다. 

'''
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 3, 1)              0
_________________________________________________________________
lstm_1 (LSTM)                (None, 3, 10)             480
_________________________________________________________________
lstm_2 (LSTM)                (None, 10)                840
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 55
_________________________________________________________________
finalone (Dense)             (None, 1)                 6
=================================================================
Total params: 1,381
'''

#3. 실행
model.compile(optimizer='adam', loss = 'mse')
#early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x, y, epochs=1100, batch_size=32) #callbacks=[early_stopping])


#4. 예측

x_predict = x_predict.reshape(1,3,1)


# y_pred = model.predict(x_pred)
# print("y_pred : ", y_pred)
print("x_predict:",x_predict)
y_predict = model.predict(x_predict)
print("y_predict:", y_predict)
