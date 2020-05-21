#앙상블 모델로 만드시오

from numpy import array
from keras.models import Model
from keras.layers import Dense, LSTM, Input
#

#1. 데이터
x1 = array([[1, 2, 3],  [2, 3, 4], [3, 4, 5], [4, 5, 6],
           [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
           [9, 10, 11],[10, 11, 12],[20, 30, 40],[30, 40,50],[40,50,60]])
x2 = array([[10, 20, 30],  [20, 30, 40], [30, 40, 50], [40, 50, 60],
           [50, 60, 70], [60, 70, 80], [70, 80, 90], [80, 90, 100],
           [90, 100, 110],[100, 110, 120],[2, 3, 4],[3, 4,5],[4,5,6]])
y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])#스칼라 4개짜리의 하나의 벡터

x1_predict = array([55, 65, 75])
x2_predict = array([65, 75, 85])



print("x1.shape", x1.shape) #x.shape (13, 3)
print("x2.shape", x2.shape) #x.shape (13, 3)
print("y.shape", y.shape) #y.shape (13,)#스칼라가 4개라는 뜻이다. 
#x = x.reshape(13,3,1)
x1 = x1.reshape(x1.shape[0], x1.shape[1], 1) 
print("x1.shape", x1.shape)             
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1) 
print("x2.shape", x2.shape)             

#2. 모델구성
###############################################################
input1 = Input(shape= (3, 1))
dense1_1=LSTM(95)(input1)
dense1_2=Dense(40)(dense1_1)
dense1_3=Dense(20)(dense1_2)




input2 = Input(shape=(3, 1))
dense2_1=LSTM(3)(input2)
dense2_2=Dense(43)(dense2_1)
dense2_3=Dense(20)(dense2_2)


from keras.layers.merge import concatenate
merge1 = concatenate([dense1_3, dense2_3])

middle1 = Dense(10)(merge1)
middle2 = Dense(20)(middle1)
middle3 = Dense(40)(middle2)

####output모델구성######
output1_1 = Dense(10)(middle3)
output1_2 = Dense(7)(output1_1)
output1_3 = Dense(1)(output1_2)
#input1 and input 2 will be merged into one. 
model = Model(inputs = [input1, input2], outputs = output1_3)
model.summary()


#3. 실행
##################################
model.compile(optimizer = 'adam', loss='mse')
model.fit([x1, x2], y, epochs=800, batch_size=32)


#4. 예측
print('x1',x1_predict)
print('x2',x2_predict)
x1_predict = x1_predict.reshape(1,3,1)
x2_predict = x2_predict.reshape(1,3,1)
y_predict = model.predict([x1_predict, x2_predict])
print(y_predict)


  
