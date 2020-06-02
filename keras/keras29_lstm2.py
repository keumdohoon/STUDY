from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3,], [2,3,4,],[3,4,5],[4,5,6]])
y = array([4,5,6,7])#스칼라 4개짜리의 하나의 벡터


print("x.shape", x.shape) #x.shape (4, 3)
print("y.shape", y.shape) #y.shape (4,)#스칼라가 4개라는 뜻이다. 
#x = x.reshape(4,3,1)
x = x.reshape(x.shape[0], x.shape[1], 1)#x.shape 0에는 4가 들어가고 1에는 3이 들어간다. 
'''
                행    ,       열   . 몇개씩 자르는지
x의 shape = (batch_size, timesteps, feature)
input_shape = (timesteps, feature)
input_length = timesteps, input_dim = feature
               `
'''
#[[1,2,3],[1,2,3]]
#[[[1,2],[4,3]],[[4,5],[5,6]]]
#[[[1],[2],[3]],[[4],[5],[6]]]
#[[[1,2,3,4]]]
#[[[[1],[2]]],[[[3],[4]]]]
print("x.shape", x.shape)#(4,3,1)
#2. 모델구성
model = Sequential()

#model.add(LSTM(10, activation='relu', input_shape=(3,1)))#원래는 (4,3,1)인데 행을 무시해서 없다. 
model.add(LSTM(985, input_length= 3, input_dim=1))#원래는 (4,3,1)인데 행을 무시해서 없다. 
model.add(Dense(430))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(1))

model.summary()

#과제 #파라미터가 왜 480개까지 되는지 찾아와라

#3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x,y, epochs=900, batch_size=1,  verbose=0)

x_predict = array([5, 6, 7])
x_predict = x_predict.reshape(1,3,1)

#4. 예측
print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)

'''
Total params: 4,419,161
Trainable params: 4,419,161
Non-trainable params: 0
'''