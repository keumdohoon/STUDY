from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3], [2,3,4],[3,4,5],[4,5,6]])
y = array([4,5,6,7])#스칼라 4개짜리의 하나의 벡터
y2 = array([[4,5,6,7]]) #y2=(1,4)
y3 = array([[4],[5],[6],[7]])  #(4,1)

print("x.shape", x.shape) #x.shape (4, 3)
print("y.shape", y.shape) #y.shape (4,)#스칼라가 4개라는 뜻이다. 
#x = x.reshape(4,3,1)
x = x.reshape(x.shape[0], x.shape[1], 1)#x.shape 0에는 4가 들어가고 1에는 3이 들어간다. 


#[[1,2,3],[1,2,3]]
#[[[1,2],[4,3]],[[4,5],[5,6]]]
#[[[1],[2],[3]],[[4],[5],[6]]]
#[[[1,2,3,4]]]
#[[[[1],[2]]],[[[3],[4]]]]
print("x.shape", x.shape)#(4,3,1)
#2. 모델구성
model = Sequential()

model.add(LSTM(10, activation='relu', input_shape=(3,1)))#원래는 (4,3,1)인데 행을 무시해서 없다. 
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(20))
model.add(Dense(5))
model.add(Dense(5))

model.add(Dense(1))

model.summary()

#과제 #파라미터가 왜 480개까지 되는지 찾아와라

#3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x,y, epochs=110, batch_size=1)

x_input = array([5, 6, 7])
x_input = x_input.reshape(1,3,1)

print(x_input)

yhat = model.predict(x_input)
print(yhat)

