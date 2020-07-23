#1. 데이터
import numpy as np 
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16, 17, 18])
#train 으로 훈련을 시키고 test으로 어느정도 맞는지를 확인한 다음에 acc와 loss가 알맞게 나온다면 
#그 가중치를 가지고 x_pred를 모델 프리딕트해준다면 y_pred를 찾아줄수 있다. 
#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(2, input_dim=1))
model.add(Dense(4))
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=110, batch_size=1)
#x_train, y_train 을 
#4. 평가와 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("loss ; ", loss)
print("mse : ", mse)

y_pred = model.predict(x_pred)
print("y_predict : ", y_pred)
#loss를 최소로 그리고 acc를 1로 맞춘다.        
#y_pred 를 근사치로 가까이오게 해야한다 이를 위하여 수행할수 있는게 몇가지 있는데,
#Layer 깊이 조정
#노드 갯수 수정
#에포 값변경
#batch_size 변경