#1. 데이터
import numpy as np 
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16, 17, 18])

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
model.fit(x_train, y_train, epochs=10010, batch_size=1)

#4. 평가와 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("loss ; ", loss)
print("mse : ", mse)

y_predict = model.predict(x_test)
print(y_predict)

#5.RMSE구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RSME:", RMSE(y_test, y_predict))

#6. R2구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

#과제 : R2 가 음수로도 나올 수 있다. R2를 은수가 아닌 0.5이하로 줄이기
#레이어는 인풋과 아웃풋을 포함한 5개이상, 노드는 레이어당 각각5개이상
#batch size=1
#epochs = 100 이상
