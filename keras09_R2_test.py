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

model.add(Dense(1000, input_dim=1))
model.add(Dense(800))
model.add(Dense(600))
model.add(Dense(5))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(1000))



model.add(Dense(1))


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

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

#과제 : R2 가 음수로도 나올 수 있다. R2를 음수가 아닌 0.5이하로 줄이기
#레이어는 인풋과 아웃풋을 포함한 5개이상, 노드는 레이어당 각각5개이상
#batch size=1
#epochs = 100 이상
#위의 조건들을 보면 너무 조건이 좋다, 그래서 정확도가 올라갈수 밖에 없는데 오히려 정보가 너무 많아버리면 데이터가 터져버린다. 
#  