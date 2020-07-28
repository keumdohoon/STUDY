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
print(model)
model.fit(x_train, y_train, epochs=10010, batch_size=1)
print(model)
#위에 있는 모델데로 훈련을 시켜주겠다는 뜻이다. model.fit을 통하여 위의 모델을 실행시키고 


#4. 평가와 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("loss ; ", loss)
print("mse : ", mse)

y_predict = model.predict(x_test)
print(y_predict)

#5.RMSE구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):#RMSE라는 함수안에는 y_test 와 y_predict 가 들어가게 된다는 뜻이다. 
    return np.sqrt(mean_squared_error(y_test, y_predict))#square root 을 적용 해주고 y_test, predict를 MSE해준거를 Square root 해준다 .
print("RSME:", RMSE(y_test, y_predict))#위에서 함수를 정해 주었으니 우리는 이제부터 RMSE를 적고 괄호안에 지정해준 파라미터만 넣어주게되면 우리가 원하는 값을 추출할 수 있다. 
#RMSE는 여기서 없기 때문에 직접 함수를 만들어 주어야 한다. 
#
#6. R2구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
#R2_score을 을 찾아주기 위해서는 y_test와 y_predict 를 사용하고 



#과제 : R2 가 음수로도 나올 수 있다. R2를 은수가 아닌 0.5이하로 줄이기
#레이어는 인풋과 아웃풋을 포함한 5개이상, 노드는 레이어당 각각5개이상
#batch size=1
#epochs = 100 이상
