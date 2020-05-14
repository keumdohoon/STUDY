#1. 데이터
import numpy as np 
x = np.array(range(1,101))
y = np.array(range(101,201))
#weight는 1bias는 100
from keras.models import Sequential
model = Sequential()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
x, y, random_state = 66, shuffle=True, 
train_size = 0.6)

x_val, x_test, y_val, y_test = train_test_split(
x_test, y_test, random_state = 66, shuffle=True, 
test_size=0.2)

#test_size = 0.2 는 20번연습하겠다는 것이다. 하겠다는 뜻이다. train size가 80이라는 뜻이다.  
print("x_train:", x_train)
print("x_test:", x_test)
print("x_val :", x_val)




model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=110, batch_size=1, 
validation_data=(x_val, y_val))
'''
print("x_train:", x_train)
print("x_test:", x_test)
print("y_train:", y_train)
print("y_test:", y_test)
#x_train= x[:60]
#x_val= x[60:80]
#x_test= x[80:]

#y_train= x[:60]
#y_val= x[60:80]
#y_test= x[80:]
#range 함수를 쓰게 되면 마지막 숫자에서 -1개 된다. 즉 (range(1,101))이면 1부터 100까지만을 센다는 것이다. 



#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(3, input_dim=1))
#현재 input_dim에 있는 1이라는 숫자가 Range에 있는 1~100까지가 들어가 있다는 것이다. 
model.add(Dense(6))
model.add(Dense(12))
model.add(Dense(24))
model.add(Dense(48))
model.add(Dense(96))
model.add(Dense(192))

model.add(Dense(96))
model.add(Dense(48))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=110, batch_size=1, 
validation_data=(x_val, y_val))

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
'''