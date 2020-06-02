#1. 데이터
import numpy as np 
x = np.transpose([range(1,101), range(311,411), range(100)])
y = np.transpose(range(711,811))
#np.array 안에 데이터들을 []로 묶는다. 
#array 말고 transpose를 쓰게 되면 행과 열이 바뀌게 된다. 

print(x.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
x, y, shuffle=False, test_size=0.2, train_size = 0.8)
# (x,y, random_state = 66, 

#열우선 행무시

# x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 66, test_size = 0.4 )

# x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, shuffle = False, test_size = 0.5) # test_size의 default 값은 0.25

#print("x_train:", x_train)
#print("x_test:", x_test)
#print("x_val:", x_val)




#x_train= x[:60]
#x_val= x[60:80]
#x_test= x[80:] 

#y_train= x[:60]
#y_val= x[60:80]
#y_test= x[80:]
#x_pred = np.array([16,17,18])
#range 함수를 쓰게 되면 마지막 숫자에서 -1개 된다. 즉 (range(1,101))이면 1부터 100까지만을 센다는 것이다. 




#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
#model = Sequential()
#model.add(Dense(5, input_dim=3))
#model.add(Dense(4))
#model.add(Dense(1))

input1 = Input(shape=(3,  ))
dense1=Dense(5, activation='relu')(input1)
dense1=Dense(10,)(dense1)
dense1=Dense(20,)(dense1)
dense1=Dense(40,)(dense1)
dense1=Dense(80,)(dense1)
dense1=Dense(160,)(dense1)
dense1=Dense(320,)(dense1)
dense1=Dense(160,)(dense1)
dense1=Dense(80,)(dense1)
dense1=Dense(40,)(dense1)
dense1=Dense(20,)(dense1)
dense1=Dense(10,)(dense1)
dense1=Dense(5,)(dense1)
dense1=Dense(3, activation='relu')(dense1)



output1 = Dense(1)(dense1)


model = Model(inputs=input1, outputs=output1)
#소스에 관해서 명시해준다, input1에서부터output1까지가 함수형 모델이다. 
model.summary()

#3. 훈련
model.compile(loss='mse', optimizer = 'adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=120, batch_size=1, validation_split=0.25, verbose=2)


#validation_data=(x_val, y_val))
#4. 평가와 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("loss ; ", loss)
print("mse : ", mse)

# y_pred = model.predict(x_pred)
# print("y_pred : ", y_pred)

y_predict = model.predict(x_test)
print("y_predict:", y_predict)

#5.RMSE구하기
from sklearn.metrics import mean_squared_error
 #RMSE 함수화 => 캐글 및 대회에서 정확도 지수로 많이 사용
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RSME:", RMSE(y_test, y_predict))

#6. R2구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


print("x_train:", x_train)
print("x_test:", x_test)
print("y_train:", y_train)
print("y_test:", y_test)
#print("x_val :", x_val)
