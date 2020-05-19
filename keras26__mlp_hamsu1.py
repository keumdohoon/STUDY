#keras14_mlp# Sequential 에서 함수형으로 변경
#earlystopping적용

#1. 데이터
import numpy as np 
x = np.transpose([range(1,101), range(311,411), range(100)])
y = np.transpose([range(101,201), range(711,811), range(100)])
#np.array 안에 데이터들을 []로 묶는다. 
#array 말고 transpose를 쓰게 되면 행과 열이 바뀌게 된다. 

print(x.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
x, y, shuffle=True, random_state = 66, test_size=0.2, train_size = 0.8)
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

input1 = Input(shape=(3,))
#현재 input_dim에 있는 1이라는 숫자가 Range에 있는 1~100까지가 들어가 있다는 것이다. 
dense1_1 = Dense(12)(input1)
dense1_2 = Dense(24)(dense1_1)
dense1_2 = Dense(24)(dense1_2)
dense1_2 = Dense(24)(dense1_2)
dense1_2 = Dense(24)(dense1_2)
dense1_2 = Dense(24)(dense1_2)
dense1_2 = Dense(24)(dense1_2)







output1_2 = Dense(32)(dense1_2)
output1_2 = Dense(16)(output1_2)
output1_2 = Dense(8)(output1_2)
output1_2 = Dense(4)(output1_2)
output1_3 = Dense(3, name='finalone')(output1_2)

model = Model(inputs = input1,
 outputs = output1_3)

#3. 훈련
model.compile(loss='mse', optimizer = 'adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stoppong = EarlyStopping(monitor='loss', patience=3, mode='auto')
model.fit(x_train, y_train,
          epochs=100, batch_size=1, validation_split=0.25, verbose=1, 
          callbacks=[early_stoppong])




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
