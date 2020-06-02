#1. 데이터
import numpy as np 
x1 = np.transpose([range(1,101), range(311,411), range(100)])
y1 = np.transpose([range(711,811), range(711, 811), range(100)])

x2 = np.transpose([range(101,201), range(411,511), range(100, 200)])
y2 = np.transpose([range(501,601), range(711, 811), range(100)])


print(x1.shape)
'''
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
x1, y1, shuffle=False, test_size=0.2, train_size = 0.8)
#x1train 은 80,3 이고 x1test는 20,3이다.           
from sklearn.model_selection import train_test_split 
x2_train, x2_test, y2_train, y2_test = train_test_split(
x2, y2, shuffle=False, test_size=0.2, train_size = 0.8)




#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
#model = Sequential()
#model.add(Dense(5, input_dim=3))
#model.add(Dense(4))
#model.add(Dense(1))

input1 = Input(shape=(3,  ))
dense1_1=Dense(5, activation='relu', name='Bitking1')(input1)
dense1_2=Dense(4,activation='relu', name='Bitking2')(dense1_1)
dense1_2=Dense(4,activation='relu', name='Bitking3')(dense1_2)



input2 = Input(shape=(3,  ))
dense2_1=Dense(5, activation='relu', name='input2_1')(input2)
dense2_2=Dense(4,activation='relu', name='input2_2')(dense2_1)
dense2_2=Dense(4,activation='relu', name='input2_3')(dense2_2)

from keras.layers.merge import concatenate
merge1 = concatenate([dense1_2, dense2_2], name='concatenate')

middle1 = Dense(30, name='middle')(merge1)
middle1 = Dense(5)(middle1)
middle1 = Dense(7)(middle1)
####output모델구성######
output1_1 = Dense(30)(middle1)
output1_2 = Dense(7)(output1_1)
output1_3 = Dense(3, name='finalone')(output1_2)


output2_1 = Dense(30)(middle1)
output2_2 = Dense(7)(output2_1)
output2_3 = Dense(3, name='finaltwo')(output2_2)




#input1 and input 2 will be merged into one. 
model = Model(inputs = [input1, input2], outputs =[output1_3, output2_3])

model.summary()




#3. 훈련
model.compile(loss='mse', optimizer = 'adam', metrics=['mse'])
model.fit([x1_train, x2_train],
          [y1_train, y2_train], 
          epochs=10, batch_size=1, validation_split=0.25, verbose=1)

#validation_data=(x_val, y_val))

#4. 평가와 예측
loss = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)
print("loss ; ", loss)

y1_predict, y2_predict = model.predict([x1_test, x2_test])
print("===============")
print(y1_predict)
print("===============")
print(y2_predict)
print("===============")



#5.RMSE구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)

print("RSME1 :", RMSE1)
print("RSME2 :", RMSE2)
print("RMSE :", (RMSE1+RMSE2)/2)

#6. R2구하기
from sklearn.metrics import r2_score
from keras.metrics import mse
r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)
print("R2_1 : ", r2_1 )
print("R2_2 : ", r2_2)
print("R2 : ", (r2_1 + r2_2)/2)

print("x_train:", x_train)
print("x_test:", x_test)
print("y_train:", y_train)
print("y_test:", y_test)
#print("x_val :", x_val)
'''