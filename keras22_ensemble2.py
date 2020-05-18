#1. 데이터
import numpy as np 
x1 = np.array([range(1,101), range(311, 411)])
x2 = np.array([range(711,811), range(711, 811)])

y1 = np.array([range(101,201), range(411, 511)])
y2 = np.array([range(501,601), range(711, 811)])
y3 = np.array([range(411,511), range(611, 711)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)
y3 = np.transpose(y3)

##########################
#####여기서부터 수정#######
##########################
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test, x2_train, x2_test, y2_train, y2_test, y3_train, y3_test = train_test_split(
    x1, y1, x2, y2,  y3, shuffle = False,
    train_size = 0.8)



#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
#model = Sequential()
#model.add(Dense(5, input_dim=3))
#model.add(Dense(4))
#model.add(Dense(1))

input1 = Input(shape=(2,  ))
dense1_1=Dense(5, activation='relu', name='Bitking1')(input1)
dense1_2=Dense(4,activation='relu', name='Bitking2')(dense1_1)



input2 = Input(shape=(2,  ))
dense2_1=Dense(5, activation='relu', name='input2_1')(input2)
dense2_2=Dense(4,activation='relu', name='input2_2')(dense2_1)

from keras.layers.merge import concatenate
merge1 = concatenate([dense1_2, dense2_2], name='concatenate')

middle1 = Dense(30, name='middle')(merge1)
middle1 = Dense(5)(middle1)
middle1 = Dense(7)(middle1)
####output모델구성######

output1_1 = Dense(30)(middle1)
output1_2 = Dense(7)(output1_1)
output1_3 = Dense(2, name='finalone')(output1_2)


output2_1 = Dense(30)(middle1)
output2_2 = Dense(7)(output2_1)
output2_3 = Dense(2, name='finaltwo')(output2_2)

output3_1 = Dense(30)(middle1)
output3_2 = Dense(7)(output3_1)
output3_3 = Dense(2, name='finalthree')(output3_2)



#input1 and input 2 will be merged into one. 
model = Model(inputs = [input1, input2],
              outputs =[output1_3, output2_3, output3_3])

model.summary()



#3. 훈련
model.compile(loss='mse', optimizer = 'adam', metrics=['mse'])
model.fit([x1_train, x2_train],
          [y1_train, y2_train, y3_train], 
          epochs=10, batch_size=1, validation_split=0.25, verbose=2)

#validation_data=(x_val, y_val))
#4. 평가와 예측
loss = model.evaluate([x1_test, x2_test], [y1_test, y2_test, y3_test], batch_size=1)

print("loss ; ", loss)

y1_predict, y2_predict, y3_predict = model.predict([x1_test, x2_test])
print("===============")
print(y1_predict)
print("===============")
print(y2_predict)
print("===============")
print(y3_predict)
print("===============")


#5.RMSE구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
RMSE3 = RMSE(y3_test, y3_predict)
print("RSME1 : ", RMSE1)
print("RSME2 : ", RMSE2) 
print("RMSE3 : ", RMSE3)
print("RMSE :", (RMSE1 + RMSE2 + RMSE3)/3)

#6. R2구하기
from sklearn.metrics import r2_score
r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)
r2_3 = r2_score(y3_test, y3_predict)
print("R2_1 : ", r2_1)
print("R2_2 : ", r2_2)
print("R2_3 : ", r2_3)
print("R2 : ", (r2_1 + r2_2 + r2_3)/3)

'''
print("x_train:", x_train)
print("x_test:", x_test)
print("y_train:", y_train)
print("y_test:", y_test)
#print("x_val :", x_val)
'''