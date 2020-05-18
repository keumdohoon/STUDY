#1. 데이터
import numpy as np 
x1 = np.transpose([range(1,101), range(311,411), range(411,511)])
x2 = np.transpose([range(711,811), range(711, 811), range(511,611)])

y = np.transpose([range(101,201), range(411,511), range(100)])

##########################
#####여기서부터 수정#######
##########################
from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1, x2, y, shuffle=False,
    train_size = 0.8)


print("\nx_train\n",x1_train)
print("\nx_test\n",x1_test)
print("\nx2_train\n",x2_train)
print("\nx2_test\n",x2_test)
print("\ny_train\n",y_train)
print("\ny_test\n",y_test)

# from sklearn.model_selection import train_test_split
# y1_train, y1_test = train_test_split(
# y1, Shuffle=False, train_size = 0.8)

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
#model = Sequential()
#model.add(Dense(5, input_dim=3))
#model.add(Dense(4))
#model.add(Dense(1))

input1 = Input(shape=(3,  ))
dense1_1=Dense(5, activation='relu')(input1)
dense1_2=Dense(10,activation='relu')(dense1_1)
dense1_2=Dense(20,activation='relu')(dense1_1)
dense1_2=Dense(40,activation='relu')(dense1_1)
dense1_2=Dense(100,activation='relu')(dense1_1)
dense1_2=Dense(50,activation='relu')(dense1_1)
dense1_2=Dense(30,activation='relu')(dense1_1)
dense1_2=Dense(10,activation='relu')(dense1_1)
dense1_2=Dense(4,activation='relu')(dense1_2)



input2 = Input(shape=(3,  ))
dense2_1=Dense(5, activation='relu')(input1)
dense2_2=Dense(10,activation='relu')(dense2_1)
dense2_2=Dense(20,activation='relu')(dense2_1)
dense2_2=Dense(40,activation='relu')(dense2_1)
dense2_2=Dense(100,activation='relu')(dense2_1)
dense2_2=Dense(50,activation='relu')(dense2_1)
dense2_2=Dense(30,activation='relu')(dense2_1)
dense2_2=Dense(10,activation='relu')(dense2_1)
dense2_2=Dense(4,activation='relu')(dense2_1)

from keras.layers.merge import concatenate
merge1 = concatenate([dense1_2, dense2_2], name='concatenate')

middle1 = Dense(30, name='middle')(merge1)
middle1 = Dense(5)(middle1)
middle1 = Dense(20)(middle1)
middle1 = Dense(30)(middle1)
middle1 = Dense(10)(middle1)
middle1 = Dense(7)(middle1)
####output모델구성######

output1_1 = Dense(30)(middle1)
output1_2 = Dense(7)(output1_1)
output1_3 = Dense(3, name='finalone')(output1_2)



#input1 and input 2 will be merged into one. 
model = Model(inputs = [input1, input2],
 outputs = output1_3)

model.summary()



#3. 훈련
model.compile(loss='mse', optimizer = 'adam', metrics=['mse'])
model.fit([x1_train, x2_train],
          [y_train], 
          epochs=110, batch_size=1, validation_split=0.25, verbose=1)

#validation_data=(x_val, y_val))
#4. 평가와 예측
loss = model.evaluate([x1_test, x2_test], y_test, batch_size=1)

print("loss ; ", loss)

y1_predict = model.predict([x1_test, x2_test])
#print("===============")
#print(y1_predict)
#print("===============")
#print(y2_predict)
#print("===============")



#5.RMSE구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
RMSE1 = RMSE(y_test, y1_predict)
print("RSME1 :", RMSE1)

#6. R2구하기
from sklearn.metrics import r2_score
from keras.metrics import mse
r2_1 = r2_score(y_test, y1_predict)
print("R2_1 : ", r2_1)
'''
print("x_train:", x_train)
print("x_test:", x_test)
print("y_train:", y_train)
print("y_test:", y_test)
#print("x_val :", x_val)
'''