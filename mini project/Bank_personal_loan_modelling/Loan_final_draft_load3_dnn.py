import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

#error fixed by putting in header and indexcol as 0

import os
# print(os.listdir("./input"))

# original = pd.read_excel('./Bank_personal_loan_modelling.xlsx',"loans")
x_data = pd.read_csv('./data/csv/loan_traintest_data.csv', index_col= 0, header= 0)
x_pred = pd.read_csv('./data/csv/loan_prediction_data.csv',  index_col= 0, header= 0)

print(x_data.shape) #(5000, 13)
print(x_pred.shape) #(5000, 12)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_data = scaler.fit_transform(x_data)
print(x_data)

x = x_data[:, :12]
y = x_data[:, -1:]

print(x.shape)
print(y.shape)#(5000, 1)


x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=66, shuffle=True, train_size = 0.8)

print(x_train.shape) #(4000, 14)
print(x_test.shape)  #(1000, 14)
print(y_train.shape) #(4000, 13)
print(y_test.shape)  #(1000, 13)


### 2. 모델
# from keras.models import Sequential
from keras.layers import Dense


# model = Sequential()

# model.add(Dense(100, input_shape= (12, )))#dnn모델이기에 위에서 가져온 10이랑 뒤에 ',' 가 붙는다. 
from keras.layers import Input, Dropout
from keras.models import Model

print(x_train.shape)
print(y_train.shape)
#2. 모델링
input1 = Input(shape=(12,))
dense1_1 = Dense(120, activation='elu')(input1)
dense1_2 = Dense(240, activation='elu')(dense1_1)
drop1 = Dropout(0.2)(dense1_2)

dense1_2 = Dense(200, activation='elu')(drop1)
dense1_2 = Dense(140, activation='elu')(dense1_2)
drop1 = Dropout(0.2)(dense1_2)

dense1_2 = Dense(80, activation='elu')(drop1)
drop1 = Dropout(0.2)(dense1_2)

dense1_2 = Dense(154, activation='elu')(drop1)
dense1_2 = Dense(250, activation='elu')(dense1_2)
drop1 = Dropout(0.2)(dense1_2)
output1_2 = Dense(300, activation='elu')(drop1)
output1_2 = Dense(200, activation='elu')(output1_2)
drop1 = Dropout(0.2)(output1_2)

output1_2 = Dense(300, activation='elu')(drop1)
drop1 = Dropout(0.2)(output1_2)

output1_2 = Dense(140, activation='elu')(drop1)
drop1 = Dropout(0.2)(output1_2)

output1_3 = Dense(1, activation= 'sigmoid')(drop1)

model = Model(inputs = input1,
 outputs = output1_3)
model.summary()

# EarlyStopping
# from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

# es = EarlyStopping(monitor = 'val_loss', patience=100, mode = 'auto')

# cp = ModelCheckpoint(filepath='./model/{epoch:02d}-{val_loss:.4f}.hdf5', monitor='val_loss', save_best_only=True, mode='auto')

# tb = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

### 3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.25) #callbacks=[es, cp, tb])


### 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print('loss:', loss)
print('acc:', acc)

plt.subplot(2,1,1)
plt.plot(hist.history['loss'], c='black', label ='loss')
plt.plot(hist.history['val_loss'], c='blue', label ='val_loss')
plt.ylabel('loss')
plt.legend()

plt.subplot(2,1,2)
plt.plot(hist.history['acc'], c='black', label ='acc')
plt.plot(hist.history['val_acc'], c='blue', label ='val_acc')
plt.ylabel('acc')
plt.legend()
plt.show()
'''
y_predict = model.predict(x_pred)
print(y_predict)
# print(y_predict.shape) #(5000, 1)
# print(y_test)
# print(y_test.shape) #(1000, 1)
 

# RMSE 구하기
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
'''