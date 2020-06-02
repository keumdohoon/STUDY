#save된 파일 불러오기
import numpy as np
import pandas as pd

kospi = np.load('./data/kospi.npy',allow_pickle=True) 
samsung = np.load('./data/samsung.npy',allow_pickle=True)

print(kospi)
print(samsung)
print(kospi.shape)
print(samsung.shape)

#그림데이터 자르기


def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, 0]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x1, y1 = split_xy5(samsung, 5, 1)
x2, y2 = split_xy5(kospi, 5, 1)
print(x2[0,:], "/n", y2[0])
print(x2.shape)#(503, 5, 5)
print(y2.shape)#(503, 1)
print(x1.shape)#(507, 1, 1)
print(y1.shape)#(507, 1)
'''
#앙상블 모델로 해줄때에는 데이터셋을 2개 구축해준다. 
x1, y1 = split_xy5(samsung, 5, 1)
x2, y2 = split_xy5(kospi, 5, 1)
print(x2[0,:], "/n", y2[0])
print(x2.shape)
print(y2.shape)




#데이터셋 나누기
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state = 1, train_size= 0.8)

#앙상블 모델일때 데이터를 2개 만들어준것
# from sklearn.model_selection import train_test_split
# #from sklearn.model_selection import cross_val_score
# x1_train, x1_test, y1_train, y1_test = train_test_split(
#     x1, y1, random_state = 1, train_size= 0.8)
# from sklearn.model_selection import train_test_split
# #from sklearn.model_selection import cross_val_score
# x2_train, x2_test, y2_train, y2_test = train_test_split(
#     x2, y2, random_state = 1, train_size= 0.8)    



print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# #dnn에 넣어주기 위해선는 x를 3차원이 아닌 2차원으로 바꾸어 주어야 한다. 
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
print(x_train.shape)
print(x_test.shape)

# (336, 25)
# (85, 25)
###w중요###데이터 전처리
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(x_train_scaled[0,:])

print(x_train.shape)
print(x_test.shape)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] ,1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] , 1))
print(x_train.shape)
print(x_test.shape)

#앙상블 모델 reshaep
# x1_train = np.reshape(x1_train, (x1_train.shape[0], x1_train.shape[1] * x1_train.shape[2]))
# x1_test = np.reshape(x1_test, (x1_test.shape[0], x1_test.shape[1] * x1_test.shape[2]))
# print(x1_train.shape)
# print(x1_test.shape)

# x2_train = np.reshape(x2_train, (x2_train.shape[0], x2_train.shape[1] * x2_train.shape[2]))
# x2_test = np.reshape(x2_test, (x2_test.shape[0], x2_test.shape[1] * x2_train.shape))
# print(x2_train.shape)
# print(x2_test.shape)



###앙상블모델 데이터 전처리
# scaler1 = StandardScaler()
# scaler1.fit(x1_train)
# x1_train_scaled = scaler.transform(x1_train)
# x1_test_scaled = scaler.transform(x1_test)
# print(x1_train_scaled[0,:])
# scaler2 = StandardScaler()
# scaler2.fit(x2_train)
# x2_train_scaled = scaler.transform(x2_train)
# x2_test_scaled = scaler.transform(x2_test)
# print(x2_train_scaled[0,:])

#앙상블 모델 리셰이프 한번더 해주는거
# x1_train_scaled = np.reshape(x1_train_scaled, (x1_train_scaled.shape[0], x1_train_scaled.shape[1] ,1))
# x1_test_scaled = np.reshape(x1_test_scaled, (x1_test_scaled.shape[0], x1_test_scaled.shape[1] , 1))
# print(x1_train_scaled.shape)
# print(x1_test_scaled.shape)

# x2_train_scaled = np.reshape(x2_train_scaled, (x2_train_scaled.shape[0], x2_train_scaled.shape[1] ,1))
# x2_test_scaled = np.reshape(x2_test_scaled, (x2_test_scaled.shape[0], x2_test_scaled.shape[1] , 1))
# print(x2_train_scaled.shape)
# print(x2_test_scaled.shape)

# print(x2_train_scaled.shape)
# print(x2_test_scaled.shape)


#모델 구성
# from keras.models import Sequential, Model
# from keras.layers import Dense, LSTM,Input


# model = Sequential()
# input1 = Input(shape=(25,1))
# dense1 = LSTM(6)(input1)
# dense2 = Dense(36)(dense1)
# dense3 = Dense(126)(dense2)
# dense4 = Dense(36)(dense3)

# output1 = Dense(6)(dense4)
# output2 = Dense(1, name='finalone')(output1)
# model = Model(inputs = input1, outputs = output2)
# model.summary()


#앙상블 모델
# input1 = Input(shape= (25,1))
# dense1_1=LSTM(120)(input1)
# dense1_2=Dense(240)(dense1_1)
# dense1_3=Dense(480)(dense1_2)


# input2 = Input(shape=(25,1))
# dense2_1=LSTM(120)(input2)
# dense2_2=Dense(240)(dense2_1)
# dense2_3=Dense(480)(dense2_2)


# from keras.layers.merge import concatenate
# merge1 = concatenate([dense1_3, dense2_3])

# middle1 = Dense(960)(merge1)
# middle2 = Dense(1920)(middle1)
# middle3 = Dense(960)(middle2)

# ####output모델구성######
# output1_1 = Dense(480)(middle3)
# output1_2 = Dense(240)(output1_1)
# output1_3 = Dense(1)(output1_2)
# #input1 and input 2 will be merged into one. 
# model = Model(inputs = [input1, input2], outputs = output1_3)
# model.summary()


model.compile(loss= 'mse', optimizer = 'adam', metrics= ['mse'])

from keras.callbacks import EarlyStopping
es = EarlyStopping(patience=20)
model.fit(x_train, y_train, validation_split =0.2, verbose= 1, batch_size =1 , epochs=100, callbacks=[es])

# from keras.callbacks import EarlyStopping
# es = EarlyStopping(patience=20)
# model.fit([x1_train_scaled,x2_test_scaled], y1_train, validation_split =0.2, verbose= 1, batch_size =1 , epochs=100, callbacks=[es])





loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('loss: ', loss)
print('mse: ', mse)
# loss, mse = model.evaluate([x1_train_scaled,x2_test_scaled], y1_test, batch_size=1)
# print('loss: ', loss)
# print('mse: ', mse)



y_predict = model.predict(x_test)

# y1_predict = model.predict([x1_train_scaled,x2_test_scaled])


for i in range(5):
    print('종가 : ', y_test[i], '/예측가 :', y_predict[i])

#  for i in range(5):
#         print('종가 : ', y1_test[i], '/예측가 :', y1_predict[i])   
'''