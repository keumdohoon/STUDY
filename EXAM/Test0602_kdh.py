#save된 파일 불러오기
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM,Input
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

hite = np.load('./data/hite.npy',allow_pickle=True)  
samsung_electronics = np.load('./data/samsung_electronics.npy',allow_pickle=True)

# print(hite)
# print(samsung_electronics) # 


# print(hite.shape) # 508,5
# print(samsung_electronics.shape) #508,1

# hite = hite[hite[0::,4].astype(np.float)

# value =  hite[0::,4]

# print(value)
# value =  samsung_electronics[0::,0]

# print(value)


# hite = float(hite)
# print('Float Value =', hite)


# hite = hite.reshape(508,5).astype('float64')

# print(hite)


# a = hite
# a= str(a)
# print(a)



#그림데이터 자르기



def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :]
        print(tmp_x)
        print(tmp_x.shape)
        tmp_y = dataset[x_end_number:y_end_number, 0]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x1, y1 = split_xy5(samsung_electronics, 1, 1)

'''
x2, y2 = split_xy5(hite, 5, 1)
print(x2[0,:], "/n", y2[0])
print('y2',y2.shape)#(508, 1)
print('y1',y1.shape)#(508,)

print(x2[0,:], "/n", y2[1])
print(x2.shape)#(503, 5, 5)
print(y2.shape)#(508, 1)
print(x1.shape)#(508,)
print(y1.shape)#((508,)

#데이터셋 나누기
#앙상블 모델일때 데이터를 2개 만들어준것
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, random_state = 1, train_size= 0.7)

from sklearn.model_selection import train_test_split
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, random_state = 1, train_size= 0.7)    

print(x1_train.shape)#(354, 1, 1)
print(y1_train.shape)#(354, 1)
print(x1_test.shape)#(153, 1, 1)
print(y1_test.shape)#(153, 1)
print(type(x1_train))#(354, 1, 1)
print(type(y1_train))#(354, 1)
print(type(x1_test))#(153, 1, 1)
print(type(y1_test))#(153, 1)

print(x2_train.shape)#(352, 5, 5)
print(y2_train.shape)#(352, 1)
print(x2_test.shape)#(151, 5, 5)
print(y2_test.shape)#151, 1)

#앙상블 모델 reshaep standard scaler 에 넣어주기위해서
x1_train = np.reshape(x1_train, (x1_train.shape[0], x1_train.shape[1] * x1_train.shape[2]))
x1_test = np.reshape(x1_test, (x1_test.shape[0], x1_test.shape[1] * x1_test.shape[2]))
print(x1_train.shape)#(354, 1)
print(x1_test.shape)#(153, 1)

x2_train = np.reshape(x2_train, (x2_train.shape[0], x2_train.shape[1] * x2_train.shape[2]))
x2_test = np.reshape(x2_test, (x2_test.shape[0], x2_test.shape[1] * x2_test.shape[2]))
print(x2_train.shape)#(352, 25)
print(x2_test.shape)#(151, 25)


# print('x1_train:', x1_train)
# print('x2_train:',x2_train)
# x2_train= x2_train.transpose(x2_train.shape)
# print('x2_train:',x2_train)


##앙상블모델 데이터 전처리
scaler1 = StandardScaler()
scaler1.fit(x1_train)
x1_train_scaled = scaler1.transform(x1_train)
x1_test_scaled = scaler1.transform(x1_test)
print(x1_train_scaled[0,:])#[-0.33883351]


scaler2 = StandardScaler()
scaler2.fit(x2_train)
x2_train_scaled = scaler2.transform(x2_train)
x2_test_scaled = scaler2.transform(x2_test)
print(x2_train_scaled[0,:])
#[-0.77928323 -0.79287587 -0.75621517 -0.75031352 -0.96556911 -0.73876723
# -0.77121413 -0.75451362 -0.78511039 -0.25322398 -0.78481737 -0.81559385
#  -0.76455983 -0.77666071 -0.99302187 -0.78271267 -0.80658407 -0.75226172
#  -0.79274759 -1.02628082 -0.79292457 -0.83290068 -0.80932192 -0.84868563
#   0.07362774]

# scaler2 = StandardScaler()
# scaler2.fit(x2_train)
# x2_train_scaled = scaler2.transform(x2_train)
# x2_test_scaled = scaler2.transform(x2_test)
# print(x2_train_scaled[0,:])

#앙상블 모델 리셰이프 한번더 해주는거 LSTM모델에 넣어주기 위해서 
x1_train_scaled = np.reshape(x1_train_scaled, (x1_train_scaled.shape[0], x1_train_scaled.shape[1] ,1))
x1_test_scaled = np.reshape(x1_test_scaled, (x1_test_scaled.shape[0], x1_test_scaled.shape[1] , 1))
print(x1_train_scaled.shape)
print(x1_test_scaled.shape)
# (354, 1, 1)
# (153, 1, 1)

x2_train_scaled = np.reshape(x2_train_scaled, (x2_train_scaled.shape[0], x2_train_scaled.shape[1] ,1))
x2_test_scaled = np.reshape(x2_test_scaled, (x2_test_scaled.shape[0], x2_test_scaled.shape[1] , 1))
print(x2_train_scaled.shape)
print(x2_test_scaled.shape)

# (352, 25, 1)
# (151, 25, 1)


#모델 구성
from keras.models import Sequential
from keras.layers import Dense


#앙상블 모델 함수
input1 = Input(shape= (1,1))
dense1_1=LSTM(120)(input1)
dense1_2=Dense(240)(dense1_1)
dense1_3=Dense(480)(dense1_2)


input2 = Input(shape=(25,1))
dense2_1=LSTM(120)(input2)
dense2_2=Dense(240)(dense2_1)
dense2_3=Dense(480)(dense2_2)


from keras.layers.merge import concatenate
merge1 = concatenate([dense1_3, dense2_3])

middle1 = Dense(960)(merge1)
middle2 = Dense(1920)(middle1)
middle3 = Dense(960)(middle2)

####output모델구성######
output1_1 = Dense(480)(middle3)
output1_2 = Dense(240)(output1_1)
output1_3 = Dense(1)(output1_2)
#input1 and input 2 will be merged into one. 
model = Model(inputs = [input1, input2], outputs = output1_3)
model.summary()


model.compile(loss= 'mse', optimizer = 'adam', metrics= ['mse'])
from keras.callbacks import EarlyStopping
es = EarlyStopping(patience=20)
model.fit([x1_train_scaled,x2_test_scaled], y1_train, 
            validation_split =0.2, verbose= 1, batch_size =1 , epochs=100, callbacks=[es])


loss, mse = model.evaluate([x1_train_scaled,x2_test_scaled], y1_test, batch_size=1)
print('loss: ', loss)
print('mse: ', mse)

y1_predict = model.predict([x1_train_scaled,x2_test_scaled])

for i in range(5):
        print('시가 : ', y1_test[i], '/예측가 :', y1_predict[i]) 
'''