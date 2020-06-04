#LSTM2개 구현

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.utils import to_categorical, np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Input, Flatten, Dropout, Input
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA


def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([j for j in subset])
    #print(type(aaa))
    return np.array(aaa)

size = 6

#1. 데이터
#. npy 불러오기
hite = np.load('./data/hite.npy',allow_pickle=True)  
samsung = np.load('./data/samsung.npy',allow_pickle=True)

print(samsung.shape)#(509, 1)
print(hite.shape)#(509, 5)
print(samsung)


#데이터를 자르게 되면 차원을 맞춰줘야하기때문에 Dense모델로 쓰게 될거란면 여기서 한번 리쉐이프를 해주는 것이 좋다. 

# samsung =samsung.reshape(samsung.shape[0],)
# print(samsung.shape) #(509,)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    hite, samsung, random_state=66, shuffle=True,
    train_size = 0.8)
#하이트랑 삼성을 tts를 통해서 x와y로 분리시킨다. 
# print(x_train)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(hite)
hite_scaled = scaler.transform(hite)
# print(hite_scaled)

from sklearn.decomposition import PCA

pca = PCA(n_components=1)
pca.fit(hite_scaled)
hite_pca = pca.transform(hite_scaled)
# print(hite_pca)
#하이트는 현재 열이 5개이고 삼성은 하나인데 pca를 통해서 하이트의 열을 압축해준다. n_component가 총 몇개로 압축시킬건지를 알려주는것


#자르기
samsung = (split_x(samsung, size))
print('samsung',samsung)
print(samsung.shape) #(504, 6)
#여기서 6은 6일치씩 자른다는 뜻이다. 
#삼성만 x와 y를 분리해주면 된다.

x_sam = samsung[:,0:5]
#이거는 전체 데이터에서 , 0에서 5까지의 숫자를 가진것들로 잘라준다는 뜻이다 여기서 5는 위에 6일치씩 잘라준다는 것에서 인덱스6이니 0에서 5까지가 되는것이다. 
y_sam = samsung[:,5]

print(x_sam.shape)#(504, 5)
print(y_sam.shape)#(504,)

x_hit = hite_pca[5:510,:]
print(x_hit.shape)  #(504, 1)

#x_hit버려야 할 데이터랑 남겨야할 데이터 제일 오래된 데이터를 없애주는 것이다. 
x_sam = x_sam.reshape(504, 5,1) #LSTM사용할때만 다시 시쉐이프해준다
x_hit = x_hit.reshape(504,1,1)

print(x_sam.shape)
print(x_hit.shape)

#2. 모델구성

input1 = Input(shape= (5,1))
dense1_1=LSTM(120)(input1)
dense1_2=Dense(240)(dense1_1)
dense1_3=Dense(480)(dense1_2)


input2 = Input(shape=(1,1))
dense2_1=LSTM(120)(input2)
dense2_2=Dense(240)(dense2_1)
dense2_3=Dense(480)(dense2_2)


from keras.layers.merge import concatenate
merge1 = Concatenate(axis=1)([dense1_3, dense2_3])

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

#3.컴파일, 훈련
model.compile(optimizer = 'adam', loss= 'mse', metrics= ['mse'])
model.fit([x_sam, x_hit], y_sam, validation_split= 0.2, epochs= 101)

'''