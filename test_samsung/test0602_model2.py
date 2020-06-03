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


#데이터를 자르게 되면 차원을 맞춰줘야하기때문에 Dense모델로 쓰게 될거란면 여기서 한번 리쉐이프를 해주는 것이 좋다. 

samsung =samsung.reshape(samsung.shape[0],)
print(samsung.shape) #(509,)




#자르기
samsung = (split_x(samsung, size))
print(samsung.shape) #(504, 6)
#여기서 6은 6일치씩 자른다는 뜻이다. 
#삼성만 x와 y를 분리해주면 된다.

x_sam = samsung[:,0:5]
#이거는 전체 데이터에서 , 0에서 5까지의 숫자를 가진것들로 잘라준다는 뜻이다 여기서 5는 위에 6일치씩 잘라준다는 것에서 인덱스6이니 0에서 5까지가 되는것이다. 
y_sam = samsung[:,5]

print(x_sam.shape)#(504, 5)
print(y_sam.shape)#(504,)

x_hit = hite[5:510,:]
print(x_hit.shape)  #(504, 5)

#x_hit버려야 할 데이터랑 남겨야할 데이터 제일 오래된 데이터를 없애주는 것이다. 
# x_sam = x_sam.reshape(504,5,1) LSTM사용할때만 다시 시쉐이프해준다


#2. 모델구성

input1 = Input(shape= (5,))
x1 = Dense(10)(input1)
x1 = Dense(10)(x1)

input2 = Input(shape= (5,))
x2= Dense(5)(input2)
x2 = Dense(5)(x2)

merge = concatenate([x1,x2])

output = Dense(1)(merge)


model = Model(inputs = [input1, input2], outputs = output)

model.summary()

#3.컴파일, 훈련
model.compile(optimizer = 'adam', loss= 'mse')
model.fit([x_sam, x_hit], y_sam, epochs= 5)
