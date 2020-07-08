##뉴스 기사를 가지고 46개의 카테고리로 나누어지는 예제이다. 
from keras.datasets import reuters, imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

(x_train, y_train), (x_test, y_test) =imdb.load_data(num_words=1000)

print(x_train.shape, x_test.shape) #(25000,) (25000,)
print(y_train.shape, y_test.shape) #(25000,) (25000,)


print(x_train[0])#[1, 14, 22, 16, 43, 530, 973, 2, 2, 65, 458, 2, 66, 2, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 2, 2, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 
# 147, 2, 19, 14, 22, 4, 2, 2, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 2, 4, 22, 17, 515, 17, 12, 16, 626, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2, 2, 16, 480, 66, 2, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 2, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 2, 15, 256, 4, 2, 7, 2, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 2, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2, 56, 
# 26, 141, 6, 194, 2, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 2, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 2, 88, 12, 16, 283, 5, 16, 2, 113, 103, 32, 15, 16, 2, 19, 178, 32]
print(y_train[0]) #1


print(len(x_train[0]))  #218
#이것들 데이터의 갯수는 일정하지 않다고 우리는 판단할수 있다 하지만 우리는 이제 이것의 와꾸를 맞춰주어야 한다는 결과를 도출 해 낼 수 있다. 

#이를 처리해 주기 위해서 우리는 pad_sequence를 사용하여 앞뒤의 빈자리를 채워주면서 맞춰 줄 수 있다는 것이다. 
#y가 총 몇개인지 모르고 있는 상황이다. 명확하게 여기서 확인을 해볼것이다. 


#y의 카테고리 갯수 출력
category = np.max(y_train) + 1
print("카테고리 :" , category)  #카테고리 :2

#y의 유니크한 값들 출력 
y_scatter = np.unique(y_train)
print(y_scatter)  #[0 1]

y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count()
print(bbb)
# 0    12500
# 1    12500
# Name: 0, dtype: int64


print(bbb.shape) #(2,)

#주간과제: groupby() 의 사용법 숙지할것
##################################################################################
# x_train에 대한 크기를 맞춰줘야한다.

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical  

x_train = pad_sequences(x_train, maxlen=111, padding='pre')
x_test = pad_sequences(x_test, maxlen=111, padding='pre')


print(x_train.shape, x_test.shape) #(25000, 111) (25000, 111)


#모델
from keras.models import Sequential
from keras.layers import Dense, MaxPooling1D, Embedding, LSTM, Flatten, Conv1D, Bidirectional

model = Sequential()
# model.add(Embedding(2000,128, input_length = 111))
model.add(Embedding(2000, 128))

model.add(Conv1D(64, 5, padding = 'valid', activation= 'relu', strides = 1))
model.add(MaxPooling1D(pool_size =4))


# model.add(Bidirectional(LSTM(64)))
model.add(LSTM(64))

model.add(Dense(1, activation= 'sigmoid'))

model.summary()

'''
model.compile(loss = 'binary_crossentropy', optimizer ='adam'
                    , metrics= ['acc'])
history = model.fit(x_train, y_train, batch_size=100, epochs=10)

acc = model.evaluate(x_test, y_test)[-1]
print("acc: ", acc)
# acc:  0.8199999928474426


#그림을 그리자

y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker='.', c= 'red', label = 'TestSet Loss')
plt.plot(y_loss, marker='.', c= 'red', label = 'TestSet Loss')
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

#첫번째 해야 할것imdb데이터 내용을 확인, 데이터 구조 파악학기 
#y값과 x값이 몇바이 몇인지도 확인하기.
# #word_size 전체 데이터 부분  변경해서 최상값 확인. 
#주관과제 groupby()의 사용법 숙지할것 
#단어를 숫자로 바꾸고 숫자를 단어로 바꾸어라 키랑 벨류 값 바꾸기


_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, None, 128)         256000
_________________________________________________________________
conv1d_1 (Conv1D)            (None, None, 64)          41024
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, None, 64)          0
_________________________________________________________________
lstm_1 (LSTM)                (None, 64)                33024
_________________________________________________________________
dense_1 (Dense)              (None, 46)                2990
=================================================================
Total params: 333,038
Trainable params: 333,038
Non-trainable params: 0
_________________________________________________________________
'''