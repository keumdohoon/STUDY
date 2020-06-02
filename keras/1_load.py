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
        y_end_number = x_end_number +y_column

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, 3]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x, y = split_xy5(samsung, 5, 1)
print(x[0,:], "/n", y[0])
print(x.shape)
print(y.shape)


#데이터셋 나누기
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state = 1, train_size= 0.8)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


#dnn에 넣어주기 위해선는 x를 3차원이 아닌 2차원으로 바꾸어 주어야 한다. 
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


#모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(2, input_shape=(25,)))
model.add(Dense(4))
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

model.compile(loss= 'mse', optimizer = 'adam', metrics= ['mse'])

from keras.callbacks import EarlyStopping
es = EarlyStopping(patience=20)
model.fit(x_train_scaled, y_train, validation_split =0.2, verbose= 1, batch_size =1 , epochs=100, callbacks=[es])

loss, mse = model.evaluate(x_test_scaled, y_test, batch_size=1)
print('loss: ', loss)
print('mse: ', mse)

y_predict = model.predict(x_test_scaled)

for i in range(5):
    print('종가 : ', y_test[i], '/예측가 :', y_predict[i])
''' 
