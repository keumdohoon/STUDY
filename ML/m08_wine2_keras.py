from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, LSTM , Flatten, Dropout, MaxPooling2D,Input
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 1. data
# wine = np.genfromtxt('./data/csv/winequality-white.csv',delimiter=';')

dataset = pd.read_csv('./data/csv/winequality-white.csv', index_col = None, header = 0, sep =  ';')

np_dataset = dataset.values

print(np_dataset.shape) #(4898, 12)

x = np_dataset[:,0:-1]
y = np_dataset[:,-1]
#d이 슬라이싱 방식은 numpy형식일때 사용 된다, 우리가 이미 numpy로 바꾸어서 정보를 가져와서 이렇게 사용할수 있다.

# one hot
y = to_categorical(y)

print(x.shape)#(4899, 11)
print(y.shape)#(4898, 10)

#scaler
scaler = MinMaxScaler()
scaler.fit(x)
x= scaler.transform(x)

x_train,x_test, y_train,y_test = train_test_split(x,y,
                                                  random_state = 33, shuffle=True,
                                                  train_size=0.8)

print(x_train.shape) # (3918, 11)
print(y_train.shape) # (3918, 10)

# 2. model
model = Sequential()
model.add(Dense(30, input_dim=11, activation='relu'))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(10, activation='softmax'))

# EarlyStopping
es = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience= 3)

# 3. compile, fit
model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])
model.fit(x_train,y_train,epochs=30,batch_size=64,callbacks=[es],verbose=2,validation_split=0.2)

# 4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test,batch_size=64)

print("loss : ",loss)
print("acc : ",acc)
