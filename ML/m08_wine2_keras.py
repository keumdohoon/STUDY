from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, LSTM , Flatten, Dropout, MaxPooling2D,Input
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np

# 1. data
x_data = np.load('./data/npy/x_data.npy')
y_data = np.load('./data/npy/y_data.npy')

y_data = np_utils.to_categorical(y_data)

x_train,x_test, y_train,y_test = train_test_split(x_data,y_data,
                                                  random_state = 66, shuffle=True,
                                                  train_size=0.9)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape) # 4408,11
print(y_train.shape) # 4408,10

# 2. model
model = Sequential()
model.add(Dense(30, input_dim=11, activation='relu'))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(10, activation='softmax'))

# 3. compile, fit
model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])

model.fit(x_train,y_train,epochs=30,batch_size=64,callbacks=[],verbose=2,validation_split=0.1)

# 4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test,batch_size=3)

print("loss : ",loss)
print("acc : ",acc)