#95번을 불러와서 96번 모델 완성.

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

x = np.load('./data/iris_data.npy')
print(x)

x_data = x[ : , 0:4]
y_data = x[ : , 4]


y= np_utils.to_categorical(y_data)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y, random_state=66, shuffle=True,
    train_size = 0.8)

print(x_train.shape) # (120, 4)
print(x_test.shape) # (30, 4)
print(y_train.shape) # (120,3)
print(y_test.shape) # (30,3)

# y_train= np_utils.to_categorical(y_train)
# y_test= np_utils.to_categorical(y_test)

### 2. 모델
from keras.models import Sequential
from keras.layers import Dense, Dropout
 #모델은 sequential을 사용해준다. 

model = Sequential()

model.add(Dense(400, input_shape= (4, )))
model.add(Dense(400, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation= 'relu'))
model.add(Dense(400, activation= 'relu'))
model.add(Dense(800, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(400, activation= 'relu'))
model.add(Dense(400, activation= 'relu'))

model.add(Dense(80, activation = 'softmax'))
model.add(Dropout(0.3))

model.add(Dense(3))

model.summary()

# EarlyStopping


### 3. 훈련
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
es = EarlyStopping(monitor = 'loss', patience=200, mode = 'auto')
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
tb_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_test, y_test, epochs=500, batch_size=32, validation_split=0.25,
          callbacks=[es, checkpoint, tb_hist])


### 4. 평가, 예측

loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print('loss:', loss)
print('acc:', acc)

# plt.subplot(2,1,1)
# plt.plot(hist.history['loss'], c='black', label ='loss')
# plt.plot(hist.history['val_loss'], c='yellow', label ='val_loss')
# plt.ylabel('loss')
# plt.xlabel('epochs')
# plt.legend()

# plt.subplot(2,1,2)
# plt.plot(hist.history['acc'], c='red', label ='acc')
# plt.plot(hist.history['val_acc'], c='green', label ='val_acc')
# plt.ylabel('acc')
# plt.xlabel('epochs')
# plt.legend()

# plt.show()



y_predict = model.predict(x_test)
print(y_predict)
