import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# import some data

dataset = load_breast_cancer()
print(dataset)
print(dataset.keys())
print(dataset['feature_names'])

x = dataset.data
y = dataset.target





from sklearn.preprocessing import MinMaxScaler
print(x.shape) 
 # (569, 30)
print(y.shape) 
 # (569,)

scaler = MinMaxScaler()
x = scaler.fit_transform(x)
print(x)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size = 0.8)

print(x_train.shape) #(455, 30)
print(x_test.shape)  #(114, 30)
print(y_train.shape) #(455,)
print(y_test.shape)  #(114,)
'''

### 2. 모델
from keras.models import Sequential
from keras.layers import Dense


model = Sequential()

model.add(Dense(100, input_shape= (30, )))#dnn모델이기에 위에서 가져온 10이랑 뒤에 ',' 가 붙는다. 
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(400))
model.add(Dense(500))
model.add(Dense(400))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(1))

model.summary()

# EarlyStopping
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

es = EarlyStopping(monitor = 'val_loss', patience=100, mode = 'auto')

cp = ModelCheckpoint(filepath='./model/{epoch:02d}-{val_loss:.4f}.hdf5', monitor='val_loss', save_best_only=True, mode='auto')

tb = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

### 3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=1, validation_split=0.25, callbacks=[es, cp, tb])


### 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=32)
print('loss:', loss)
print('mse:', mse)

plt.subplot(2,1,1)
plt.plot(hist.history['loss'], c='black', label ='loss')
plt.plot(hist.history['val_loss'], c='blue', label ='val_loss')
plt.ylabel('loss')
plt.legend()

plt.subplot(2,1,2)
plt.plot(hist.history['acc'], c='blue', label ='lmse')
plt.plot(hist.history['val_acc'], c='blue', label ='val_acc')
plt.ylabel('acc')
plt.legend()

y_predict = model.predict(x_test)
print(y_predict)


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