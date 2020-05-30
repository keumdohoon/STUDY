import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

# import some data

dataset = load_iris()
print(dataset)
print(dataset.keys())
print(dataset['feature_names'])

x = dataset.data
y = dataset.target

print(x.shape) # (150, 4)
print(y.shape) # (150,)
# y= np_utils.to_categorical(y)

# scaler = StandardScaler()
# scaler.fit(x)
# x_scaled = scaler.transform(x)
# print(x_scaled)


# pca = PCA(n_components=2)
# pca.fit(x_scaled)
# x_pca = pca.transform(x_scaled)
# print(x_pca)
# print(x_pca.shape)  #(150, 2)

##
#train test split

x_train, x_test, y_train, y_test = train_test_split(
    # x_pca
    x, y, random_state=66, shuffle=True,
    train_size = 0.8)

print(x_train.shape) #(120, 2)
print(x_test.shape)  #(30, 2)
print(y_train.shape) #(120,)
print(y_test.shape)  #(30,)


### 2. 모델
from keras.models import Sequential
from keras.layers import Dense


x_train = x_train.reshape(120, 2, 2)
x_test = x_test.reshape(30, 2, 2)

from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(100, input_shape=(4,)))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(400))
model.add(Dense(500))
model.add(Dense(400))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(3, activation='softmax'))


model.summary()

# EarlyStopping
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

es = EarlyStopping(monitor = 'loss', patience=100, mode = 'auto')
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto')

tb_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

### 3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train,
          epochs=3, batch_size=32, verbose=1,
          validation_split=0.25,
          callbacks=[es, checkpoint, tb_hist])


### 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=32)
print('loss:', loss)
print('mse:', mse)

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
