import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
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
y= np_utils.to_categorical(y)

scaler = StandardScaler()
scaler.fit(x)
x = scaler.fit_transform(x)
print(x)

x= x.reshape(x.shape[0], 1, 1, 4)

###
#train test split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, shuffle=True,
    train_size = 0.8)

print(x_train.shape) #(120, 2)
print(x_test.shape)  #(30, 2)
print(y_train.shape) #(120,)
print(y_test.shape)  #(30,)


### 2. 모델
from keras.models import Sequential
from keras.layers import Dense



model = Sequential()
model.add(Conv2D(80, (1,1), activation='relu', padding='same', input_shape=(1,1,4)))
model.add(Conv2D(70, (1,1), activation='relu', padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(160, (1,1), activation='relu', padding='same'))
model.add(Conv2D(100, (1,1), activation='relu', padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(80, (1,1), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(40, (1,1), activation='relu', padding='same'))
model.add(Conv2D(30, (1,1), activation='relu', padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(30, (1,1), activation='relu', padding='same'))
model.add(Conv2D(100, (1,1), activation='relu', padding='same'))

model.add(Flatten())
# model.add(Dense(20))
# model.add(Dense(10))
model.add(Dense(3, activation='softmax'))
model.summary()


# EarlyStopping



### 3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor = 'loss', patience=100, mode = 'auto')

tb = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

cp = ModelCheckpoint(filepath ='./model/{epoch:02d}-{val_loss:.4f}.hdf5', monitor='val_loss', save_best_only=True, mode='auto')

hist = model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=1, validation_split=0.25, callbacks=[es, cp, tb])




### 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print('loss:', loss)
print('acc:', acc)

y_predict = model.predict(x_test)
print(y_predict)


plt.subplot(2,1,1)
plt.plot(hist.history['loss'], c='black', label ='loss')
plt.plot(hist.history['val_loss'], c='yellow', label ='val_loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()

plt.subplot(2,1,2)
plt.plot(hist.history['acc'], c='red', label ='acc')
plt.plot(hist.history['val_acc'], c='green', label ='val_acc')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend()

plt.show()

# # RMSE 구하기
# from sklearn.metrics import mean_squared_error

# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# print("RMSE : ", RMSE(y_test, y_predict))

# # R2 구하기
# from sklearn.metrics import r2_score

# r2 = r2_score(y_test, y_predict)
# print("R2 : ", r2)

#loss: 0.9990180134773254
#acc: 0.6333333253860474