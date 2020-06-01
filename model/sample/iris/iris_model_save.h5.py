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
y= np_utils.to_categorical(y)

scaler = StandardScaler()
x = scaler.fit_transform(x)
# x_scaled = scaler.transform(x)
# print(x_scaled)


# pca = PCA(n_components=2)
# pca.fit(x_scaled)
# x_pca = pca.transform(x_scaled)
# print(x_pca)
# print(x_pca.shape)  #(150, 2)

##
#train test split
x= x.reshape(x.shape[0], 1, x.shape[1])
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size = 0.8)

print(x_train.shape) #(120, 2)
print(x_test.shape)  #(30, 2)
print(y_train.shape) #(120,)
print(y_test.shape)  #(30,)


### 2. 모델
from keras.models import Sequential
from keras.layers import Dense



from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(100, input_shape=(1, 4)))
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
model.save('./model/sample/iris/iris_model_save.h5')

# EarlyStopping
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint




### 3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor = 'loss', patience=100, mode = 'auto')

tb = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

cp = ModelCheckpoint(filepath ='./model/sample/iris/{epoch:02d}-{val_loss:.4f}.hdf5', monitor='val_loss', save_best_only=True, mode='auto')

hist =model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=1, validation_split=0.25, callbacks=[es, cp, tb])

model.save('./model/sample/iris/iris_model_save.h5')
model.save_weights('./model/sample/iris/iris_save_weights.h5')

### 4. 평가, 예측


loss_acc = model.evaluate(x_test, y_test)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc= hist.history['val_acc']


print("loss : {loss}", loss)
print("acc : {acc}", acc)
print("val_acc: ", val_acc)
print("loss_acc: ", loss_acc)



# loss, acc = model.evaluate(x_test, y_test, batch_size=32)
# print('loss:', loss)
# print('acc:', acc)

# y_predict = model.predict(x_test)
# print(y_predict)

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

# # RMSE 구하기
# from sklearn.metrics import mean_squared_error

# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# print("RMSE : ", RMSE(y_test, y_predict))

# # R2 구하기
# from sklearn.metrics import r2_score

# r2 = r2_score(y_test, y_predict)
# print("R2 : ", r2)

# #loss: 0.15265245735645294
# #acc: 0.9666666388511658