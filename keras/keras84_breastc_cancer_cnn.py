import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, Dense, Dropout, Input, Flatten
from keras. models import Model, Sequential




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


#train test split
x = x.reshape(x.shape[0], 1, 1 , x.shape[1])

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size = 0.8)

print(x_train.shape) #(455,1, 1, 30)
print(x_test.shape)  #(114, 1, 1, 30)
print(y_train.shape) #(455,)
print(y_test.shape)  #(114,)


# 2.  모델



# model = Sequential()

# model.add(Conv2D(30, (1,1), input_shape=( 
# 1,1, 30)))
# model.add(Flatten())
# model.add(Dense(1))
# model.add(Dense(1))
# model.add(Dense(1))

# model.summary()

# input1 = Input(shape=(1, 1,30))
# conv2d = Conv2D(32, (2, 2), activation='elu', padding='same')(input1)
# dense1_2 = Dense(240, activation='elu')(conv2d)
# drop1 = Dropout(0.2)(dense1_2)
# conv1 = Flatten()(drop1)
# dense1_2 = Dense(200, activation='elu')(conv1)
# dense1_2 = Dense(140, activation='elu')(dense1_2)
# drop1 = Dropout(0.2)(dense1_2)

# dense1_2 = Dense(80, activation='elu')(drop1)
# drop1 = Dropout(0.2)(dense1_2)

# dense1_2 = Dense(154, activation='elu')(drop1)
# dense1_2 = Dense(250, activation='elu')(dense1_2)
# drop1 = Dropout(0.2)(dense1_2)

# dense1_2 = Dense(154, activation='elu')(drop1)
# dense1_2 = Dense(250, activation='elu')(dense1_2)
# drop1 = Dropout(0.2)(dense1_2)

# dense1_2 = Dense(154, activation='elu')(drop1)
# dense1_2 = Dense(250, activation='elu')(dense1_2)
# drop1 = Dropout(0.2)(dense1_2)
# output1_2 = Dense(300, activation='elu')(drop1)
# drop1 = Dropout(0.2)(output1_2)

# output1_2 = Dense(200, activation='elu')(output1_2)
# drop1 = Dropout(0.2)(output1_2)

# output1_2 = Dense(300, activation='elu')(drop1)
# drop1 = Dropout(0.2)(output1_2)

# output1_2 = Dense(140, activation='elu')(drop1)
# drop1 = Dropout(0.2)(output1_2)

# output1_3 = Dense(1, activation='elu')(drop1)

# model = Model(inputs = input1,
#  outputs = output1_3)
# model.summary()


input1 = Input(shape=(1, 1,30))
conv2d = Conv2D(32, (2, 2), activation='elu', padding='same')(input1)
dense1_2 = Dense(24, activation='elu')(conv2d)
drop1 = Dropout(0.2)(dense1_2)
conv1 = Flatten()(drop1)
dense1_2 = Dense(20, activation='elu')(conv1)
dense1_2 = Dense(10, activation='elu')(dense1_2)
drop1 = Dropout(0.2)(dense1_2)

dense1_2 = Dense(80, activation='elu')(drop1)
drop1 = Dropout(0.2)(dense1_2)

dense1_2 = Dense(14, activation='elu')(drop1)
drop1 = Dropout(0.2)(dense1_2)

dense1_2 = Dense(54, activation='elu')(drop1)
drop1 = Dropout(0.2)(dense1_2)

dense1_2 = Dense(50, activation='elu')(drop1)
drop1 = Dropout(0.2)(dense1_2)
output1_2 = Dense(30, activation='elu')(drop1)
drop1 = Dropout(0.2)(output1_2)


output1_2 = Dense(30, activation='elu')(drop1)
drop1 = Dropout(0.2)(output1_2)

output1_2 = Dense(14, activation='elu')(drop1)
drop1 = Dropout(0.2)(output1_2)

output1_3 = Dense(1, activation='elu')(drop1)

model = Model(inputs = input1,
 outputs = output1_3)
model.summary()

# 3. 컴파일(훈련준비),실행(훈련)
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

es = EarlyStopping(monitor = 'val_loss', patience=10, mode = 'auto')

cp = ModelCheckpoint(filepath='./model/{epoch:02d}-{val_loss:.5f}.hdf5', monitor='val_loss', save_best_only=True, mode='auto')

tb = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)

model.compile(optimizer='adam',loss = 'binary_crossentropy', metrics = ['acc'])

hist = model.fit(x_train,y_train,epochs=242, batch_size=110, validation_split= 0.3, callbacks=[es, cp, tb])



### 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print('loss:', loss)
print('acc:', acc)

plt.subplot(2,1,1)
plt.plot(hist.history['loss'], c='black', label ='loss')
plt.plot(hist.history['val_loss'], c='blue', label ='val_loss')
plt.ylabel('loss')
plt.legend()

plt.subplot(2,1,2)
plt.plot(hist.history['acc'], c='blue', label ='acc')
plt.plot(hist.history['val_acc'], c='yellow', label ='val_acc')
plt.ylabel('acc')
plt.legend()

plt.show()

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



# loss: 0.32491903556020635
# acc: 0.9035087823867798