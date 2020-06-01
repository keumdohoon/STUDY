from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train[0])
print('y_train[0] :', y_train[0]) #y_train[0] : [6]

print(x_train.shape)              #(50000, 32, 32, 3)
print(x_train.shape)              #(50000, 32, 32, 3)
print(y_train.shape)              #(50000, 1)
print(y_test.shape)               #(10000, 1)

plt.imshow(x_train[0])
plt.show()

#데이터 전처리
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)#(50000, 10)

x_train = x_train.reshape(50000, 32, 96).astype('float32') / 255
x_test = x_test.reshape(10000, 32, 96).astype('float32') / 255


#2. 모델구성

input1 = Input(shape=(32,96))
dense1 = LSTM(6)(input1)
dense2 = Dense(36)(dense1)
dense3 = Dense(126)(dense2)
dense4 = Dense(36)(dense3)

output1 = Dense(6)(dense4)
output2 = Dense(10)(output1)
model = Model(inputs = input1, outputs = output2)
model.summary()

model.save('./model/sample/cifar10/cifar10_model_save.h5')


#3. 실행
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

es = EarlyStopping(monitor = 'val_loss', patience=100, mode = 'auto')

cp = ModelCheckpoint(filepath='./model/sample/cifar10/{epoch:02d}-{val_loss:.4f}.hdf5', monitor='val_loss', save_best_only=True, mode='auto')

tb = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

### 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.25, callbacks=[es, cp, tb])
model.save('./model/sample/cifar10/cifar10_model_save.h5')
model.save_weights('./model/sample/cifar10/cifar10_model_save_weights.h5')


# #4. 예측


loss_acc = model.evaluate(x_test, y_test)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc= hist.history['val_acc']


print("loss : {loss}", loss)
print("acc : {acc}", acc)
print("val_acc: ", val_acc)
print("loss_acc: ", loss_acc)

#전에 있던 모델

# loss,acc = model.evaluate(x_test, y_test, batch_size=30)

# print("loss : {loss}", loss)
# print("acc : {acc}", acc)






