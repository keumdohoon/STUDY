import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, Dropout, Flatten, Dense
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Flatten
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
plt.imshow(x_train[0])
#plt.show()


print(x_train[0])
print('y_train[0] :', y_train[0])
 #y_train[0] : 9
print(x_train.shape)
 #(60000, 28, 28)
print(x_test.shape)
 #(10000, 28, 28)
print(y_train.shape)
 # (60000,)
print(y_test.shape)
 #(10000,)

# 데이터 전처리
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)
 #(60000, 10)
print(y_test.shape)
 #(10000, 10)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/ 255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/ 255

#2. 모델
model = Sequential()
model.add(Conv2D(filters=10, kernel_size=2, activation='relu', padding='same', input_shape= (28,28,1)))
model.add(Conv2D(20, kernel_size=3, activation='relu', padding='same'))

model.add(Conv2D(filters= 25, kernel_size= 3, padding= 'same', activation= 'elu'))
model.add(Conv2D(filters= 15, kernel_size= 2, padding= 'same', activation= 'elu'))
model.add(Dropout(0.2))

model.add(Conv2D(filters= 10, kernel_size= 3, padding = 'same', activation= 'elu'))
model.add(Conv2D(filters= 20, kernel_size= 2, padding = 'same', activation= 'elu'))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()
model.save('./model/sample/fashion/fashion_model_save.h5')

print(x_train)
print(y_train)

#3. 훈련
model.save('./model/sample/fashion/fashion_model_save.h5')
model.save_weights('./model/sample/fashion/fashion_save_weights.h5')


from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

es = EarlyStopping(monitor = 'val_loss', patience=100, mode = 'auto')

cp = ModelCheckpoint(filepath='./model/sample/fashion/{epoch:02d}-{val_loss:.4f}.hdf5', monitor='val_loss', save_best_only=True, mode='auto')

tb = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

### 3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=10, batch_size=42, verbose=1, validation_split=0.2, callbacks=[es, cp, tb])
model.save('./model/sample/fashion/fashion_model_save.h5')
model.save_weights('./model/sample/fashion/fashion_save_weights.h5')

#4, 예측


loss_acc = model.evaluate(x_test, y_test)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc= hist.history['val_acc']


print("loss : {loss}", loss)
print("acc : {acc}", acc)
print("val_acc: ", val_acc)
print("loss_acc: ", loss_acc)

##############################
#####loss:0.3068034695446491
#####acc :0.9024999737739563#######
##################################