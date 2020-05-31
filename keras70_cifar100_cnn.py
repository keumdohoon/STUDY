#cnn모델

from keras.datasets import cifar100
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint,  TensorBoard

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train[0])
print('y_train[0] :', y_train[0]) #y_train[0] : [19]

print(x_train.shape)  #(50000, 32, 32, 3)
print(x_train.shape)  #(50000, 32, 32, 3)
print(y_train.shape)  #(50000, 1)
print(y_test.shape)   #(10000, 1)

plt.imshow(x_train[0])
plt.show()


# 데이터 전처리
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)  #(50000, 100)
print(y_test.shape)   #(10000, 100) 




x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/ 255.0
x_test = x_test.reshape(10000, 32, 32, 3).astype('float32')/ 255.0


#2. 모델


input1= Input(shape= (32,32,3))
Conv2d1 = Conv2D(filters= 20, kernel_size=3, padding="same", activation="relu")(input1)
drop1 = Dropout(0.1)(Conv2d1)
Conv2d2= Conv2D(filters= 40, kernel_size=3, padding="same", activation="relu")(Conv2d1) 

drop2 = Dropout(0.5)(Conv2d2)



Conv2d6= Conv2D(filters= 20, kernel_size=3, padding="same", activation="relu")(drop2) 
drop4 = Dropout(0.2)(Conv2d6)

Conv2d7= Conv2D(filters= 60, kernel_size=3, padding="same", activation="relu")(drop4) 
pool3 =MaxPooling2D(pool_size=3)(Conv2d7)
drop5 = Dropout(0.2)(pool3)

Conv2d8= Conv2D(filters= 20, kernel_size=3, padding="same", activation="relu")(drop5) 
drop6 = Dropout(0.2)(Conv2d8)

output1 = (Flatten())(drop6)
output2 = Dense(100, activation = 'softmax')(output1)
model = Model(inputs=input1, outputs=output2)
model.summary()

tb_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

#3. 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = "rmsprop", metrics = ['acc'])
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto' )

hist = model.fit(x_train, y_train, epochs = 45, validation_split= 0.2, batch_size=42, callbacks=[early_stopping])

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


import matplotlib.pyplot as plt
 #그래프를 그려주는 것을  plt라고 하겠다

plt.figure(figsize=(10,6))


plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker ='.', c='red', label ='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label= 'val_loss')

plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'])


plt.subplot(2,1,2)
plt.plot(hist.history['acc'], marker ='.', c='red', label ='acc')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label= 'val_acc')
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])
plt.show()

#loss_acc:  [2.7696992603302, 0.3425999879837036]