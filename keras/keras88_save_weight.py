#85번 파일 카피
#모델 세이브할(model.save)때 모델만 저장되는 것이 아니라
#save data를 모델 부분밑에 한번이랑 model fit밑에 한번 두번해주고 model밑에 있는것 및애를 주석 퍼리해준다. 
#67번 68번 파일을 확인해볼것 weight값만 따로 저장해주는 새로운 무언가가있다. 
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0])
print('y_train :',y_train[0])  #y_train : 5
print(x_train.shape)  #(60000, 28, 28)
print(x_test.shape)   #(10000, 28, 28)
print(y_train.shape)  #(60000,)
print(y_test.shape)  #(10000,)

print(x_train[0].shape)  #(28, 28)
plt.imshow(x_train[0], 'gray')

#데이터 전처리
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)#(60000,10)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255
print("x.shape", x_train.shape)  #(60000, 28, 28, 1)
print("y.shape", y_train.shape)  #(60000, 10)

#2. 모델링


model = Sequential()
model.add(Conv2D(10, (2,2), input_shape= (28,28,1)))   #(9,9,10)
model.add(Conv2D(15, (3,3))) ##(7,7,7)
model.add(Conv2D(25, (2,2), padding = 'same'))       #(7,7,5)
model.add(Conv2D(35, (2,2), padding = 'same'))      
model.add(Conv2D(45, (2,2), padding = 'same'))      
model.add(Conv2D(55, (2,2), padding = 'same'))      
model.add(Conv2D(45, (2,2)))       
model.add(Conv2D(35, (2,2)))       
model.add(Conv2D(25, (2,2), padding = 'same'))       #(7,7,5)

model.add(Conv2D(15, (2,2)))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))
model.summary()
# print(x_train)
# print(y_train)


# model.save('./model/model_test01.h5')


#3. 훈련

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam', metrics = ['acc'])
early_stopping = EarlyStopping(monitor='loss', patience='30', mode='auto' )
model.fit(x_train, y_train, epochs = 1, batch_size = 51, validation_split=0.2, verbose=2, callbacks=[early_stopping])

model.save('./model/model_test0t1.h5')
model.save_weights('./model/test_weight1.h5')

#4. 예측
loss_acc = model.evaluate(x_test, y_test)

loss = ['loss']
val_loss = ['val_loss']
acc = ['acc']
val_acc= ['val_acc']


print("loss : {loss}", loss)
print("acc : {acc}", acc)
print("val_acc: ", val_acc)
print("loss_acc: ", loss_acc)

# loss : {loss} ['loss']
# acc : {acc} ['acc']
# val_acc:  ['val_acc']
# loss_acc:  [0.13199199991859495, 0.9628000259399414]
