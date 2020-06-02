import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = mnist.load_data()
 #x_train, y_train, x_test, y_test를 반환해 준다.

print(x_train[0])
 #x의 0번째를 한번본다
print('y_train :',y_train[0])  #y_train : 5
 #y_train : 5
print(x_train.shape)  #(60000, 28, 28)
print(x_test.shape)   #(10000, 28, 28)
print(y_train.shape)  #(60000,)
print(y_test.shape)  #(10000,)
 #(60000, 28, 28)
 #(10000, 28, 28)
 #(60000,)#60000개의 스칼라를 가진 디멘션하나짜리
 #(10000,)

print(x_train[0].shape)  #(28, 28)
plt.imshow(x_train[0], 'gray')
 #plt.imshow(x_train[0])
 #plt.show()

#데이터 전처리
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)#(60000,10)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255


#print(x_train)
#print(x_test)
#print(y_train.shape)
#print(y_test.shape)


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
print(x_train)
print(y_train)

#결과값: 0.9945

#3. 훈련

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam', metrics = ['acc'])
early_stopping = EarlyStopping(monitor='loss', patience='30', mode='auto' )
hist = model.fit(x_train, y_train, epochs = 1, batch_size = 51, validation_split=0.2, verbose=2, callbacks=[early_stopping])

#4. 예측
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
 #나는 그림을 2장 그리겠다. 2행 1열에 그림을 그리겠다. 2행1열의 1번째 그림을 사용하겠다. 
plt.plot(hist.history['loss'], marker ='.', c='red', label ='loss')
 #y값만 넣었다는 뜻이다 , loss값만을 하겠다. hist에 history에 로스 값만을 하겠다. 
# plt.plot(hist.history['acc'])
 #y값만 넣었다는 뜻이다 , loss값만을 하겠다. hist에 history에 로스 값만을 하겠다. 
plt.plot(hist.history['val_loss'], marker='.', c='blue', label= 'val_loss')
 #y값만 넣었다는 뜻이다 , loss값만을 하겠다. hist에 history에 로스 값만을 하겠다. 
# plt.plot(hist.history['val_acc'])
 #y값만 넣었다는 뜻이다 , loss값만을 하겠다. hist에 history에 로스 값만을 하겠다. 

 #plt.plot(hist.history['val_loss'])#validation을 지금 지정안해뒀기 때문에 빼주는것
plt.grid()
plt.title('loss')
 #제목
plt.ylabel('loss')
 #y라벨
plt.xlabel('epoch')
 #x라벨
# plt.legend(['loss', 'val_loss'])
plt.legend(['loss= upper right'])

#plt.show()

plt.subplot(2,1,2)
 #나는 그림을 2장 그리겠다. 2행 1열에 그림을 그리겠다. 2행1열의 1번째 그림을 사용하겠다. 
plt.plot(hist.history['acc'])
 #y값만 넣었다는 뜻이다 , loss값만을 하겠다. hist에 history에 로스 값만을 하겠다. 
 # plt.plot(hist.history['acc'])
 #y값만 넣었다는 뜻이다 , loss값만을 하겠다. hist에 history에 로스 값만을 하겠다. 
plt.plot(hist.history['val_acc'])
 #y값만 넣었다는 뜻이다 , loss값만을 하겠다. hist에 history에 로스 값만을 하겠다. 
 # plt.plot(hist.history['val_acc'])
 #y값만 넣었다는 뜻이다 , loss값만을 하겠다. hist에 history에 로스 값만을 하겠다. 

 #plt.plot(hist.history['val_loss'])#validation을 지금 지정안해뒀기 때문에 빼주는것
plt.grid()
plt.title('acc')
 #제목
plt.ylabel('acc')
 #y라벨
plt.xlabel('epoch')
 #x라벨
plt.legend(['acc', 'val_acc'])
plt.show()
