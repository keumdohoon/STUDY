#86번 파일 카피
#로드한 파일 밑에 3개의 레이어를 추가해준다. 
#모델핏까지 저장되어있는 데이터를 가져와서 레이어를 추가해준 상태가 된다. 
#가중치까지 같이 가져오기때문에 문제 자체가 오류가 있다. 그래서 만약 가중치를 가져오게 된다면 위에 모델이 있어야한다. 
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.callbacks import EarlyStopping
from keras.models import load_model

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
model = load_model('./model/model_test01.h5')
model.add(Dense(20))
model.add(Dense(25))
model.add(Dense(10, activation = 'softmax'))
model.summary()


model.summary

# #3. 훈련

#4. 평과와 예측
loss_acc = model.evaluate(x_test, y_test)

loss = ['loss']
val_loss = ['val_loss']
acc = ['acc']
val_acc= ['val_acc']


print("loss : {loss}", loss)
print("acc : {acc}", acc)
print("val_acc: ", val_acc)
print("loss_acc: ", loss_acc)


# import matplotlib.pyplot as plt
#  #그래프를 그려주는 것을  plt라고 하겠다

# plt.figure(figsize=(10,6))


# plt.subplot(2,1,1)
#  #나는 그림을 2장 그리겠다. 2행 1열에 그림을 그리겠다. 2행1열의 1번째 그림을 사용하겠다. 
# plt.plot(hist.history['loss'], marker ='.', c='red', label ='loss')
#  #y값만 넣었다는 뜻이다 , loss값만을 하겠다. hist에 history에 로스 값만을 하겠다. 
# # plt.plot(hist.history['acc'])
#  #y값만 넣었다는 뜻이다 , loss값만을 하겠다. hist에 history에 로스 값만을 하겠다. 
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label= 'val_loss')
#  #y값만 넣었다는 뜻이다 , loss값만을 하겠다. hist에 history에 로스 값만을 하겠다. 
# # plt.plot(hist.history['val_acc'])
#  #y값만 넣었다는 뜻이다 , loss값만을 하겠다. hist에 history에 로스 값만을 하겠다. 

#  #plt.plot(hist.history['val_loss'])#validation을 지금 지정안해뒀기 때문에 빼주는것
# plt.grid()
# plt.title('loss')
#  #제목
# plt.ylabel('loss')
#  #y라벨
# plt.xlabel('epoch')
#  #x라벨
# # plt.legend(['loss', 'val_loss'])
# plt.legend(['loss= upper right'])

# #plt.show()

# plt.subplot(2,1,2)
#  #나는 그림을 2장 그리겠다. 2행 1열에 그림을 그리겠다. 2행1열의 1번째 그림을 사용하겠다. 
# plt.plot(hist.history['acc'])
#  #y값만 넣었다는 뜻이다 , loss값만을 하겠다. hist에 history에 로스 값만을 하겠다. 
#  # plt.plot(hist.history['acc'])
#  #y값만 넣었다는 뜻이다 , loss값만을 하겠다. hist에 history에 로스 값만을 하겠다. 
# plt.plot(hist.history['val_acc'])
#  #y값만 넣었다는 뜻이다 , loss값만을 하겠다. hist에 history에 로스 값만을 하겠다. 
#  # plt.plot(hist.history['val_acc'])
#  #y값만 넣었다는 뜻이다 , loss값만을 하겠다. hist에 history에 로스 값만을 하겠다. 

#  #plt.plot(hist.history['val_loss'])#validation을 지금 지정안해뒀기 때문에 빼주는것
# plt.grid()
# plt.title('acc')
#  #제목
# plt.ylabel('acc')
#  #y라벨
# plt.xlabel('epoch')
#  #x라벨
# plt.legend(['acc', 'val_acc'])
# plt.show()

# #


