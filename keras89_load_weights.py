#88번 파일 카피, save models,주석처리
#피팅 위에 있는 모든것들을 다 가져온것이기 때문에 일단 피팅 과 그 위에 것들을 다 주석 처리를 해준다 
#하지만 이런 model에 노란줄이 생기게 되는데 이건 모델을 주석처리 하면 안된다는 뜻이다
#모델을 풀고 돌려도 안되니 핏까지 풀어 줘야한다
#모델 컴파일은 두고 핏을 지우면 그래도 이건 구동 가능하다 우리가 웨이트 값을 가져온것이기 때문이다. 
#activation softmax가 한줄이 추가 되어도 돌아는 간다. 
#기존에 저장한 레이어의 갯수와 불러오기한 레이어의 갯수가 다르다 그래서 웨이트를 쓰려면 기존 모델과 같아야 한다 기존에 했던 모델 세이브는 조금 유도리가 있는 반면에 
#세이브 웨이트는 약간 유도리가 떨어짐으로 가중치만 저장된다. 
#모델을 저장하고 가중치를 저장하고,  모델을 불러오고 가중치를 불러오고 그 외에도 파일도 저장하고 파일도 불러 올수 있다. 
#파일을 에포와 에큐러시만을 가져왔다, h5라는 파일이름은 뭔가 있어보이는 파일명이다.
#그림 파일을 저장하게 된다면 epo값만큼 파일이 하나씩 만들어지기 때문에 우리는 가장 좋은 loss값과 acc값을 가지고 있는 모델만 쏘옷 빼올수 있다. 
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.callbacks import EarlyStopping

#1. 데이터
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
model.add(Dense(10, activation = 'softmax'))

model.summary()
print(x_train)
print(y_train)


# model.save('./model/model_test01.h5')



#3. 훈련

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam', metrics = ['acc'])
early_stopping = EarlyStopping(monitor='loss', patience='30', mode='auto' )
# model.fit(x_train, y_train, epochs = 1, batch_size = 51, validation_split=0.2, verbose=2, callbacks=[early_stopping])

# model.save('./model/model_test0t1.h5')
# model.save_weights('./model/test_weight1.h5')

model.load_weights('./model/test_weight1.h5')


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
