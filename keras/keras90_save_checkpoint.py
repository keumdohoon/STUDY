#85번 파일 카피
#모델 세이브할(model.save)때 모델만 저장되는 것이 아니라
#save data를 모델 부분밑에 한번이랑 model fit밑에 한번 두번해주고 model밑에 있는것 및애를 주석 퍼리해준다. 
#체크 포인트를 가져와주고 파일명(65번라인)도 추가해준다. 여기서 우리는 웨이트만 가져오는게 아니라 다른것들도 가져오기때문에  체크 포인트에 save_weights_only는 false로 해준다. 
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint

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
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True,save_weights_only=False, mode='auto')

early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto' )
hist = model.fit(x_train, y_train, epochs = 30, batch_size = 51, validation_split=0.2, verbose=2, callbacks=[cp, early_stopping])
model.save('./model/model_test01.h5')
model.save_weights('./model/test_weight1.h5')

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
# loss_acc:  [0.0995692705052148, 0.9753999710083008]

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
