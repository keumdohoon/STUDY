import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint


(x_train, y_train), (x_test, y_test) = mnist.load_data()
 #x_train, y_train, x_test, y_test를 반환해 준다.

print(x_train[0])
 #x의 0번째를 한번본다
print('y_train :',y_train[0])
 #y_train : 5
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
 #(60000, 28, 28)
 #(10000, 28, 28)
 #(60000,)#60000개의 스칼라를 가진 디멘션하나짜리
 #(10000,)

print(x_train[0].shape)
plt.imshow(x_train[0], 'gray')
 #plt.imshow(x_train[0])
# plt.show()

#데이터 전처리
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)#(60000,10)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255
 #reshape로 4차원을 만들어준 다음에(CNN모델을 집어넣기 위해서, 가로 새로 채널), 
 #astype은 파일을 변환해준다는 것이다 원래는 정수형태로(각 픽셀마다 255중에 255면 완전 찐한 검정색)데이터가 0부터 255까지가 들어가 있는데
 #정수형태로 들어가 있다. 하지만 우리가 넣으려고 하는 minmax는 0부터 1까지인 실수이기때문에 float를 정수에서 실수로 바꾸어주게 된다. 
 # #나누기 255는 0부터 1까지로 나누어주기위해서 그 사이를 255개로 쪼개 주는 것이다. 이것이 정규화이다.  
 #255로 나누게 되면 최댓값이 1이 되고 최소값이 0이 된다. 

 #print(x_train)
 #print(x_test)
 #print(y_train.shape)
 #print(y_test.shape)


print("x.shape", x_train.shape) 
# x.shape (60000, 28, 28, 1)
print("y.shape", y_train.shape)  
# y.shape (60000, 10)


#2. 모델링
model = Sequential()
model.add(Flatten(input_shape= (28,28,1)))   #(9,9,10)
model.add(Dense(15, activation ='relu'))
model.add(Dense(10, activation ='relu'))
model.add(Dropout(0.2))


model.add(Dense(10, activation ='relu')) 
model.add(Dense(15, activation ='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation = 'softmax'))
model.summary()
print(x_train)
print(y_train)

# model.save('./model/sample/mnist/mnits_model_save.h5')
#결과값: accuracy: 0.9932

#3. 훈련

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam', metrics = ['acc'])
modelpath = './model/sample/mnist/{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True,save_weights_only=False, mode='auto')
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto' )
hist = model.fit(x_train, y_train, epochs = 30, batch_size = 51, validation_split=0.2, verbose=2, callbacks=[cp, early_stopping])
model.save('./model/sample/mnist/mnits_model_save.h5')
model.save_weights('./model/sample/mnist/mnits_save_weights.h5')

#4. 평가와 예측
loss_acc = model.evaluate(x_test, y_test)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc= hist.history['val_acc']


print("loss : {loss}", loss)
print("acc : {acc}", acc)
print("val_acc: ", val_acc)
print("loss_acc: ", loss_acc)

# loss,acc = model.evaluate(x_test,y_test,batch_size=30)

# print(f"loss : {loss}")
# print(f"acc : {acc}")
