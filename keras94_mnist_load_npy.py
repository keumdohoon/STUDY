#93번 파일 카피
#93번파일에서 복사를 해 준것이기에 94번의 부럴오기부터 전처리전까지의 데이터를 싹 지우거나 주석 처리해준다. 결국 93번은 세이브 이후부터의 내용은 필요없기 때문에 세이브 이후의 것들은 다 지워준다. 
#93번은 데이터 전처리까지의 부분을 사용하고 94번에서는 데이터 전처리 이후의 정보를 사용해준다. 93번에서 위쪽을 쓰고 94번에서는 밑에 쪽을 쓰고.
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint

x_train = np.load('./data/mnist_train_x.npy')
x_test = np.load('./data/mnist_test_x.npy')
y_train = np.load('./data/mnist_train_y.npy')
y_test = np.load('./data/mnist_test_y.npy')
#save했던 데이터를 불러와주는 과정인데 93번에서 있었던 arr는 지워주고 (그거는 그 데이터를 저장한다는거니까) load를 한 다음에는 앞에 명칭을 붙여줘야지 된다.
#명칭은 현재 폴더에 하위에 있는 이름들로 통일시켜주면 소스를 그대로 사용할수 있으니 편리해진다. 
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


# model = Sequential()
# model.add(Conv2D(10, (2,2), input_shape= (28,28,1)))   #(9,9,10)
# model.add(Conv2D(15, (3,3))) ##(7,7,7)
# model.add(Conv2D(25, (2,2), padding = 'same'))       #(7,7,5)
# model.add(Conv2D(35, (2,2), padding = 'same'))      
# model.add(Conv2D(45, (2,2), padding = 'same'))      
# model.add(Conv2D(55, (2,2), padding = 'same'))      
# model.add(Conv2D(45, (2,2)))       
# model.add(Conv2D(35, (2,2)))       
# model.add(Conv2D(25, (2,2), padding = 'same'))       #(7,7,5)

# model.add(Conv2D(15, (2,2)))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(10, activation = 'softmax'))
# model.summary()
# print(x_train)
# print(y_train)


# model.save('./model/model_test01.h5')


#3. 훈련

# model.compile(loss = 'categorical_crossentropy',
#               optimizer = 'adam', metrics = ['acc'])
# modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
# cp = ModelCheckpoint(filepath='./model/{epoch:02d}-{val_loss:.4f}.hdf5', monitor='val_loss', save_best_only=True,save_weights_only=False, mode='auto')

# early_stopping = EarlyStopping(monitor='loss', patience='30', mode='auto' )
# hist = model.fit(x_train, y_train, epochs = 30, batch_size = 51, validation_split=0.2, verbose=2, callbacks=[cp, early_stopping])
# model.save('./model/model_test01.h5')
from keras.models import load_model
model = load_model('./model/15-0.0739.hdf5')

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
#여기서 hist를 안해주는 이유는 우리가 가져온 정보자체가 이미 hist를 내장하고 있고 여기서는 결과값만 보려고 하는 것이기에 hist를 지워준다는 것이다. 
# loss_acc:  [0.07477523971106274, 0.9775999784469604]





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
