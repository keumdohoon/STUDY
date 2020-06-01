#85번 파일 카피
#모델 세이브할(model.save)때 모델만 저장되는 것이 아니라
#model save의 피팅한 다음에 save를 하게 되니 우리가 fit한 결과까지 저장이 되었다. 이 뜻은 가중치까지 저장이 되었다는 뜻이다. 
#세이브한 모델의 파일은 계속 같은 파일 위에 덮어쓰기가 된다. 
#85번 파일의 세이브 한것을 86번 파일에 빈 부분에 붙인다. 
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






# #3. 훈련

#4. 평과와 예측
loss_acc = model.evaluate(x_test, y_test)
print('loss, acc :', loss_acc)

loss = ['loss']
val_loss = ['val_loss']
acc = ['acc']
val_acc= ['val_acc']


print("loss : {loss}", loss)
print("acc : {acc}", acc)
print("val_acc: ", val_acc)
print("loss_acc: ", loss_acc)
# loss, acc : [0.12585525950184093, 0.9617000222206116]
# loss : {loss} ['loss']
# acc : {acc} ['acc']
# val_acc:  ['val_acc']
# loss_acc:  [0.12585525950184093, 0.9617000222206116]

#

