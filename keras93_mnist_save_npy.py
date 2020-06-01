#91번 파일 카피
#92번 파일에 있는거를 가져오는데 이를 전처리하기전인 단계에 붙여넣기 해줘야 전처리가 되어서 이 폴더에 쓸수 있게 된다. 하지만 만약 어차피 여기서 한번더 shape를 바꿔줘야한다면 전처리 한후에 이ㅆ는곳에 저장해서 전처리를 직접 한번 시켜주면 된다. 
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

np.save('./data/mnist_train_x.npy', arr=x_train)
np.save('./data/mnist_test_x.npy', arr=x_test)
np.save('./data/mnist_train_y.npy', arr=y_train)
np.save('./data/mnist_test_y.npy', arr=y_test)


#arr 이라는 어레이를 (x_train)을 이경로의 파일명('./data/mnist_train_x.npy')으로 저장하겠다

#

# print(x_train[0].shape)  #(28, 28)
# plt.imshow(x_train[0], 'gray')
