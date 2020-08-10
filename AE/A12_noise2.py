# 202008050915
# 원본과 노이즈와 아웃풋
#CNN으로 ㅇ오토인코딩을 만드시오 노이즈 제거 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose
import numpy as np
import tensorflow as tf
# def autoencoder(hidden_layer_size):
#     model = Sequential()
#     model.add(Dense(units = hidden_layer_size, input_shape=(784, ), activation = 'relu'))
#     model.add(Dense(units = 784, activation = 'sigmoid'))

#     return model


def autoencoder():
    model = Sequential()
    model.add(Conv2D(filters = 128, kernel_size=(3,3),
                                    padding = 'valid', input_shape=(28,28,1),
                                    activation = 'relu')),
    model.add(Conv2DTranspose(filters = 1, kernel_size = (3,3),
                                                padding = 'valid', activation = 'sigmoid'))
    return model



from tensorflow.keras.datasets import mnist

train_set, test_set = mnist.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1],x_train.shape[2],1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1],x_test.shape[2],1))
x_train = x_train/255.
x_test = x_test/255.


######################
#add some noise#####
# 입력 데이터에 노이즈 생성 ; 픽셀에 랜덤하게 0을 뿌려준다
# 문제가 있다 ; / 255.를 해줌으로써 현재 데이터의 분포는 0 ~ 1 사이에 있음
# 평균이 0, 표준편차가 0.5면 0 ~ 1 사이의 범위를 벗어날 수 있음

x_train_noised = x_train + np.random.normal(0,0.1,size = x_train.shape)
#0과 0.5에 따른 정규 분포를 알려줘라
x_test_noised = x_test + np.random.normal(0,0.1,size = x_test.shape)
#평균이 0 표준편차가 0.5에 음수가 들어갈 수가 있다. 그래서 우리가 원하지 않는 음수가 나오지 않게 하기 위해서 0~1사이로 지정해줄 것이다. 
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

model = autoencoder()

##########################



# model.compile(optimizer = 'adam', loss = 'mse', metrics = ['acc'])                
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])  

model.fit(x_train_noised,x_train_noised, epochs = 10)

output = model.predict(x_test_noised)

import matplotlib.pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),(ax11, ax12, ax13, ax14, ax15)) = \
    plt.subplots(3, 5, figsize = (20, 7))

# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i ==0:
        ax.set_ylabel('INPUT', size = 40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i ==0:
        ax.set_ylabel('Noise', size = 40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
                                                                        

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i ==0:
        ax.set_ylabel('OUTPUT', size = 40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()                                                                         
