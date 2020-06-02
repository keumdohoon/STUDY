import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

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
plt.show()