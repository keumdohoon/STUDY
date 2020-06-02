import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout,Input
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
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
print("y.shape", y_train.shape)  


#2. 모델링
# model = Sequential()
# input1= Input(shape= (28,28,1))
# Conv2d1 = Conv2D(filters= 15, kernel_size=2, padding="same", activation="relu")(input1)
# Conv2d2= Conv2D(filters= 15, kernel_size=2, padding="same", activation="relu")(Conv2d1) 
# model.add(Dropout(0.2))

# Conv2d3= Conv2D(filters= 15, kernel_size=2, padding="same", activation="relu")(Conv2d2) 
# Conv2d4= Conv2D(filters= 15, kernel_size=2, padding="same", activation="relu")(Conv2d3) 
# model.add(Dropout(0.2))

# Conv2d5= Conv2D(filters= 10, kernel_size=2, padding="same", activation="relu")(Conv2d4) 
# Conv2d6= Conv2D(filters= 15, kernel_size=2, padding="same", activation="relu")(Conv2d5) 
# model.add(Dropout(0.2))

# output1 = (Flatten())(Conv2d6)
# output2 = Dense(10, activation = 'softmax')(output1)
# model = Model(inputs=input1, outputs=output2)
# model.summary()
# print(x_train)
# print(y_train)

#결과값 - 6s - loss: 0.0049 - accuracy: 0.9985

# input1 = Input(shape=(28,28,1))
# fl1 = (Flatten())(input1)
# dense1 = Dense(560,activation='relu')(fl1)

model = Sequential()
input1= Input_dim = (784)
dense1 = Dense(15, activation='relu')(input1)
dense1 = Dense(10)(dense1)
model.add(Dropout(0.2))

dense1 = Dense(10)(dense1)
dense1 = Dense(15)(dense1)
model.add(Dropout(0.2))

output1 = Dense(10, activation = 'softmax')(dense1)
model = Model(inputs=input1, outputs=output1)
model.summary()
print(x_train)
print(y_train)

#결과값: accuracy: 0.9932

#3. 훈련

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 15, batch_size = 50, verbose= 2)


#y값이 0부터9까지인 수이다. 그 거를 60000가지고 있고 이를 셰이프 시키면 (60000,)
#그리고 여기서 우리는 OneHotencoding을 해줌으로 인해서 
0,1,2,3,4,5,6,7,8,9
#0=1,0,0,0,0,0,0,0,0,0
#1=0,1,0,0,0,0,0,0,0,0
#2=0,0,1,0,0,0,0,0,0,0
#3=0,0,0,1,0,0,0,0,0,0
#4=0,0,0,0,1,0,0,0,0,0
#....이러한 식으로 간다.

#accuracy: 0.9971