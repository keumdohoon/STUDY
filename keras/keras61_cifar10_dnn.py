from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train[0])
print('y_train[0] :', y_train[0]) #y_train[0] : [6]

print(x_train.shape)              #(50000, 32, 32, 3)
print(x_test.shape)               #(10000, 32, 32, 3)
print(y_train.shape)              #(50000, 1)
print(y_test.shape)               #(10000, 1)




plt.imshow(x_train[0])
plt.show()

x_train= x_train.reshape(x_train.shape[0],3072 )

print('x_train:', x_train)
 # [[ 59  62  63 ... 123  92  72]
 #  [154 177 187 ... 143 133 144]
 #  [255 255 255 ...  80  86  84]
 #  ...
 #  [ 35 178 235 ...  12  31  50]
 #  [189 211 240 ... 195 190 171]
 #  [229 229 239 ... 163 163 161]]
print('x_train_shape: ', x_train.shape)
 # (50000, 3072)


#데이터 전처리
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)#(50000, 10)

x_train = x_train.reshape(50000, 3072,).astype('float32') / 255
x_test = x_test.reshape(10000, 3072,).astype('float32') / 255

#2. 모델링
input1 = Input(shape=(3072,))
dense1_1 = Dense(12)(input1)
dense1_2 = Dense(24)(dense1_1)
dense1_2 = Dense(24)(dense1_2)
dense1_2 = Dense(24)(dense1_2)
dense1_2 = Dense(24)(dense1_2)
dense1_2 = Dense(24)(dense1_2)
dense1_2 = Dense(24)(dense1_2)

output1_2 = Dense(32)(dense1_2)
output1_2 = Dense(16)(output1_2)
output1_2 = Dense(8)(output1_2)
output1_2 = Dense(4)(output1_2)
output1_3 = Dense(10)(output1_2)

model = Model(inputs = input1,
 outputs = output1_3)

#3. 훈련
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 15, batch_size = 50, verbose= 2)


#acc75프로로 잡아라
#4. 예측
loss,acc = model.evaluate(x_test,y_test,batch_size=30)

print(f"loss : {loss}")
print(f"acc : {acc}")

















