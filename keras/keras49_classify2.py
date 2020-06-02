# 1. 모듈 임포트
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

# from sklearn.preprocessing import OneHotEncoder
import numpy as np

# 1-1. 객체 생성
es = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)
# enc = OneHotEncoder()


# 2. 데이터
x = np.array(range(1, 11))
y = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])#.reshape(-1, 1)

# enc.fit(y)
# y = enc.transform(y).toarray()
y = to_categorical(y) #y를 1과 0으로 표현해주는것 
# print(y)
# [[0. 1. 0. 0. 0. 0.] 얘는 1
#  [0. 0. 1. 0. 0. 0.] 애는 2
#  [0. 0. 0. 1. 0. 0.] 애는 3
#  [0. 0. 0. 0. 1. 0.] 애는 4
#  [0. 0. 0. 0. 0. 1.] 애는 5
#  [0. 1. 0. 0. 0. 0.] 얘는 1
#  [0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 0. 1.]]

y = y[:, 1:]  #이게 앞에 앞자리에 0들을 지워주는거
# print(y)
# [[1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [0. 0. 1. 0. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1.]
#  [1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [0. 0. 1. 0. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1.]]

print(x) 
#[ 1  2  3  4  5  6  7  8  9 10]
print(x.shape)
#(10,)
print(y)
# [[1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [0. 0. 1. 0. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1.]
#  [1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [0. 0. 1. 0. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1.]]
print(y.shape)
#(10, 5)

# 3. 모델 구성
input1 = Input(shape = (1, ))
dense1 = Dense(3, activation = 'relu')(input1)
dense2 = Dense(6, activation = 'relu')(dense1)
dense3 = Dense(12, activation = 'relu')(dense2)
dense4 = Dense(24, activation = 'relu')(dense3)
dense5 = Dense(48, activation = 'relu')(dense4)
dense6 = Dense(96, activation = 'relu')(dense5)
dense7 = Dense(192, activation = 'relu')(dense6)
dense8 = Dense(96, activation = 'relu')(dense7)
dense8 = Dense(48, activation = 'relu')(dense7)
dense8 = Dense(24, activation = 'relu')(dense7)
dense8 = Dense(12, activation = 'relu')(dense7)

dense9 = Dense(6, activation = 'relu')(dense8)

output1 = Dense(3)(dense9)
output2 = Dense(5, activation = 'softmax')(output1)

model = Model(inputs = input1, outputs = output2)

model.summary()


# 4. 컴파일 및 훈련
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam', metrics = ['accuracy'])
model.fit(x, y, epochs = 122, batch_size = 1)


# 5. 평가 및 예측
loss, acc = model.evaluate(x, y, batch_size = 1)
print("loss : ", loss)
print("acc : ", acc)

x_pred = np.array([1, 2, 3, 4, 5])
y_pred = model.predict(x_pred)
y_pred = np.argmax(y_pred, axis = 1)+1
# y_pred = enc.transform(y_pred)
#y_pred = to_categorical(y_pred)
print("y_pred : \n", y_pred)
