#compile 에 있는 loss를 줄이기 위해서 우리느 ㄴoptimizer을 쓰고 그 안에 우리는 보통 adam을 써 줬었다


#데이터
import numpy as np
x = np.array([1,2,3,4])
y = np.array([1,2,3,4])

#모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=1, activation= 'relu'))
model.add(Dense(3))
model.add(Dense(11))
model.add(Dense(1))


from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Adamax , Nadam
# optimizer = Adam(lr=0.001)
# loss : [0.19643323123455048
# pred: [[2.610412 ]
#  [3.9187891]]


# optimizer = RMSprop(lr=0.001)
# loss : [0.010662270709872246
# pred: [[2.9927568], [4.8196545]]

# optimizer = SGD(lr=0.001)
# loss : [0.058813609182834625, 0.058813609182834625]
# pred: [[2.9790134],  [4.5702343]]

# optimizer = Adadelta(lr=0.001)
# loss : [6.087420463562012, 6.087420463562012]
# pred: [[0.2972415 ], [0.49420673]]


# optimizer = Adagrad(lr=0.001)
# loss : [7.146006107330322, 7.146006107330322]
# pred: [[0.07196042], [0.0568344 ]]

# optimizer =  Adamax(lr=0.001)
# loss : [0.17867955565452576, 0.17867955565452576]
# pred: [[2.7001843], [3.974261 ]]

optimizer = Nadam(lr=0.001)
# loss : [0.006105212494730949, 0.006105212494730949]
# pred: [[2.9938602], [4.862501 ]]

#adam도 경사하강법 중에 하나이다. 

model.compile(loss='mse', optimizer =optimizer, metrics = ['mse'])
#우리는 loss의 최적값을 구하기 위해서 경사하강법을 사용해 주었다.
model.fit(x, y, epochs = 100)

loss = model.evaluate(x,y)
print("loss :", loss)

pred1 = model.predict([3.5])
print("pred:", pred1)