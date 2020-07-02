#분류와 회기를 하나의 모델에서 가져와서 사용해 보자
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])

#2 .모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

model =Sequential()
model.add(Dense(10, input_shape=(1,)))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1, activation ='sigmoid'))


model.summary()

#3. 컴파일, 훈련
model.compile(loss = ['binary_crossentropy'], 
              optimizer='adam',
              metrics= ['acc'])

model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가 예측
loss = model.evaluate(x_train, y_train)
print('loss: ', loss)

x1_pred  = np.array([11, 12, 13, 14])

y_pred = model.predict(x1_pred)
print('y-pred', y_pred)
#loss, 가 총 7개 나오게 되는데 우리가 했던 회귀 모델의 우웃풋(2) 부분과 분류형 모델의 아웃풋(2) 부분이다. 


# relu 는 0이상인 값에 대해선 linear을 적용해주고 0이하인 값에 대해서는 sigmoid 처럼 해준다. 그래서 relu가 가장 제일 잘 돌아가는 녀석이다. 