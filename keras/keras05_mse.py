#1. 데이터
import numpy as np 
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x_pred = np.array([11, 12, 13])


#이 모델구성으로 보아서는 x,y룰 트레인한 후에 그 가중치를 가지고 x_pred에 적용시켜주억서
#결론적으로는 Y_pred를 알아내주고 싶다는 것이다. 
#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim=1))
model.add(Dense(3))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x, y, epochs=30, batch_size=1)

#이제 위에 레이어를 만들어준 모델을 가지고 모델 컴파일과 모델 핏을 해주는 것이다. 



#4. 평가와 예측
loss, mse = model.evaluate(x, y, batch_size=1)
print("loss ; ", loss)
print("mse : ", mse)
#위에서 핏한 모델을 가지고 모델evaluate을 통하여 loss, 와 mse를 구해주는 것이다. 
#그것을 하는 과정에서는 x_pred만으로도 가능하다. 
y_pred = model.predict(x_pred)
print("y_predict : ", y_pred)

#loss와 mse와는 별개로 이제 위에서 model을 가져와서 그것을
#predict해주는데 그 predict해주는 것이 x_pred를 넣어서 y를 찾아주는 것이다. 



#mse
#1. Find the regression line.
#2. Insert your X values into the linear regression equation to find the new Y values (Y’).
#3. Subtract the new Y value from the original to get the error.
#4. Square the errors.
#5. Add up the errors.
#3. Find the mean.

#
#