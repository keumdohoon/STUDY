#1. 데이터
import numpy as np 
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x_pred = np.array([11, 12, 13])

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

#4. 평가와 예측
loss, mse = model.evaluate(x, y, batch_size=1)
print("loss ; ", loss)
print("mse : ", mse)

y_pred = model.predict(x_pred)
print("y_predict : ", y_pred)

#mse
#1. Find the regression line.
#2. Insert your X values into the linear regression equation to find the new Y values (Y’).
#3. Subtract the new Y value from the original to get the error.
#4. Square the errors.
#5. Add up the errors.
#3. Find the mean.

#
#