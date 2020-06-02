from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([2,3,4,5,6,7,8,9,10,11])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([20,21,22,23,24,25,26,27,28,29])

model = Sequential()
model.add(Dense(500, input_dim =1 , activation='relu'))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['acc'])
#loss 손실률을 적게 하는건 mse로 하겠다.  
#최적화 시키는건 adam으로 하겠다. 
#metrics를 에큐러시로 하겠다

model.fit(x_train,y_train, epochs=500, batch_size=100, validation_data = (x_train, y_train))
loss, acc = model.evaluate(x_test, y_test, batch_size=100)
#fit 은 훈련시키다, 훈련을 시키는데 x,t, train으로 훈련시키겠다. Epochs500번 훈련 시키겠다. Batch size= 디폴트를 하려면 32개 이상인 데이터를 사용한다. 32개 이하면 의미가 없어진다.  


print("loss : ", loss)
print("acc : ", acc)

y_predict = model.predict(x_test)
print("결과물 : \n", y_predict)
...