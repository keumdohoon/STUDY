#1. 데이터
import numpy as np 
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
#np.array의 뜻은 np안에 있는 Array 를 땡겨온다는 것이다. 
#batch_size의 Deafault 값은 32이다.

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

#from keras.models import Sequential 은 keras 안에 models안에 Sequential 을 땡겨오겠다는 뜻이다.  
#from keras.layers import Dense 은 keras 안에 models 안에 Dense를 땡겨오겠다는 뜻이다.  
#model = Sequential() 은 앞으로 Sequential 을 Model 이라 칭하겠다는 뜻이다. 
 
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
loss, acc = model.evaluate(x, y, batch_size=1)
print("loss ; ", loss)
print("acc : ", acc)

#두가지 예측방식이 있는데 회귀(linear) 방식과 분류(classifier) 방식이다. 
#MSE는 고정값이 아닌 자유로운 방식으로 더해져나아간다. 
#훈련용 데이터와 시험용 데이터의 값을 둘다 주어야하는 것은 훈련용 데이터를 토대로 얻은 결과 값으로 인하여 시험용 데이터에 대입하여 컴퓨터가 계산하는 것이다. 훈려과 데이터를 7:3정도로 나누는 경우가 제일 많다. 
#평가데이터는 모델에 반영되지 않는다. 평가는 우리가 보기위한 데이터이기 때문이다..



