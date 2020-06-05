#boston데이터로 오늘 배운 SVC, linearSVC, KNeighborsClassifier, KNeighborsRegressor
#중 하나를 사용하여 만든다.

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
dataset = load_boston()
print("data: ",  dataset.data)
print('target: ', dataset.target)
x = dataset.data
y = dataset.target
print(x.shape)
print(y.shape) 
print(x)
print(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, shuffle=True,
    train_size = 0.8)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#transform 을 시켜줘서 범위 밖에 나온것들까지 정리해준다. 




#2. 모델
#model = LinearSVC()
#model = SVC()
# model = KNeighborsClassifier()  
model = KNeighborsRegressor()  #score 0.8351801249847223
#model = RandomForestClassifier()  
# model = RandomForestRegressor()  #score 0.9794735024687414
#모델을 이런식으로 짜줘서 금방금방 값을 산출 할 수 있게 해줬다 

# model.RandomForestClassifier(Dense(1, input_dim=4, activation='softmax'))


#3. 실행
# model.compile(loss='categorical_crossentropy',
            #   optimizer='adam',
            #   metrics=['accuracy'])

model.fit(x_train,y_train)
score = model.score(x_test, y_test)
y_pred = model.predict(x_test)
#y_test 와 y_predict 가 비교값이된다. 


# #4, 평가와 예측
# from sklearn.metrics import accuracy_score, r2_score
# print("x_test : \n",x_test,"\npred values : \n",y_pred)
# acc = accuracy_score(y_test,y_pred)

#1. 회귀
# score 외 R2비교

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)

print('score',score)
# print("acc : ",acc)
print("R2 : ", r2)

#score와 acc와  R2를 각각 프린트하면 우리는 score과 동인한거를 보고 머신이 이거를 뭐로 분류하고 계산했는지 알 수 있다. 
#socre는 평가(test) fit은 훈련이니(train)
#   