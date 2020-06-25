#iris데이터로 오늘 배운 SVC, linearSVC, KNeighborsClassifier, KNeighborsRegressor
#중 하나를 사용하여 만든다.

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
dataset = load_iris()
print("data: ",  dataset.data)
print('target: ', dataset.target)
x = dataset.data
y = dataset.target
print(x.shape) # (150, 4)
print(y.shape) # (150,)
print(x)
print(y)
'''
y= np_utils.to_categorical(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, shuffle=True,
    train_size = 0.8)

scale = StandardScaler()
x = scale.fit_transform(x)
#2. 모델
# model = LinearSVC()
# model = SVC()
model = KNeighborsClassifier()  #acc = 0.96666, R2 :0.90215310
#model = KNeighborsRegressor()
# model = RandomForestClassifier()   #acc :  0.9, R2 :  0.8521531030
# model = RandomForestRegressor()


# model.RandomForestClassifier(Dense(1, input_dim=4, activation='softmax'))


#3. 실행
model.fit(x_train,y_train)
#  batch_size=1, epochs=100)
score = model.score(x_test, y_test)
print('score', score)
from sklearn.metrics import accuracy_score, r2_score

#4, 평가와 예측
y_pred = model.predict(x_test)
print("x_test : \n",x_test,"\npred values : \n",y_pred)
acc = accuracy_score(y_test,y_pred)
print(x_test, "의 예측 결과 :", y_pred)
# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)

print("R2 : ", r2)
print("acc : ",acc)
print('score: ', score)
'''