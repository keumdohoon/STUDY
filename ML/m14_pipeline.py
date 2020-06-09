#. 
# #20200608 오후 수업 13번 복사해온것 
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC

#1. DATA
iris = load_iris()
x = iris.data
y = iris.target
#싸이킷 런에서 땡기는 방식과 케라스에서 땡기는 방식이 따로 있다.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, 
                                                    shuffle = True, random_state = 43)


#그리드 서치에서 사용할 매게변수(딕셔너리가 리스트 형식으로 연결되어있는것)
parameter = [{"svm__C":[1, 10, 100, 1000],'svm__kernel':['linear']}, 
             {"svm__C":[1, 10, 100, 1000], 'svm__kernel':['rbf'], 
                                         'svm__gamma':[0.001, 0.0001]},   
             {"svm__C":[1, 10, 100, 1000], 'svm__kernel':['sigmoid'],
                                         'svm__gamma':[0.001, 0.0001]}
            ]#총 20가지의 경우의수가 가능한 파라미터가 만들어 진것이다. 
#언더바를 두개씩 넣어줘야한다. 
#2. 모델
model = SVC()

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])
pipe = make_pipeline(MinMaxScaler(), SVC())
pipe.fit(x_train, y_train)
print("acc : ", pipe.score(x_test, y_test))
#MinMax scaler을 이용하여 한방에 전처리까지 다 해줬다. 
model = RandomizedSearchCV(pipe, parameter, cv = 5)
#모델에 파이프를 넣어준다. 
#make_pipeline 이라는 것이 있다. 

#3. 훈련
model.fit(x_train, y_train)

#4. 평가와 예측
acc = model.score(x_test, y_test)
print('최적의 매개 변수 = ', model.best_estimator_)
print("acc : ", acc)






































