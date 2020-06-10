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
parameter = [{"svc__C":[1, 10, 100, 1000],'svm__kernel':['linear']}, 
             {"svc__C":[1, 10, 100, 1000], 'svm__kernel':['rbf'], 
                                         'svm__gamma':[0.001, 0.0001]},   
             {"svc__C":[1, 10, 100, 1000], 'svm__kernel':['sigmoid'],
                                         'svm__gamma':[0.001, 0.0001]}
            ]#총 20가지의 경우의수가 가능한 파라미터가 만들어 진것이다. 
#언더바를 두개씩 넣어줘야한다. 
#2. 모델


from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
pipe = make_pipeline(MinMaxScaler(), SVC())
# pipeline은 scaler 쓰고 어떤 기법을 쓸지 명시, 모델쓰고, 기법 명시
# make pipeline은 (전처리, 모델
# pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])
# pipe = make_pipeline(MinMaxScaler(), SVC())
#make pipeline 역시도 이름을 명시해 줘야한다. 

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



#통상 그리드 서치를 할때에는 그리드 서피 명만 적어주면 된다. 하지만 파이프라인으로 엮게 된다면 앞에다가 모델명이랑 __언더라인 두개를 추가해줘야한다.
#SVM은 SVC에서 이미 명시를 해준것이다.   


































