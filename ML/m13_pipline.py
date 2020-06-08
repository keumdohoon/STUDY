#20200608 오후 수업
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#1. DATA
iris = load_iris()
x = iris.data
y = iris.target
#싸이킷 런에서 땡기는 방식과 케라스에서 땡기는 방식이 따로 있다.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, 
shuffle = True, random_state = 43)


#2. 모델
#model = SVC() <-원래 이게 모델 만드는것에 끝이였다 그래서 우리는 이제 새로운 방식으로 해주자. 

model = SVC()
#여태까지는 하이퍼파라미터튜닝을 했다 이것은 케이폴드를 사용하여 일정부분에자료에서 발리데이션을 하는 것이였다.
#전처리 친구가 파이프라인이다. 파이프라인과 전처리를 연결할수 맀을 것 같다.
# 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])
pipe.fit(x_train, y_train)
print("acc : ", pipe.score(x_test, y_test))
#SVC와 MINMAX scaler를 쓰겠다. 
#pipeline안에다가 케라스에서 해줬던것처럼 wrap을 해주고 이것을...아...뭐라했노....

#파이프라인에 그리드서치를 넣어 보자 그리드서치에 들어가는 3가지는 , 모델, 파라미터, cv이다 지금 우리가 이 파일에서는 모델을 이미 가지고 있다.
#한방에 전처리와 모델을 함께 돌리는 모델이다.  









































