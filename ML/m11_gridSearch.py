import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings
from sklearn.metrics import r2_score
import sklearn
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score
from sklearn.svm import SVC


#SVC에 쓰는 하이퍼 파라미터
#1. 데이터
iris = pd.read_csv('./data/csv/iris.csv',sep=',',header=0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]

# print(x)
# print(y)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=33)

Parameters = [
    {"C": [1, 10, 100, 1000], "kernel" :["linear"]},
    {"C": [1, 10, 100, 1000], "kernel" :["rbf"], "gamma": [0.001,0.0001]},
    {"C": [1, 10, 100, 1000], "kernel" :["sigmoid"], "gamma": [0.001, 0.0001]}
]
#c에 1이 커넬 linear로 나오고 10이 kernel 에 linear로 나오고
kfold = KFold(n_splits = 5, shuffle=True)


model = GridSearchCV(SVC(), Parameters, cv=kfold)
#, SVC(), 무엇을 쓰는 것인지, 어떠한 파라미터를 쓰는 건지. cv= 몇개로 cross validation하는 것인지

model.fit(x_train, y_train)
print("최적의 매개변수 :", model.best_estimator_)
y_pred = model.predict(x_test)
print("최종 정답률 = ", accuracy_score(y_test, y_pred))
#가장 좋은 결과 값을 준 파라미터의 갯수를 알 수 있다 어떤 커널과 어떤 C를 사용해야지 좋은 값이 나오는지를 알 수 있는 것이다