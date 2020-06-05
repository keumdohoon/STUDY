import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings
from sklearn.metrics import r2_score
import sklearn
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from keras.datasets import mnist
#randomforest 적용
#breast_cancer적용

#SVC에 쓰는 하이퍼 파라미터
#1. 데이터
cancer = pd.read_csv('./data/csv/breast_cancer.csv',sep=',',header=0)
#breast cancer이라는 2차원의 데이터를 불러 와준다. 지금 여기서는 2차원만을 지원하기 때문에 만약 3차원인 데이터를 가져오게 된다면 reshape을 해줘야한다. 
# dataset = .load_data()
print(cancer)
print(type(cancer))

x = cancer.iloc[:, 0:3]
y = cancer.iloc[:, 3]

#데이터를 x와y로 나누어주는데 뭐를 어떻게 잘라주는지를 지정해 주는 것이다. 


print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=33)

Parameters = [
    {"max_depth":[1,2,3,4], "n_estimators":[10,100,110,120], "random_state":[0,1,2]}]
#파라미터를 정해준다, 여기서의 파라미터는 우리가 지정해주고 나준네 GridSearch 에 사용하기 위함이다. 
#우리가 사용하는 모델은 GridSearch이기에 거기에 들어가는 파라미터와 RandomForest를 사용하게 된다. 그리고 CV를 Kfold로 설정해주게 된다. 
#c에 1이 커넬 linear로 나오고 10이 kernel 에 linear로 나오고!~
kfold = KFold(n_splits = 5, shuffle=True)


model = GridSearchCV(RandomForestClassifier(), Parameters, cv=kfold)
#, SVC(), 무엇을 쓰는 것인지, 어떠한 파라미터를 쓰는 건지. cv= 몇개로 cross validation하는 것인지

model.fit(x_train, y_train)
print("최적의 매개변수 :", model.best_estimator_)
y_pred = model.predict(x_test)
print("최종 정답률 = ", accuracy_score(y_test, y_pred))


#가장 좋은 결과 값을 준 파라미터의 갯수를 알 수 있다 어떤 커널과 어떤 C를 사용해야지 좋은 값이 나오는지를 알 수 있는 것이다.
# , criterion='gini',
#             max_depth=3, max_features='auto', max_leaf_nodes=None,        
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,  
#             oob_score=False, random_state=2, verbose=0, warm_start=False) 
# 최종 정답률 =  0.9298245614035088