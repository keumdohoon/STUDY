import pandas as pd
from sklearn.metrics import accuracy_score
import warnings
from sklearn.metrics import r2_score
from sklearn.utils.testing import all_estimators
import sklearn
from sklearn.model_selection import train_test_split, KFold, cross_val_score

warnings.filterwarnings('ignore')

#data
iris = pd.read_csv('./data/csv/iris.csv',header=0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]

print(x)
print(y)

# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=33)

kfold = KFold(n_splits=5, shuffle=True) # 5등분을 하겠다 데이터를 조각내고 각 조각들을 val에 사용함 5번 실행하게됨
# kfold가 어떻게 돌아가는지 알고 있자 
#예제로 총 데이터셋의 갯수가 200이고 k를 5로 준다면 총 10조각의 하나에 20개의 데이터가 있는 블록들이 생긴다. 
#이 블록들 각각을 다 한번씩 test set으로 설정해두고 나머지 9 개를 train set으로 설정하여 연산한다면 총 10번계산을 하게 되는 것이고 그 값의 평균 가중치를 가져가게 되는 것이다. 

allAlgorithms = all_estimators(type_filter='classifier') # iris에 대한 모든 모델링

for (name, algorithm) in allAlgorithms:
    model = algorithm()
    
    scores = cross_val_score(model, x, y, cv=kfold)     # train과 test로 짤리지 않은 5개의 데이터의 점수 계산
                                              
    print(name, '의 정답률 :')
    print(scores)

import sklearn
print(sklearn.__version__)
# ###############################################################################################################################
