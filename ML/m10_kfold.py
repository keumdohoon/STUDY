import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import sklearn
from sklearn.model_selection import train_test_split, KFold, cross_val_score
warnings.filterwarnings('ignore')


warnings.filterwarnings('ignore')

iris = pd.read_csv('./data/csv/iris.csv',sep=',',header=0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]

print(x)
print(y)

# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=33)

kfold = KFold(n_splits=5, shuffle=True) # 5등분을 하겠다 데이터를 조각내고 각 조각들을 val에 사용함 5번 실행하게됨
# kfold가 어떻게 돌아가는지 알고 있자 

allAlgorithms = all_estimators(type_filter='classifier') # iris에 대한 모든 모델링

for (name, algorithm) in allAlgorithms:
    model = algorithm() # -> 존나 희안한 문법인거 같은데 

    scores = cross_val_score(model,x,y, cv=kfold) # 분리 하지 않은 데이터를 넣어도 알아서 잘라서 학습하고 평가한다 
    print(name,"의 정답률 = ", scores)

print(sklearn.__version__)
# ###############################################################################################################################
# boston = load_boston()
# print(boston)
# x = boston.data
# y = boston.target

# # boston = pd.read_csv('./data/csv/boston_house_prices.csv', header = 0)
# x = boston.iloc[:, 0:13]
# y = boston.iloc[:, 13]

# print(x)
# print(y)
# # warnings.filterwarning('ignore')







# # x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2, random_state = 66)
# # warnings.filterwarning('ignore')
# KFold = KFold(n_splits= 5, shuffle=True)

# allAlgorithms = all_estimators(type_filter = 'regressor')

# for (name, algorithm) in allAlgorithms:
#         model = algorithm()
    
#         scores = cross_val_score(model,x,y,cv=KFold)

#         print(name, "의정답률=")
#         print(scores)
# #여기서 스코어는
# import sklearn
# print(sklearn.__version__)
# # warnings.filterwarning('ignore')

