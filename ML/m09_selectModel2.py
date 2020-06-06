#전과 같은 모델에 보스턴 데이타를 추가해준 것이다. 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')



boston = load_boston()
print(boston)
x = boston.data
y = boston.target

# boston = pd.read_csv('./data/csv/boston_house_prices.csv', header = 0)


# print(boston)
# print(boston.shape)
# print(type(boston))



# x = boston.iloc[:, 0:13]
# y = boston.iloc[:, 13]

print(x)
print(y)
# warnings.filterwarning('ignore')


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2, random_state = 66)
# warnings.filterwarning('ignore')


allAlgorithms = all_estimators(type_filter = 'regressor')

for (name, algorithm) in allAlgorithms:
        model = algorithm()
    
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, "의 정답률 =", r2_score(y_test, y_pred))


import sklearn
print(sklearn.__version__)
# warnings.filterwarning('ignore')
