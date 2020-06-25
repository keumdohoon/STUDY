#전과 같은 모델에 보스턴 데이타를 추가해준 것이다. 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
import warnings
from sklearn.datasets import load_boston

warnings.filterwarnings('ignore')



# boston = load_boston()
# print(boston)
# x = boston.data
# y = boston.target

boston = pd.read_csv('./data/csv/boston_house_prices.csv', header = 1)


# print(boston)
# print(boston.shape)
# print(type(boston))



x = boston.iloc[:, 0:13]
y = boston.iloc[:, 13]

print(x)
print(y)
# warnings.filterwarning('ignore')


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state = 1)
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
# Name: MEDV, Length: 506, dtype: float64
# ARDRegression 의 정답률 = 0.8012569266998009
# AdaBoostRegressor 의 정답률 = 0.894048414606304
# BaggingRegressor 의 정답률 = 0.9142013529303371
# BayesianRidge 의 정답률 = 0.7937918622384774
# CCA 의 정답률 = 0.7913477184424628
# DecisionTreeRegressor 의 정답률 = 0.7903362302771831
# DummyRegressor 의 정답률 = -0.0005370164400797517
# ElasticNet 의 정답률 = 0.7338335519267194
# ElasticNetCV 의 정답률 = 0.7167760356856181
# ExtraTreeRegressor 의 정답률 = 0.6741489458885821
# ExtraTreesRegressor 의 정답률 = 0.9350798890078814
# GammaRegressor 의 정답률 = -0.0005370164400797517
# GaussianProcessRegressor 의 정답률 = -6.073105259620457
# GeneralizedLinearRegressor 의 정답률 = 0.7447888980918173
# GradientBoostingRegressor 의 정답률 = 0.9452022467146529
# HistGradientBoostingRegressor 의 정답률 = 0.9323597806119726
# HuberRegressor 의 정답률 = 0.7426943860576776