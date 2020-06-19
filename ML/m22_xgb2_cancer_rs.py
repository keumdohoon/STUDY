#breast cancer 이중분류

#feature을 xgboost로 건들겠다 
from xgboost import XGBRegressor, XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.utils.testing import all_estimators
import warnings
from sklearn.metrics import r2_score
import sklearn
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from keras.datasets import mnist

dataset = load_breast_cancer()
x= dataset.data
y = dataset.target

print(x.shape) #(569, 30)
print(y.shape) #(569,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8,
                                                    shuffle = True, random_state = 66)



Parameters = parameters = {"n_estimators": [210, 100, 1000],
              "learning_rate": [0.035, 0.045, 0.05, 0.021],
              "colsample_bytree": [0.75, 0.65, 0.55, 0.59],
              "colsample_bylevel": [0.69, 0.55, 0.56, 0.49]}


# n_estimators = 210
# learning_rate = 0.035      
# colsample_bytree = 0.75   
# colsample_bylevel = 0.69  
# 점수:  0.9736842105263158

max_depth = 5
n_jobs = -1

#cv는 꼭 써서 결과치를 보고 feature importance도 봐줘야한다. 
#위에 이것 4개만 조정해주면 된다. 
max_depth = 5
n_jobs = -1
#tree model 이 ensemble 된게 forest 이다
#boosting계열에서는 tree구조에서 쓰던 특성을 다 가져온다 
#특성에는 : 전처리가 필요없다, 결측치 제거를 안해도된다, 
#xgboost는 앙상블이니까 decision tree보다는 많이 느리다 .
# xgb = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, colsample_bytree=colsample_bytree
#                                     , n_estimators=n_estimators, n_jobs=n_jobs, colsample_bylevel= colsample_bylevel)


model = RandomizedSearchCV(XGBClassifier(), Parameters, cv=5,n_jobs = -1)

model.fit(x_train, y_train)
print('================================')

score = model.score(x_test, y_test)
print('점수: ', score)

print('================================')

print("최적의 매개변수 :", model.best_estimator_)

print('================================')

print("최종 정답률 = ", model.score(x_test, y_test))
                    
# 최적의 매개변수 : XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.69,
#               colsample_bynode=1, colsample_bytree=0.75, gamma=0, gpu_id=-1,
#               importance_type='gain', interaction_constraints='',
#               learning_rate=0.05, max_delta_step=0, max_depth=6,
#               min_child_weight=1, missing=nan, monotone_constraints='()',
#               n_estimators=1000, n_jobs=0, num_parallel_tree=1, random_state=0,
#               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#               tree_method='exact', validate_parameters=1, verbosity=None)
# ================================
# 최종 정답률 =  0.9736842105263158


# print(model.feature_importances_)
# print('========================================')
# print(model.best_estimator_)
# print('========================================')

# print(model.best_params_)
# print('========================================')

# plot_importance(model)
# plt.show()
#어떤 feature이 중요한지 쉽게 볼수가 있다. 딱 한 줄로 정리가 가능 
