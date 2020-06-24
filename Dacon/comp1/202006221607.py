# 파라미터 찾기

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


train = pd.read_csv('./data/dacon/comp1/train.csv', header=0, index_col=2)
test = pd.read_csv('./data/dacon/comp1/test.csv', header=0, index_col=2)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header=0, index_col=2)

print('train.shape : ', train.shape)         #x_train, test #train.shape :  (10000, 75)
print('test.shape : ', test.shape)           #x_test.shape :   (10000, 71)
print('summit.shape : ', submission.shape)   #y_predict.summit.shape : (10000, 4)


train = train.interpolate(axis = 0)                       
test = test.interpolate(axis = 0)



#x의 
x_train = train.iloc[:,:71]
y_train = train.iloc[:,-4:]


x_train = x_train.fillna(x_train.mean())
test = test.fillna(test.mean())


# print(x.shape) #(10000, 71)
# print(y.shape) #(10000, 4)

x = x_train
y = y_train
x_pred = test.values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)



###기본 모델###
model = XGBRegressor(n_estimators = 100, learning_rate = 0.1, n_jobs = -1)


model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('R2: ', score) #R2:  0.9323782715918234

from sklearn.feature_selection import SelectFromModel
import numpy as np
###feature engineerg###
thresholds = np.sort(model.feature_importances_)
print('threshold:',thresholds)
 #   [0.00165952 0.00253551 0.01019184 0.01064387 0.01437491 0.01552134
 #   0.01602842 0.02064591 0.0255824  0.04278987 0.04454341 0.25299022
 #   0.54249275]


models = []
res = np.array([])

for thresh in thresholds:      
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                            #median
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test) 
    model2 = XGBRegressor(n_estomators=100, learning_rate = 0.1, n_jobs = -1)
    model2.fit(select_x_train, y_train, verbose = False, eval_metric=['logloss', 'rmse'], 
            eval_set = [(select_x_train, y_train), (select_x_test, y_test)], early_stopping_rounds = 20)

    y_pred = model2.predict(select_x_test)
    score = r2_score(y_test, y_pred)
    shape = select_x_train.shape
    models.append(model2)                    #모델을 배열에 저장

    print(thresh, score)
    res = np.append(res,score)              #결과값 전부를 배열에 저장
print(res.shape)
best_idx = res.argmax()                     #결과값 최대값의 index 저장
score = res[best_idx]                       #위 인덱스 기반으로 점수 호출
total_col = x_train.shape[1] - best_idx     #전체 컬럼 계산
models[best_idx].save_model(f'./model/xgb_save/comp1--{score}--{total_col}--.model') #인덱스 기반으로 모델 저장
'''