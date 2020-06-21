#xgb+randomized search+threshold

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from keras.datasets import mnist
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier, XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import plot_importance
from sklearn.metrics import r2_score

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

x = x_train
y = y_train
x_pred = test.values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)

xgb = XGBRegressor()
model = MultiOutputRegressor(xgb)
model.fit(x_train, y_train)
print('================================')
model.best_estimator_
print(model.best_estimator_)

print("================================")
model.best_params_
print(model.best_params_)

print("================================")
score = model.score(x_test, y_test)
print('점수: ', score)


print(len(model.estimators_))   


for i in range(len(model.estimators_)):
    threshold = np.sort(model.estimators_[i].feature_importances_)
    print(threshold)

    for thres in threshold:
        selection = SelectFromModel(model.estimators_[i], threshold = thres, prefit = True)
        
        parameter = {
            'n_estimators': [100, 300],
            'learning_rate' : [0.01, 0.03,0.07],
            'colsample_bytree': [0.6, 0.7],
            'colsample_bylevel':[0.6, 0.7],
            'max_depth': [5, 6]
        }
    
        search = RandomizedSearchCV(XGBRegressor(), parameter, cv =5, n_jobs = -1)

        select_x_train = selection.transform(x_train)

        multi_search = MultiOutputRegressor(search)
        multi_search.fit(select_x_train, y_train)
        
        select_x_test = selection.transform(x_test)

        y_pred = multi_search.predict(select_x_test)
        score =r2_score(y_test, y_pred)
        print("Thresh=%.3f, n = %d, R2 : %.2f%%" %(thres, select_x_train.shape[1], score*100.0))
 
        select_x_pred = selection.transform(x_pred)
        y_predict = multi_search.predict(select_x_pred)
        # submission
        a = np.arange(10000,20000)
        submission = pd.DataFrame(y_predict, a)
        submission.to_csv('./Dacon/comp1/sub_XGGS.csv',index = True, 
        header=['hhb','hbo2','ca','na'],index_label='id')


# Thresh=0.030, n = 1, R2 : 1.37%
'''
thresholds = np.sort(model.feature_importances_)
             #정렬 #중요도가 낮은 것부터 높은것 까지

print(thresholds)   
'''