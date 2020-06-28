import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
from sklearn.metrics import r2_score
import sklearn
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier, LGBMRegressor
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, mean_absolute_error
train = pd.read_csv('./data/dacon/comp1/train.csv', header=0, index_col=2)
test = pd.read_csv('./data/dacon/comp1/test.csv', header=0, index_col=2)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header=0, index_col=2)

print('train.shape : ', train.shape)         #x_train, test #train.shape :  (10000, 75)
print('test.shape : ', test.shape)           #x_test.shape :   (10000, 71)
print('summit.shape : ', submission.shape)   #y_predict.summit.shape : (10000, 4)



#x의 
x_train = train.iloc[:,:71]
y_train = train.iloc[:,-4:]


x_train = x_train.fillna(x_train.mean())
test = test.fillna(test.mean())


# print('x',x_train.shape)  #(10000, 71)
# print('y',y_train.shape)  #(10000, 4)



x = x_train.values
y = y_train.values
x_pred = test.values

# print('x',type(x)) <class 'numpy.ndarray'>
# print('y',type(y)) <class 'numpy.ndarray'>


scaler= MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
scaler.fit(y)
y = scaler.transform(y)
# y = scaler.transform(y)
print(x.shape)
print(y.shape)
# print(x)
# print(y)
# print(x_pred)




x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)

print("x_tr",x_train)
print("y_tr",y_train)
print("x_tes", x_test)
print("y_tes", y_test)

###기본 모델###
# LGBMRegressor(n_estimators = 300, max_depth =5, scale_pos_weight=1, colsample_bytree=1, learning_rate = 0.1, n_jobs = -1, num_leaves=3)
# model = MultiOutputRegressor(m?odel)

# lgbm = LGBMRegressor(n_estimators = 200, max_depth =6,colsample_bytree=0.7, scale_pos_weight=1.2, 
#                      num_iterations=100, early_stopping_round=20,learning_rate = 0.2, n_jobs = -1,bagging_fraction=3)


lgbm = LGBMRegressor(max_depth =5,dart=0.3,num_iterations =1000,learning_rate=0.07,   n_jobs=-1)
multi_LGBM = MultiOutputRegressor(lgbm)
multi_LGBM.fit(x_train, y_train, verbose=False, eval_metric='logloss', eval_set = [(x_train, y_train), (x_test, y_test)]
                        , early_stopping_rounds=20)


y_pred = multi_LGBM.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print("acc : ", acc)
'''
score = multi_LGBM.score(x_test, y_test)
print('R2: ', score) #R2:  0.9323782715918234

print(len(multi_LGBM.estimators_)) #4
# threshold = (model.estimators_)

# print(regr_multi_RF.estimators_[0].feature_importances_)
# print(regr_multi_RF.estimators_[1].feature_importances_)
# print(regr_multi_RF.estimators_[2].feature_importances_)
# print(regr_multi_RF.estimators_[3].feature_importances_)

# regr_Enet = ElasticNet()
# regr_multi_Enet= MultiOutputRegressor(regr_Enet)
# regr_multi_Enet.fit(x_train, y_train)


# regr_multi_Enet.estimators_[0].coef_
# regr_multi_Enet.estimators_[1].coef_
# regr_multi_Enet.estimators_[2].coef_
# regr_multi_Enet.estimators_[3].coef_


for i in range(len(multi_LGBM.estimators_)):
    threshold = np.sort(multi_LGBM.estimators_[i].feature_importances_)

    for thres in threshold:
        selection = SelectFromModel(multi_LGBM.estimators_[i], threshold = thres, prefit = True)
        
        params = {
            'n_estimators': [100, 200, 400],
            'learning_rate' : [0.03, 0.05, 0.07, 0.1],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'colsample_bylevel':[0.6, 0.7, 0.8, 0.9],
            'max_depth': [4, 5, 6]
        }
    
        search = RandomizedSearchCV( LGBMRegressor(n_estimators = 2000, max_depth =6, scale_pos_weight=2,
                                 colsample_bytree=1, learning_rate = 0.2, n_jobs = -1, num_leaves=5), 
                                 params, cv =5)

        select_x_train = selection.transform(x_train)

        multi_search = MultiOutputRegressor(search,n_jobs = -1)
        multi_search.fit(select_x_train, y_train )
        
        select_x_test = selection.transform(x_test)

        y_pred = multi_search.predict(select_x_test)
        mae = mean_absolute_error(y_test, y_pred)
        score =r2_score(y_test, y_pred)
        print("Thresh=%.3f, n = %d, R2 : %.2f%%, MAE : %.3f"%(thres, select_x_train.shape[1], score*100.0, mae))
 
        select_x_pred = selection.transform(x_pred)
        y_predict = multi_search.predict(select_x_pred)
        # submission
        a = np.arange(10000,20000)
        submission = pd.DataFrame(y_predict, a)
        submission.to_csv('./Dacon/comp1/select_LG%d_%.5f.csv'%(i, mae),index = True, header=['hhb','hbo2','ca','na'],index_label='id')





models = []
res = np.array([])

for thresh in thresholds:      
    selection = SelectFromModel(regr_multi_RF, threshold=thresh, prefit=True)
                                            #median
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test) 
    model2 = MultiOutputRegressor(LGBMRegressor(n_estimators = 300, max_depth =5, scale_pos_weight=1, colsample_bytree=1, learning_rate = 0.1, n_jobs = -1, num_leaves=5))
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