#m24_SelectFromModel1.boston 에서 Gridsearch를 사용해준 모델
#XG + select from model + gridsearch
#책에 없는 내용이니 주의를 매우 기우릴것
from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, r2_score

# datasets = load_boston()
# x = dataset.data
# y = dataset.target

datasets = load_boston()


x, y = load_boston(return_X_y=True)# 이렇게 적어줘도됨

print(x.shape) #(506, 13)
print(y.shape) #(506,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8,
                                                    shuffle = True, random_state = 66)

model = XGBRegressor(n_estimators = 100, learning_rate = 0.1, n_jobs = 1)

model.fit(x_train, y_train, verbose=True, eval_metric=["logloss","rmse"],
 eval_set=[(x_train, y_train), (x_test, y_test)],
            early_stopping_rounds=20)
#rmse, mae, logloss, error, auc
#error은 회기 모델 지표가 아니다
#eval metric을 두가지 이상으로 할때는 리스트 형식으로 쓴다. 
result = model.evals_result()
print("eval's results :", result)


y_predict = model.predict(x_test)
r2 = r2_score(y_predict, y_test)
print("r2 Score : %.2f%%" %(r2 * 100.0))
print("r2 :", r2)
# Stopping. Best iteration:
# [28]    validation_0-rmse:0.06268       validation_1-rmse:0.28525
#validation 이 올라가기 시작하면서 끊겼다. loss랑 validation 중에 중요한것은  validation 이다. 

import matplotlib.pyplot as plt

epochs = len(result['validation_0']['logloss'])
#우리가 하게 된 에포의 길이 
x_axis = range(0, epochs)




thresholds = np.sort(model.feature_importances_)
             #정렬 #중요도가 낮은 것부터 높은것 까지

print(thresholds)   

import time
start = time.time()

for thresh in thresholds:      
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                            #median
    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)  

    parameters = [
    {"n_estimators":[1000], "learning_rate":[0.1],
     "max_depth":[5]}
    # {"n_estimators":[900,1000,2000], "learning_rate":[0.1,0.01,0.05],
    # "colsample_bytree": [0.6,0.9,1], "max_depth":[4,5,6]},
    # {"n_estimators":[900,2000,1010], "learning_rate":[0.03,0.02,0.3, 0.05],
    # "colsample_bytree": [0.6,0.9,1], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.7,0.9]}
    ]


    selection_model = GridSearchCV(XGBRegressor(), parameters, cv=5)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    x_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, x_pred)
    print("R2 : ", score)

    print("Thresh= %.3f,n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1],
                score*100.0))
    
end = time.time() - start
print("그냥 걸린 시간: ", end)


import time
start = time.time()

for thresh in thresholds:      
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                            #median
    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)  

    parameters = [
    {"n_estimators":[1000], "learning_rate":[0.1],
     "max_depth":[5]}
    # {"n_estimators":[900,1000,2000], "learning_rate":[0.1,0.01,0.05],
    # "colsample_bytree": [0.6,0.9,1], "max_depth":[4,5,6]},
    # {"n_estimators":[900,2000,1010], "learning_rate":[0.03,0.02,0.3, 0.05],
    # "colsample_bytree": [0.6,0.9,1], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.7,0.9]}
    ]


    selection_model = GridSearchCV(XGBRegressor(), parameters, cv=5, n_jobs =5)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    x_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, x_pred)
    print("R2 : ", score)

    print("Thresh= %.3f,n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1],
                score*100.0))
    
end = time.time() - start
print("잡스 걸린 시간: ", end)


fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['logloss'], label = 'Train')
ax.plot(x_axis, result['validation_1']['logloss'], label = 'Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.show()


fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['rmse'], label = 'Train')
ax.plot(x_axis, result['validation_1']['rmse'], label = 'Test')
ax.legend()
plt.ylabel('Rmse')
plt.title('XGBoost Rmse')
plt.show()


# (404, 13)
# R2 :  0.9370405029162931
# Thresh= 0.002,n=13, R2: 93.70%
# (404, 12)
# R2 :  0.9197032930703535
# Thresh= 0.003,n=12, R2: 91.97%
# (404, 11)
# R2 :  0.9285399551925833
# Thresh= 0.010,n=11, R2: 92.85%
# (404, 10)
# R2 :  0.9352632312137594
# Thresh= 0.011,n=10, R2: 93.53%
# (404, 9)
# R2 :  0.9328034891041637
# Thresh= 0.014,n=9, R2: 93.28%
# (404, 8)
# R2 :  0.9289790809229624
# Thresh= 0.015,n=8, R2: 92.90%
# (404, 7)
# R2 :  0.9307001396315936
# Thresh= 0.016,n=7, R2: 93.07%
# (404, 6)
# R2 :  0.9325782121195402
# Thresh= 0.020,n=6, R2: 93.26%
# (404, 5)
# R2 :  0.9173310645249152
# Thresh= 0.025,n=5, R2: 91.73%
# (404, 4)
# R2 :  0.9179020438063288
# Thresh= 0.043,n=4, R2: 91.79%
# (404, 3)
# R2 :  0.8896482460515585
# Thresh= 0.045,n=3, R2: 88.96%
# (404, 2)
# R2 :  0.7952488993972475
# Thresh= 0.253,n=2, R2: 79.52%
# (404, 1)
# R2 :  0.66558224701126
# Thresh= 0.543,n=1, R2: 66.56%