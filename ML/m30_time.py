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
print(x_test.shape) #(2000, 71)
print(y_test.shape) #(2000, 4)
print(type(x_test)) #<class 'numpy.ndarray'>
print(type(y_test)) #<class 'numpy.ndarray'>


model = XGBRegressor(n_estimators = 400, learning_rate = 0.1)


model = XGBRegressor()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("R2 :", score) #R2 : 0.925782578365577
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
    {"n_estimators":[1000,2000,3000], "learning_rate":[0.1,0.3,0.01,0.05],
     "max_depth":[4,5,6]},
    {"n_estimators":[900,1000,2000], "learning_rate":[0.1,0.01,0.05],
    "colsample_bytree": [0.6,0.9,1], "max_depth":[4,5,6]},
    {"n_estimators":[900,2000,1010], "learning_rate":[0.03,0.02,0.3, 0.05],
    "colsample_bytree": [0.6,0.9,1], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.7,0.9]}]


    selection_model = GridSearchCV(XGBRegressor(), parameters, cv=5, n_jobs =-1)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    x_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, x_pred)
    print("R2 : ", score)

    print("Thresh= %.3f,n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1],
                score*100.0))
    

end = time.time() - start
print("총 걸린 시간: ", end)
