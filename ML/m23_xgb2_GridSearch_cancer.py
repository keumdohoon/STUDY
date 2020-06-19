#breast cancer 이중분류
#GridSearch 이용

#feature을 xgboost로 건들겠다 
from xgboost import XGBRegressor, XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import plot_importance
import matplotlib.pyplot as plt

dataset = load_breast_cancer()
x= dataset.data
y = dataset.target

print(x.shape) #(150, 4)
print(y.shape) #(506,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8,
                                                    shuffle = True, random_state = 120)






n_estimators = 220
learning_rate = 0.035      
colsample_bytree = 0.7   
colsample_bylevel = 0.7  
# 점수:  1.0


#cv는 꼭 써서 결과치를 보고 feature importance도 봐줘야한다. 
#위에 이것 4개만 조정해주면 된다. 
max_depth = 5

#tree model 이 ensemble 된게 forest 이다
#boosting계열에서는 tree구조에서 쓰던 특성을 다 가져온다 
#특성에는 : 전처리가 필요없다, 결측치 제거를 안해도된다, 
#xgboost는 앙상블이니까 decision tree보다는 많이 느리다 .
parameters = [
    {"n_estimators":[100,200,300], "learning_rate":[0.1,0.3,0.01,0.5],
     "max_depth":[4,5,6]},
    {"n_estimators":[90,100,110], "learning_rate":[0.1,0.01,0.5],
    "colsample_bytree": [0.6,0.9,1], "max_depth":[4,5,6]},
    {"n_estimators":[90,100,110], "learning_rate":[0.03,0.02,0.3],
    "colsample_bytree": [0.6,0.9,1], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.7,0.9]}]

n_jobs = -1
# 점수:  0.9736842105263158
# XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,       
#               colsample_bynode=1, colsample_bytree=0.6, gamma=0, gpu_id=-1,
#               importance_type='gain', interaction_constraints='',
#               learning_rate=0.5, max_delta_step=0, max_depth=5,
#               min_child_weight=1, missing=nan, monotone_constraints='()',  
#               n_estimators=90, n_jobs=0, num_parallel_tree=1, random_state=0,
#               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,  
#               tree_method='exact', validate_parameters=1, verbosity=None)  
# {'colsample_bytree': 0.6, 'learning_rate': 0.5, 'max_depth': 5, 'n_estimators': 90}
#즉 4덩어리가 있는 두번째게 제일 좋은것이다. 







# xgb = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, colsample_bytree=colsample_bytree
#                                     , n_estimators=n_estimators, n_jobs=n_jobs, colsample_bylevel= colsample_bylevel)

model = GridSearchCV(XGBClassifier(), parameters, cv=5, n_jobs=-1)
#n_jobs를 xgb에 넣어도 되지만 계속 끊었다가 돌아가고 끊었다가 돌아가고 하니까 속도가 느려진다 그래서 gridsearch에다가 넣어주게 되면 통으로 돌아간다. 
model.fit(x_train, y_train)
print('================================')
model.best_estimator_
print(model.best_estimator_)

print("================================")
model.best_params_
print(model.best_params_)

print("================================")


# XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,       
#               colsample_bynode=1, colsample_bytree=0.6, gamma=0, gpu_id=-1,
#총 파라미터의 수가 몇개의 덩어리인지를 보면 얘가 고른게 뭔지 알 수 있다. 

score = model.score(x_test, y_test)
print('점수: ', score)

# print(model.feature_importances_)
# print('========================================')
print(model.best_estimator_)
# print('========================================')

print(model.best_params_)
# print('========================================')

# plot_importance(model)
# plt.show()
#어떤 feature이 중요한지 쉽게 볼수가 있다. 딱 한 줄로 정리가 가능 