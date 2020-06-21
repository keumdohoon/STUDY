#breast cancer 이중분류
#GridSearch 이용

#feature을 xgboost로 건들겠다 
from xgboost import XGBRegressor, XGBClassifier
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import plot_importance
import matplotlib.pyplot as plt

dataset = load_boston()
x= dataset.data
y = dataset.target

print(x.shape) 
print(y.shape) 

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8,
                                                    shuffle = True, random_state = 120)

n_estimators = 220
learning_rate = 0.035      
colsample_bytree = 0.7   
colsample_bylevel = 0.7  
# 점수:  1.0

max_depth = 5

parameters = [
    {"n_estimators":[1000,2000,3000], "learning_rate":[0.1,0.3,0.01,0.05],
     "max_depth":[4,5,6]},
    {"n_estimators":[1010], "learning_rate":[0.03],
    "colsample_bytree": [0.6], "max_depth":[4]},
    {"n_estimators":[900,2000,1010], "learning_rate":[0.03,0.02,0.3, 0.05],
    "colsample_bytree": [0.6,0.9,1], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.7,0.9]}
    ]

n_jobs = -1
# 점수:  0.8858573207924959
# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,        
#              colsample_bynode=1, colsample_bytree=0.6, gamma=0, gpu_id=-1, 
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.03, max_delta_step=0, max_depth=4,
#              min_child_weight=1, missing=nan, monotone_constraints='()',   
#              n_estimators=1010, n_jobs=0, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,   
#              tree_method='exact', validate_parameters=1, verbosity=None)   
# {'colsample_bytree': 0.6, 'learning_rate': 0.03, 'max_depth': 4, 'n_estimators': 1010}

model = GridSearchCV(XGBRegressor(), parameters, cv=5, n_jobs=-1)
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


plot_importance(model)
plt.show()