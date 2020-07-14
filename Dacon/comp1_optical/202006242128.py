#XGB_ML_
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error as mae
test = pd.read_csv('./data/dacon/comp1/test.csv', sep=',', header = 0, index_col = 0)

x_train = np.load('./dacon/comp1/x_train.npy')
y_train = np.load('./dacon/comp1/y_train.npy')
x_pred = np.load('./dacon/comp1/x_pred.npy')




x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, train_size = 0.8, random_state = 66
)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

# 2. model

parameters =[
    {'n_estimators': [3000],
    'learning_rate': [0.1],
    'max_depth': [6], 
    'booster': ['dart'], 
    'rate_drop' : [0.21],
    'eval_metric': ['logloss','mae'], 
    'is_training_metric': [True], 
    'max_leaves': [144], 
    'colsample_bytree': [0.8], 
    'subsample': [0.8],
    'seed': [66]
    }
]
kfold = KFold(n_splits=5, shuffle=True, random_state=66)
y_test_pred = []
y_pred = []
search = RandomizedSearchCV(XGBRegressor(n_jobs=6), parameters, cv = kfold, n_iter=1)

for i in range(4):
    fit_params = {
        'verbose': True,
        'eval_metric': ['logloss','mae'],
        'eval_set' : [(x_train,y_train[:,i]),(x_test,y_test[:,i])],
        'early_stopping_rounds' : 5
    }
    search.fit(x_train, y_train[:,i],**fit_params)
    y_pred.append(search.predict(x_pred))
    y_test_pred.append(search.predict(x_test))
    # print(search.best_score_)



#############################

y_pred = np.array(y_pred).T
y_test_pred = np.array(y_test_pred).T

print(y_pred.shape)
r2 = r2_score(y_test,y_test_pred)
mae = mae(y_test,y_test_pred)
# print('r2 :', r2)
# print('mae :', mae)
print(y_pred)
print(y_pred.shape)


submissions = pd.DataFrame({
    "id": np.array(range(10000,20000)),
    "hhb": y_pred[:, 0],
    "hbo2": y_pred[:, 1],
    "ca": y_pred[:, 2],
    "na": y_pred[:, 3]
})
print(submissions)
submissions.to_csv('./Dacon/comp1/comp1_sub.csv', index = False)
