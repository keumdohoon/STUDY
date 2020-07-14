from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
train = pd.read_csv('./data/dacon/comp1/train.csv', index_col= 0 , header = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', index_col= 0 , header = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', index_col= 0 , header = 0)

print('train.shape: ', train.shape)              # (10000, 75)  = x_train, test
print('test.shape: ', test.shape)                # (10000, 71)  = x_predict
print('submission.shape: ', submission.shape)    # (10000, 4)   = y_predict
                    

train = train.interpolate()                       
test = test.interpolate()

x_data = train.iloc[:, :71]                           
y_data = train.iloc[:, -4:]


x_data = x_data.fillna(x_data.mean())
test = test.fillna(test.mean())


x = x_data.values
y = y_data.values
x_pred = test.values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state =33)

# model = DecisionTreeRegressor(max_depth =4)                     # max_depth 몇 이상 올라가면 구분 잘 못함
# model = RandomForestRegressor(n_estimators = 200, max_depth=3)
# model = GradientBoostingRegressor()
model = XGBRegressor()
# n_estimators=1000
# model = MultiOutputRegressor(xgb.XGBRFRegressor())

# 2. model

parameters = {
    'booster' : ['gbtree', 'gblinear', 'dart'],
    'validate_parameters' : [True, False],
    'n_jobs' : [-1]
}
y_pred = []
search = RandomizedSearchCV(model, parameters, cv = 5, n_iter=5)


# model.fit(x_train,y_train)
# score = model.score(x_test,y_test)
# print(score)
# y4 = model.predict(test.values)
# print(search)

for i in range(4):
    search.fit(x_train, y_train[:,i])

    print(search.best_params_)
    print("MAE :", search.score(x_test,y_test[:,i]))

    y_pred.append(search.predict(x_pred))

y_pred = np.array(y_pred)
submissions = pd.DataFrame({
    "id": test.index,
    "hhb": y_pred[0,:],
    "hbo2": y_pred[1,:],
    "ca": y_pred[2,:],
    "na": y_pred[3,:]
})

submissions.to_csv('./dacon/comp1/comp1_sub.csv', index = False)
