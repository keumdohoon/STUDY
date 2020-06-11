#using XGB and GBBooster other than using the for function

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
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
# model = XGBRegressor()


model = MultiOutputRegressor(GradientBoostingRegressor())
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print(score)
y4 = model.predict(test.values)


#여기서 definition과 for 문을 써준 이유는 GB와 XGB에서는 스칼라 형태일때만 정보가 받아지기 때문에 저 두개의 모델을 구동시키기 위해서는 
#현재 가지고 있는 데이터셋을 총 4번(4컬럼이니까) 으로 잘라줘서 스칼라의 형태로 만들어주는 것이다 . 이 for문은 그것을 진행해주기 위해서 있는것이다.
#나머지 random forest와 decision tree는 스칼라의 형태로 구동을 하더라도 전혀 상관 없이 잘 구동된다.  

# y_predict = tree_fit(y_train, y_test)

print(y4.shape)

'''
# submission
a = np.arange(10000,20000)
submission = pd.DataFrame(y4, a)
submission.to_csv('D:/Study/Bitcamp/Dacon/comp1/sub_GB.csv',index = True, header=['hhb','hbo2','ca','na'],index_label='id')

# print(model.feature_importances_)


## feature_importances
def plot_feature_importances(model):
    plt.figure(figsize= (10, 40))
    n_features = x_data.shape[1]                                # n_features = column개수 
    plt.barh(np.arange(n_features), model.feature_importances_,      # barh : 가로방향 bar chart
              align = 'center')                                      # align : 정렬 / 'edge' : x축 label이 막대 왼쪽 가장자리에 위치
    plt.yticks(np.arange(n_features), x_data.columns)          # tick = 축상의 위치표시 지점
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)             # y축의 최솟값, 최댓값을 지정/ x는 xlim

plot_feature_importances(model)
plt.show()
'''
