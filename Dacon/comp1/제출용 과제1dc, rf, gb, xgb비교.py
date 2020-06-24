from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

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

def tree_fit(y_train, y_test):
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print('score: ', score)
    y_predict = model.predict(x_pred)
    y_pred1 = model.predict(x_test)
    print('mae: ', mean_absolute_error(y_test, y_pred1))
    return y_predict

def boost_fit_acc(y_train, y_test):
    y_predict = []
    for i in range(len(submission.columns)):
       print(i)
       y_train1 = y_train[:, i]  
       model.fit(x_train, y_train1)
       
       y_test1 = y_test[:, i]
       score = model.score(x_test, y_test1)
       print('score: ', score)

       y_pred = model.predict(x_pred)
       y_pred1 = model.predict(x_test)
       print('mae: ', mean_absolute_error(y_test1, y_pred1))

       y_predict.append(y_pred)     
    return np.array(y_predict)
#여기서 definition과 for 문을 써준 이유는 GB와 XGB에서는 스칼라 형태일때만 정보가 받아지기 때문에 저 두개의 모델을 구동시키기 위해서는 
#현재 가지고 있는 데이터셋을 총 4번(4컬럼이니까) 으로 잘라줘서 스칼라의 형태로 만들어주는 것이다 . 이 for문은 그것을 진행해주기 위해서 있는것이다.
#나머지 random forest와 decision tree는 스칼라의 형태로 구동을 하더라도 전혀 상관 없이 잘 구동된다.  

# y_predict = tree_fit(y_train, y_test)
y_predict = boost_fit_acc(y_train, y_test).reshape(-1, 4) 

print(y_predict.shape)


# submission
a = np.arange(10000,20000)
submission = pd.DataFrame(y_predict, a)
submission.to_csv('D:/Study/Bitcamp/Dacon/comp1/sub_XG.csv',index = True, header=['hhb','hbo2','ca','na'],index_label='id')

print(model.feature_importances_)


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

# Decision Tree
# score:  -0.01426995728561642
# mae:  1.5261941093237883
# (10000, 4)
# [0.         0.         0.00030665 0.00911951 0.11334899 0. 
#  0.12136225 0.10666495 0.09691415 0.         0.         0. 
#  0.         0.         0.         0.         0.         0. 
#  0.         0.00249662 0.         0.02987637 0.06040505 0.03948063
#  0.         0.         0.         0.         0.10946241 0.05104018
#  0.         0.         0.25952224 0.         0.         0. 
#  0.         0.         0.         0.         0.         0. 
#  0.         0.         0.         0.         0.         0. 
#  0.         0.         0.         0.         0.         0. 
#  0.         0.         0.         0.         0.         0. 
#  0.         0.         0.         0.         0.         0. 
#  0.         0.         0.         0.         0.        ]   
#Random Forest Regressor
# score:  -0.001432196452210599
# mae:  1.5198243896427608
# (10000, 4)
# [0.00609831 0.02498917 0.0226179  0.01900474 0.06731379 0.00700843
#  0.04343121 0.03124981 0.05624135 0.06924191 0.01600715 0.06497589
#  0.04176191 0.03356403 0.01582278 0.03791666 0.00779199 0.01525668
#  0.01979719 0.02936414 0.03522273 0.02024683 0.01511147 0.01524717
#  0.0165726  0.01227296 0.02177937 0.02155611 0.04252312 0.03951366
#  0.00804185 0.01376025 0.01808606 0.01239635 0.01321803 0.06499638
#  0.         0.         0.         0.         0.         0. 
#  0.         0.         0.         0.         0.         0. 
#  0.         0.         0.         0.         0.         0. 
#  0.         0.         0.         0.         0.         0. 
#  0.         0.         0.         0.         0.         0. 
#  0.         0.         0.         0.         0.        ]   
#GradientBoostRegressor
# score:  -0.013738246405915477
# mae:  1.5312480087687914
# (10000, 4)
# [0.00718467 0.02616578 0.02339543 0.023618   0.02969708 0.01116704
#  0.03162878 0.01942549 0.02125801 0.02796429 0.02507008 0.03649741
#  0.03062295 0.057623   0.02037826 0.02152289 0.02393738 0.04621921
#  0.0209846  0.05749204 0.0516735  0.04426792 0.02202807 0.02328686
#  0.01387298 0.01818325 0.04273139 0.02341886 0.03885417 0.03105696
#  0.00658926 0.01901451 0.02890491 0.03050716 0.01018113 0.03357672
#  0.         0.         0.         0.         0.         0. 
#  0.         0.         0.         0.         0.         0. 
#  0.         0.         0.         0.         0.         0. 
#  0.         0.         0.         0.         0.         0. 
#  0.         0.         0.         0.         0.         0. 
#  0.         0.         0.         0.         0.        ]  
##XGBOOSTER
# score:  0.05151398280090491
# mae:  1.4631492877960204
# (10000, 4)
# [0.00714453 0.00456563 0.00523687 0.00760908 0.00678088 0.00609987
#  0.01076802 0.00948863 0.01044757 0.01051742 0.00870712 0.01005827
#  0.01080324 0.01445976 0.00863465 0.00999479 0.01383694 0.01053726
#  0.01033302 0.013858   0.01236561 0.01186646 0.0114035  0.01210822
#  0.01168458 0.01149829 0.01143724 0.01593832 0.01389337 0.01312196
#  0.01431437 0.01188886 0.01445564 0.01280423 0.01480111 0.01269285
#  0.01433546 0.01052709 0.01429052 0.0100729  0.01207726 0.01429687
#  0.01217189 0.01429322 0.02461783 0.02595455 0.01788696 0.01153878
#  0.0147355  0.01482237 0.01462235 0.01609324 0.01367005 0.01539843
#  0.01336798 0.0178892  0.02250576 0.02116024 0.02215998 0.01577337
#  0.02519553 0.02593758 0.03072716 0.0268355  0.01777377 0.01827069
#  0.01372255 0.01831213 0.01399318 0.01131341 0.01750056]   
#전체 데이터를 보면 xg부스터의 정보가 가장 고루고루 잘 퍼져 있소 score 와 mae 도 음수가 아닌 양수를 띈다. 이로써 XGbooster을 사용하여 나온 결과가 가장 신뢰할수 있다고 본다. 