import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings
from sklearn.metrics import r2_score
import sklearn
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from keras.datasets import mnist
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, Dense
from keras.layers import MaxPooling2D
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import numpy as np

train = pd.read_csv('./data/dacon/comp1/train.csv', header=0, index_col=0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header=0, index_col=0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header=0, index_col=0)

print('train.shape : ', train.shape)         #x_train, test #train.shape :  (10000, 75)
print('test.shape : ', test.shape)           #x_predict#test.shape :   (10000, 71)
print('summit.shape : ', submission.shape)   #y_predict#summit.shape : (10000, 4)
#test는 지금 xpredict밖에 되지 않음 

# print(train.isnull().sum())
 # rho           0
 # 650_src       0
 # 660_src       0
 # 670_src       0
 # 680_src       0
 #            ...
 # 990_dst    1987
 # hhb           0
 # hbo2          0
 # ca            0
 # na            0
 # Length: 75, dtype: int64
train = train.interpolate() #보간법 //선형보간
print(train)
 # rho  650_src  660_src  670_src  ...    hhb  hbo2     ca    na
 # id                                    ...
 # 0      25  0.37950  0.42993  0.52076  ...   5.59  4.32   8.92  4.29
 # 1      10  0.00000  0.00000  0.01813  ...   0.00  2.83   7.25  4.64
 # 2      25  0.00000  0.03289  0.02416  ...  10.64  3.00   8.40  5.16
 # 3      10  0.27503  0.31281  0.32898  ...   5.67  4.01   5.05  4.35
 # 4      15  1.01521  1.00872  0.98930  ...  11.97  4.41  10.78  2.42
 # ...   ...      ...      ...      ...  ...    ...   ...    ...   ...
 # 9995   15  0.23929  0.30265  0.39929  ...  12.68  4.11  12.31  0.10
 # 9996   20  0.02583  0.00946  0.03650  ...   8.46  4.11  10.46  3.12
 # 9997   10  0.57589  0.62976  0.70571  ...   9.84  3.20  10.45  2.06
 # 9998   15  1.01477  1.01504  0.99125  ...   6.38  4.06  11.28  4.03
 # 9999   10  0.24452  0.28182  0.36493  ...   9.35  4.34   9.73  3.54

 # [10000 rows x 75 columns]
# print(test.isnull().sum())
 # rho           0
 # 650_src       0
 # 660_src       0
 # 670_src       0
 # 680_src       0
 #            ...
 # 950_dst    1949
 # 960_dst    2020
 # 970_dst    1976
 # 980_dst    2011
 # 990_dst    1970
 # Length: 71, dtype: int64
test = test.interpolate() #보간법 //선형보간
print(test)
 #       rho  650_src  660_src  ...       970_dst       980_dst       990_dst 
 # id                            ...
 # 10000   15  0.15406  0.23275  ...  0.000000e+00           NaN  7.320236e-14 
 # 10001   15  0.48552  0.56939  ...  7.348414e-14  1.259055e-13  2.349874e-13 
 # 10002   10  0.46883  0.56085  ...  1.219010e-11  2.059362e-11  1.573968e-13 
 # 10003   10  0.06905  0.07517  ...  3.304247e-12  4.106134e-11  7.980625e-14 
 # 10004   25  0.00253  0.00757  ...  0.000000e+00  1.910775e-16  2.215673e-15 
 # ...    ...      ...      ...  ...           ...           ...           ... 
 # 19995   15  0.04334  0.03279  ...  1.472213e-13  1.479745e-12  5.391520e-12 
 # 19996   25  0.00020  0.02009  ...  7.432893e-14  3.928314e-18  2.586781e-17 
 # 19997   15  0.00000  0.00000  ...  1.436617e-15  0.000000e+00  1.743751e-13 
 # 19998   20  0.08390  0.05690  ...  4.489194e-18  2.892986e-17  9.689104e-13 
 # 19999   15  0.11016  0.11505  ...  1.152641e-13  5.594120e-13  1.763446e-12 

 # [10000 rows x 71 columns]
#컬럼별로 보간이기때문에 옆에 컬럼에는 영향을 미치지 않는다. 
'''
#x의 


# y_pred.to_csv(경로)
# predict할 sample-submission파일을 만든다. 

train = train.fillna(method = 'bfill')


MM = MinMaxScaler()
train = MM.fit_transform(train)
test1 = MM.fit_transform(test)
print(train)
print(train)


x_train = train[:,:71]
y_train = train[:,71:75]
#d이 슬라이싱 방식은 numpy형식일때 사용 된다, 우리가 이미 numpy로 바꾸어서 정보를 가져와서 이렇게 사용할수 있다.


print(x_train.shape)#(10000, 71)
print(y_train.shape)#(10000, 4)

print(x_train)
print(y_train)


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,shuffle=True,  test_size = 0.2, random_state=33)

def build_model(drop=0.5, optimizer= 'adam'):
    inputs = Input(shape=(71, ), name = 'input')
    x = Dense(512, activation = 'relu', name= 'hidden1')(inputs)
    x= Dropout(drop)(x)
    x = Dense(256, activation = 'relu', name= 'hidden2')(x)
    x= Dropout(drop)(x)
    x = Dense(128, activation = 'relu', name= 'hidden3')(x)
    x= Dropout(drop)(x)
    output  = Dense(4, activation = 'elu', name= 'outputs')(x)
    model = Model(inputs =inputs, outputs = output)
    model.compile (optimizer, metrics =["mae"], loss = 'mae')
    return model

def create_hyperparameters():
    batches =[40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.3,0.8, 5)
    return{"batch_size" : batches, "optimizer" : optimizers,
        "drop" :dropout}


from keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
model= KerasRegressor(build_fn= build_model, verbose= 1)
hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model, hyperparameters, cv = 4)
search.fit(x_train, y_train)
print(search.best_params_)


# acc = search.score(x_test, y_test, verbose=0)
print(search.best_params_)

###########################################
model 

model.compile(loss='mae', optimizer='rmsprop', metrics=['mae'])
model.fit(x_train,y_train, epochs=10, batch_size=5)

loss , mae = model.evaluate(x_test, y_test)

y_pred = model.predict(test1)
print(y_pred)


# # print("mae: ", mae)

# # submission = pd.DataFrame({
# #     "PassengerId": test_[:,0].astype(int),
# #     "Survived": y_pred
# # })

# a = np.arange(10000,20000)
# y_pred = pd.DataFrame(y_pred,a)
# y_pred.to_csv('./data/dacon/comp1/sample_submission.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')

# # submission.to_csv('./submit/submission_dnn.csv', index = False)
'''