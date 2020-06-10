import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

train_features = pd.read_csv('./data/dacon/comp3/KAERI_dataset/train_features.csv',header = 0,index_col=0)
test_features = pd.read_csv('./data/dacon/comp3/KAERI_dataset/test_features.csv',header = 0,index_col=0)
train_target = pd.read_csv('./data/dacon/comp3/KAERI_dataset/train_target.csv',header = 0,index_col=0)

print('train_feature.shape : ', train_features.shape)         #(1050000, 6)
print('test_features.shape : ', test_features.shape)          #(262500, 6)
print('train_target.shape : ', train_target.shape)            #(2800   , 4)



x_train = train_features
y_train = train_target
y_pred =test_features
print(x_train.shape)        # (1050000, 6)
print(y_train.shape)        # (2800, 5)
print(y_pred.shape)
print(x_train.head())
x_train = x_train.drop('Time', axis = 1)

x_train = np.sqrt(x_train.groupby(x_train['id']).mean())
print(x_train.shape)        # (2800, 4)

x_train = pd.read_csv('./data/dacon/comp3/KAERI_datasets/train_features.csv', header = 0, index_col=0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, 
                                                    shuffle = True, random_state = 43)


parameter = [{"svm__C":[1, 10, 100, 1000],'svm__kernel':['linear']}, 
             {"svm__C":[1, 10, 100, 1000], 'svm__kernel':['rbf'], 
                                         'svm__gamma':[0.001, 0.0001]},   
             {"svm__C":[1, 10, 100, 1000], 'svm__kernel':['sigmoid'],
                                         'svm__gamma':[0.001, 0.0001]}
            ] 

#2. 모델
model = SVC()

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])
pipe = make_pipeline(MinMaxScaler(), SVC())
pipe.fit(x_train, y_train)
print("acc : ", pipe.score(x_test, y_test))
#MinMax scaler을 이용하여 한방에 전처리까지 다 해줬다. 
model = RandomizedSearchCV(pipe, parameter, cv = 5)
#모델에 파이프를 넣어준다. 
#make_pipeline 이라는 것이 있다. 

#3. 훈련
model.fit(x_train, y_train)

#4. 평가와 예측
acc = model.score(x_test, y_test)
print('최적의 매개 변수 = ', model.best_estimator_)
print("acc : ", acc)
'''