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

train_features = pd.read_csv('./data/dacon/comp3/KAERI_dataset/train_features.csv')
test_features = pd.read_csv('./data/dacon/comp3/KAERI_dataset/test_features.csv')
train_target = pd.read_csv('./data/dacon/comp3/KAERI_dataset/train_target.csv',index_col='id')

print('train_feature.shape : ', train_features.shape)         #(1050000, 6)
print('test_features.shape : ', test_features.shape)          #(262500, 6)
print('train_target.shape : ', train_target.shape)            #(2800   , 4)


def preprocessing_KAERI(data) :
    _data = data.groupby('id').head(30)
    _data['Time'] = _data['Time'].astype('str')
    _data = _data.pivot_table(index = 'id', columns = 'Time', values = ['S1', 'S2', 'S3', 'S4'])
    _data.columns = ['_'.join(col) for col in _data.columns.values]
    return _data

train_features = preprocessing_KAERI(train_features)
test_features = preprocessing_KAERI(test_features)

print(f'train_features {train_features.shape}')
print(f'test_features {test_features.shape}')

print(train_features.shape)  #(2800, 120) 
print(test_features.shape)   #(700, 120)

#모델 구축
import sklearn
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_jobs=-1, random_state=0)

#검증

model.fit(train_features, train_target)


y_pred = model.predict(test_features)


# submit = pd.read_csv('./data/dacon/comp3/sample_submission.csv')

# submit.head()

# for i in range(4):
#     submit.iloc[:,i+1] = y_pred[:,i]
# submit.head()

# submit.to_csv('Dacon_baseline.csv', index = False)

print("Predict : \n", y_pred)


submit = pd.DataFrame(y_pred)
print(submit.head())

submit.to_csv('./data/dacon/comp3/KAERI_dataset/sample_submission.csv')




























