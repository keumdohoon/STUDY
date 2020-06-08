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
train = train.interpolate() #보간법 //선형보간
# print(train.isnull().sum())
test = test.interpolate() #보간법 //선형보간
#컬럼별로 보간이기때문에 옆에 컬럼에는 영향을 미치지 않는다. 

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
