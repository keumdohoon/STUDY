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
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, Dense
from keras.layers import MaxPooling2D
import numpy as np
from sklearn.preprocessing import StandardScaler
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


print(train)




x = train.iloc[:, :71]
y = train.iloc[:, 71:]

stan = StandardScaler()
x = stan.fit_transform(x)


# x = x.reshape(10000,71,1).astype('float')
print(x.shape)
print(y.shape)
print(x)
print(y)


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

model = Sequential()
model.add(Dense(10, input_dim=71))
model.add(Dense(100, activation='relu'))
model.add(Dense(4))


# model.add(Dense(5, input_dim=1))
# model.add(Dense(3))
# # model.add(Dense(1000000))
# # model.add(Dense(1000000))
# # model.add(Dense(1000000))
# # model.add(Dense(1000000))
# # model.add(Dense(1000000))
# model.add(Dense(500))
# model.add(Dense(500))
# model.add(Dense(500))
# model.add(Dense(500))
# model.add(Dense(500))
# model.add(Dense(1))




model.compile(loss='mae', optimizer='rmsprop', metrics=['mae'])
model.fit(x_train,y_train, epochs=10, batch_size=5)

loss , mae = model.evaluate(x_test, y_test)

print("mae: ", mae)
