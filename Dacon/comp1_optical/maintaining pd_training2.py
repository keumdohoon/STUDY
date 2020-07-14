import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

train = pd.read_csv('./data/dacon/comp1/train.csv', header=0, index_col=2)
test = pd.read_csv('./data/dacon/comp1/test.csv', header=0, index_col=2)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header=0, index_col=2)

print('train.shape : ', train.shape)         #x_train, test #train.shape :  (10000, 75)
print('test.shape : ', test.shape)           #x_predict#test.shape :   (10000, 71)
print('summit.shape : ', submission.shape)   #y_predict#summit.shape : (10000, 4)
#test는 지금 xpredict밖에 되지 않음 

print(train.isnull().sum())
train = train.interpolate() #보간법 //선형보간
# print(train.isnull().sum())
test = test.interpolate() #보간법 //선형보간
#컬럼별로 보간이기때문에 옆에 컬럼에는 영향을 미치지 않는다. 
print("test", test)
print("test", type(test))
print("test", test.shape)

#x의 
x_train = train.iloc[:,:71]
y_train = train.iloc[:,-4:]

# y_pred.to_csv(경로)
# predict할 sample-submission파일을 만든다. 
print(x_train.shape)
print(y_train.shape)
print(type(x_train))
print(type(y_train))


x_train = x_train.fillna(method = 'bfill')
test = test.fillna(method ='bfill')
np_test = test.values

print("np_test print:", type(np_test))
print("np_test print:", np_test)
print("np_test print:", np_test.shape)



ss = StandardScaler()
ss.fit(x_train)
x_train = ss.transform(x_train)
np_test = ss.transform(np_test)



x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, shuffle=True,  test_size = 0.2, random_state=33)


model = Sequential()
inputs = Input(shape=(71, ), name = 'input')
x = Dense(512, activation = 'relu', name= 'hidden1')(inputs)
x= Dropout(0.5)(x)
x = Dense(256, activation = 'relu', name= 'hidden2')(x)
x= Dropout(0.5)(x)
x = Dense(128, activation = 'relu', name= 'hidden3')(x)
x= Dropout(0.5)(x)
output  = Dense(4, activation = 'elu', name= 'outputs')(x)
model = Model(inputs =inputs, outputs = output)


model.compile(loss='mae', optimizer='rmsprop', metrics=['mae'])
model.fit(x_train, y_train, epochs=1, batch_size=5)



print('np_test', np_test.shape)
#4, evaluate
loss , mae = model.evaluate(x_test, y_test, batch_size = 5)
print("loss :", loss)
print("mae :", mae)

y_pred = model.predict(np_test)
print(y_pred)


# })

a = np.arange(10000,20000)
y_pred = pd.DataFrame(y_pred,a)
y_pred.to_csv('./data/dacon/comp1/sample_submission.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')

# submission.to_csv('./submit/submission_dnn.csv', index = False)
