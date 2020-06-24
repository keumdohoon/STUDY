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

train = pd.read_csv('./data/dacon/comp1/train.csv', header=0, index_col=0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header=0, index_col=0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header=0, index_col=0)


#1데이터 분석 
#Shape
# print(train.shape)      #(10000, 75)
# print(test.shape)        #(10000, 71)
# print(submission.shape) #(10000, 4)


print(train.isnull().sum())                      #null값에 대한 summary

train = train.interpolate()
test = test.interpolate()


#판다스를 활용한 데이터셋 컷
x = train.iloc[:, :71]
y = train.iloc[:, -4:]
# print(x.shape)  #(10000, 71)
# print(y.shape)   #(10000, 4)

x = x.fillna(method = 'bfill')
test = test.fillna(method = 'bfill')
np_submission= submission.values
np_test = test.values
np_train= train.values       
# print(np_submission.shape)  #(10000, 4)
# print(np_test.shape)  #(10000, 71)
# print(np_train.shape) #(10000, 75)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x= scaler.transform(x)
np_test = scaler.transform(np_test)

#스플릿
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True)
# print(x_train.shape)    # (8000, 71)
# print(x_test.shape)     # (2000, 71)
# print(y_train.shape)    # (8000, 4)
# print(y_test.shape)     # (2000, 4)



parameters ={
    'rf__n_estimators' : [100],
    'rf__max_depth' : [10],
    'rf__min_samples_leaf' : [ 3],
    'rf__min_samples_split' : [5]
}
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor


''' 2. 모델 '''
pipe = Pipeline([('scaler', MinMaxScaler()), ('rf', RandomForestRegressor())])
kfold = KFold(n_splits=5, shuffle=True)
model = RandomizedSearchCV(pipe, parameters, cv=kfold)

''' 3. 훈련 '''
model.fit(x_train, y_train)


''' 4. 평가, 예측 '''
score = model.score(x_test, y_test)

print('최적의 매개변수 :', model.best_params_)
print('score :', score)


y_pred = model.predict(x_pred)
# print(y_pred)
y_pred1 = model.predict(x_test)

'''
ss = StandardScaler()
ss.fit(x1_train)
x1_train = ss.transform(x1_train)
ss.fit(x2_train)
x2_train = ss.transform(x2_train)
# ss.fit(test)
# test = ss.transform(test)


# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, shuffle=True,  test_size = 0.2, random_state=33)
x1_train, x1_test, x2_train, x2_test = train_test_split(x1_train, x2_train, shuffle=True, random_state=33,
    train_size = 0.8)


x_test = test[:,1:36]
y_test = test[:,-4:]
x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, shuffle=True, random_state=33,
    train_size = 0.8)

print(x_test.shape)     #(2000, 35)
print(x_train.shape)    #(7999, 35)
print(y_test.shape)     #(2000, 4)
print(y_train.shape)    #(7999, 4)

print(x1_train.shape)   #(7999, 35)
print(x2_train.shape)   #(7999, 35)
print(x1_test.shape)   #(2000, 35)
print(x2_test.shape)   #(2000, 35)



#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
#model = Sequential()
#model.add(Dense(5, input_dim=3))
#model.add(Dense(4))
#model.add(Dense(1))

input1 = Input(shape=(35,))
dense1_1=Dense(5, activation='relu')(input1)
dense1_2=Dense(10,activation='relu')(dense1_1)
dense1_2=Dense(20,activation='relu')(dense1_1)
dense1_2=Dense(40,activation='relu')(dense1_1)
dense1_2=Dense(100,activation='relu')(dense1_1)
dense1_2=Dense(50,activation='relu')(dense1_1)
dense1_2=Dense(30,activation='relu')(dense1_1)
dense1_2=Dense(10,activation='relu')(dense1_1)
dense1_2=Dense(4,activation='relu')(dense1_2)



input2 = Input(shape=(35, ))
dense2_1=Dense(5, activation='relu')(input1)
dense2_2=Dense(10,activation='relu')(dense2_1)
dense2_2=Dense(20,activation='relu')(dense2_1)
dense2_2=Dense(40,activation='relu')(dense2_1)
dense2_2=Dense(100,activation='relu')(dense2_1)
dense2_2=Dense(50,activation='relu')(dense2_1)
dense2_2=Dense(30,activation='relu')(dense2_1)
dense2_2=Dense(10,activation='relu')(dense2_1)
dense2_2=Dense(4,activation='relu')(dense2_1)

from keras.layers.merge import concatenate
merge1 = concatenate([dense1_2, dense2_2], name='concatenate')

middle1 = Dense(10, name='middle')(merge1)
middle1 = Dense(20)(middle1)
middle1 = Dense(40)(middle1)
middle1 = Dense(13)(middle1)
middle1 = Dense(40)(middle1)
middle1 = Dense(20)(middle1)
middle1 = Dense(10)(middle1)
####output모델구성######
output1_1 = Dense(10)(middle1)
output1_2 = Dense(7)(output1_1)
output1_3 = Dense(4, name='finale')(output1_2)
#input1 and input 2 will be merged into one. 
model = Model(inputs = [input1, input2],
 outputs = output1_3)
model.summary()
print()




#3. 훈련
model.compile(loss='mae', optimizer = 'adam', metrics=['mae'])
model.fit([x1_train, x2_train], y_train, epochs=1, batch_size=32, validation_split=0.25, verbose=1)

#validation_data=(x_val, y_val))

#4, evaluate
loss , mae = model.evaluate([x1_test,x2_test], y_test, batch_size = 5)
print("loss :", loss)
print("mae :", mae)

xfinal=x1_train + x1_test
print(xfinal.shape)

y_pred = model.predict([x1_test,x2_test])
print(y_pred)
print('y_pred', y_pred)
print('y_predshape', y_pred.shape)

# # })

a = np.arange(10000,20000)
y_pred = pd.DataFrame(y_pred,a)
y_pred.to_csv('./data/dacon/comp1/sample_submission.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')

# submission.to_csv('./submit/submission_dnn.csv', index = False)
'''