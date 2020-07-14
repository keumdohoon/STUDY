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
print("trainshape", train.shape)

#x의 
x1_train = train.iloc[:,0]   
y1_train = train.iloc[:,1:36]
x2_train = train.iloc[:,36:71]
y2_train = train.iloc[:,-4:]


print(x1_train)
print(y1_train)
print(x2_train)
print(y2_train)

x1_predict = test.iloc[1:,1:36]
y1_predict = test.iloc[1:,-4:]
x2_predict = test.iloc[1:,36:71]
y2_predict = test.iloc[1:,-4:]
# y_pred.to_csv(경로)
# predict할 sample-submission파일을 만든다. 
print(x1_train.shape)   #(9999, 35)
print(y1_train.shape)  #(9999, 4)
print(x2_train.shape)  #(9999, 35)
print(y2_train.shape)  #(9999, 4)
print('test.shape : ', test.shape)   #test.shape :  (10000, 71)

print(type(x1_train))
print(type(y1_train))




x1_train = x1_train.fillna(method = 'bfill')
x2_train = x2_train.fillna(method = 'bfill')

x1_predict = x1_predict.fillna(method ='bfill')
x2_predict = x2_predict.fillna(method ='bfill')

# print("np_test print:", type(np_test))
# print("np_test print:", np_test)
# print("np_test print:", np_test.shape)



ss = StandardScaler()
ss.fit(x1_train)
x1_train = ss.transform(x1_train)
ss.fit(x2_train)
x2_train = ss.transform(x2_train)
# ss.fit(np_test)
# np_test = ss.transform(np_test)
ss = StandardScaler()
ss.fit(x1_predict)
x1_predict = ss.transform(x1_predict)
ss.fit(x2_predict)
x2_predict = ss.transform(x2_predict)


# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, shuffle=True,  test_size = 0.2, random_state=33)
x1_train, x1_test, x2_train, x2_test = train_test_split(x1_train, x2_train, shuffle=True, random_state=33,
    train_size = 0.8)


print(x1_predict.shape)     #(9999, 35)
# print(x1_predicttest.shape)    #(2000, 35)
print(x2_predict.shape)     #(9999, 35)
# print(x2_predicttest.shape)    #(2000, 35)

print(x1_train.shape)   #(7999, 35)
print(x2_train.shape)   #(7999, 35)
print(x1_test.shape)   #(2000, 35)
print(x2_test.shape)   #(2000, 35)

x1_predict, x1_predicttest, x2_predict, x2_predicttest = train_test_split(x1_predict, x2_predict, shuffle=True, random_state=33,
    train_size = 0.8)

y1_predict, y1_predicttest, y2_predict, y2_predicttest = train_test_split(y1_predict, y2_predict, shuffle=True, random_state=33,
    train_size = 0.8)

# print(x1_predict.shape)     #(7999, 35)
# print(x1_predicttest.shape)    #(2000, 35)
# print(x2_predict.shape)     #(7999, 35)
# print(x2_predicttest.shape)    #(2000, 35)

# print(x1_train.shape)   #(7999, 35)
# print(x2_train.shape)   #(7999, 35)
# print(x1_test.shape)   #(2000, 35)
# print(x2_test.shape)   #(2000, 35)
# x1_predict = test[1:,1:36]
# x2_predict = test[1:,-4:]



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
middle1 = Dense(80)(middle1)
middle1 = Dense(160)(middle1)
middle1 = Dense(80)(middle1)
middle1 = Dense(13)(middle1)
middle1 = Dense(40)(middle1)
middle1 = Dense(20)(middle1)
middle1 = Dense(10)(middle1)
####output모델구성######
output1_1 = Dense(10)(middle1)
output1_2 = Dense(7)(output1_1)
output1_2 = Dense(7)(output1_2)
output1_3 = Dense(4)(output1_2)
#input1 and input 2 will be merged into one. 

output2_1 = Dense(10)(middle1)
output2_2 = Dense(7)(output2_1)
output2_3 = Dense(7)(output2_2)
output2_4 = Dense(4)(output2_3)
model = Model(inputs = [input1, input2],
 outputs = [output1_3, output2_4])

model.summary()
# print(model)

# print(x1_predict.shape)     #(7999, 35)
# print(x1_predicttest.shape)    #(2000, 35)
# print(x2_predict.shape)     #(7999, 35)
# print(x2_predicttest.shape)    #(2000, 35)
# print(y1_predict.shape)      #(7999, 4)
# print(y2_predict.shape)      #(7999, 4)
# print(y1_predicttest.shape)  #(2000, 4)
# print(y2_predicttest.shape)  #(2000, 4)

# print(x1_train.shape)   #(7999, 35)
# print(x2_train.shape)   #(7999, 35)
# print(x1_test.shape)   #(2000, 35)
# print(x2_test.shape)   #(2000, 35)



#3. 훈련
model.compile(loss='mae', optimizer = 'adam', metrics=['mae'])
model.fit([x1_train, x2_train], [y1_predict, y2_predict], epochs=1, batch_size=32, validation_split=0.25, verbose=1)

#validation_data=(x_val, y_val))

#4, evaluate
loss, loss1, loss2, mse1,mse2  = model.evaluate([x1_test,x2_test], [y1_predicttest,y2_predicttest], batch_size = 32)
# loss = loss
print('loss : ',loss)
print('mae : ',(mse1+mse2)/2)

# m  = loss[-2] 
# print("loss :", loss[0])
# print("mae :", m)


print('x1_test',x1_test.shape)
print('x2_test',x2_test.shape)
# x1_test (2000, 35)
# x2_test (2000, 35)

y_pred = model.predict([x1_test, x2_test])
print('y_pred', y_pred)
# print('y_predshape', y_pred.shape)

print(y_pred)



a = np.arange(10000,20000)
y_pred = pd.DataFrame(y_pred,a)
y_pred.to_csv('./data/dacon/comp1/sample_submission.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')

# submission.to_csv('./submit/submission_dnn.csv', index = False)
