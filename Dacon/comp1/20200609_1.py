#기존모델 다:다 모델레서 다:1인 모델로 바꿔준다. 아렇게 해주는 이유는 다:다 모델로 하니까 우리의 로스값이 더해져서 너무 커진다datetime A combination of a date and a time. Attributes: ()import numpy as np
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
x1_train = train.iloc[:,1:36]
y1_train = train.iloc[:,-4:]
x2_train = train.iloc[:,36:71]
y2_train = train.iloc[:,-4:]
x3_train = train.iloc[:,:1]
y3_train = train.iloc[:,-4:]
# x1= 트레인 안에서 앞에 35개
# x2 = 트레인안에서 뒤에 35개
# x3 = 트레레인 안에서 rho

# test = test.iloc[1:, :]
# y_pred.to_csv(경로)
# predict할 sample-submission파일을 만든다. 
# print(x1_train.shape)
# print(y1_train.shape)
# print(x2_train.shape)
# print(y2_train.shape)
# print(x3_train.shape)
# print(y3_train.shape)

print('test.shape : ', test.shape)#(10000, 71)

print(type(x1_train))
print(type(y1_train))


x1_train = x1_train.fillna(method = 'bfill')
x2_train = x2_train.fillna(method = 'bfill')

# test = test.fillna(method ='bfill')
# test = test.values

# print("np_test print:", type(test))
# print("np_test print:", test)
# print("np_test print:", test.shape)

print('test.shape : ', test.shape)#(10000, 71)


ss = StandardScaler()
ss.fit(x1_train)
x1_train = ss.transform(x1_train)
ss.fit(x2_train)
x2_train = ss.transform(x2_train)
ss.fit(x3_train)
x3_train = ss.transform(x3_train)
ss.fit(test)
test = ss.transform(test)


print('x1.shape : ', x1_train.shape)# (9999, 35)
print('x2.shape : ', x2_train.shape)# (9999, 35)
print('x3.shape : ', x3_train.shape)# (9999, 35)

print('test.shape : ', test.shape)#(10000, 71)



# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, shuffle=True,  test_size = 0.2, random_state=33)
x1_train, x1_test, y1_train, y1_test = train_test_split(x1_train, y1_train, shuffle=True, random_state=33,
    train_size = 0.8)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2_train, y2_train, shuffle=True, random_state=33,
    train_size = 0.8)
x3_train, x3_test, y3_train, y3_test = train_test_split(x3_train, y3_train, shuffle=True, random_state=33,
    train_size = 0.8)

# from sklearn.model_selection import train_test_split
# x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
#     x1, x2, y, shuffle=True,
#     train_size = 0.8)


print(x1_train.shape) #(8000, 35)
print(y1_train.shape) #(8000, 4)
print(x2_train.shape) #(8000, 35)
print(y2_train.shape) #(8000, 4)
print(x3_train.shape) #(8000, 1)
print(y3_train.shape) #(8000, 4)
print(x1_test.shape)  #(2000, 35)
print(y1_test.shape)  #(2000, 4)
print(x2_test.shape)  #(2000, 35)
print(y2_test.shape)  #(2000, 4)

# x_test = test[:,1:36]
# y_test = test[:,-4:]
# x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, shuffle=True, random_state=33,
#     train_size = 0.8)

# print(x_test.shape)     #(2000, 35)
# print(x_train.shape)    #(7999, 35)
# print(y_test.shape)     #(2000, 4)
# print(y_train.shape)    #(, 4)

print(x1_train.shape)   #(8000, 35)
print(x2_train.shape)   #(8000, 35)
print(x3_train.shape)   #(8000, 1)
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


input3 = Input(shape=(1, ))
dense3_1=Dense(5, activation='relu')(input1)
dense3_2=Dense(10,activation='relu')(dense3_1)
dense3_2=Dense(20,activation='relu')(dense3_1)
dense3_2=Dense(40,activation='relu')(dense3_1)
dense3_2=Dense(100,activation='relu')(dense3_1)
dense3_2=Dense(50,activation='relu')(dense3_1)
dense3_2=Dense(30,activation='relu')(dense3_1)
dense3_2=Dense(10,activation='relu')(dense3_1)
dense3_2=Dense(4,activation='relu')(dense3_1)

from keras.layers.merge import concatenate
merge1 = concatenate([dense1_2, dense2_2, dense3_2], name='concatenate')

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
output1_3 = Dense(4)(output1_2)
#input1 and input 2 will be merged into one. 
output2_1 = Dense(10)(middle1)
output2_2 = Dense(7)(output2_1)
output2_3 = Dense(4)(output2_2)

output3_1 = Dense(10)(middle1)
output3_2 = Dense(7)(output3_1)
output3_3 = Dense(4)(output3_2)
model = Model(inputs = [input1, input2, input3],
 outputs = [output1_3, output2_3, output3_3])
model.summary()
print()


print(x1_train.shape) #(8000, 35)
print(y1_train.shape) #(8000, 4)
print(x2_train.shape) #(8000, 35)
print(y2_train.shape) #(8000, 4)
print(x3_train.shape) #(8000, 1)
print(y3_train.shape) #(8000, 4)

#3. 훈련
model.compile(loss='mae', optimizer = 'adam', metrics=['mae'])
model.fit([x1_train, x2_train, x3_train], [y1_train,y2_train, y3_train], epochs=1, batch_size=32, validation_split=0.25, verbose=1)

#validation_data=(x_val, y_val))

print(x1_test.shape)#(2000, 35)
# print(y1_test.shape)#(2000, 4)
print(x2_test.shape)#(2000, 35)
# print(y2_test.shape)#(2000, 4)
x_test = np.append( x1_test, x2_test, axis=1)
print(x_test.shape)
x_test = np.append(x3_test,  x_test, axis=1)
print(x_test.shape)#(2000, 71)

print(y1_test.shape)
print(y2_test.shape)

# y_test = np.append(y1_test,  y2_test, axis=1)
# print(y_test.shape)  #(2000, 71)
# y_test = np.append(y3_test,  y_test, axis=1)
# print(y_test.shape)  #(2000, 71)

#4, evaluate
avgloss, loss1_1, loss1_2 , loss1_3, mae1, mae2, mae3 = model.evaluate([x1_test,x2_test, x3_test],[y1_test, y2_test, y3_test] , batch_size = 5)
print("avgloss :", avgloss)
print("loss1 :", loss1_1)
print("loss2 :", loss1_2)
print("loss3 :", loss1_3)

print("mae1 :", mae1)
print("mae2 :", mae2)
print("mae3 :", mae3)


print(test.shape) #(10000, 71)

# xfinal= x1_train + x1_test
# print(xfinal.shape)
y_pred = model.predict(test)

# y_pred = model.predict(test)
print(y_pred)
print('y_pred', y_pred)
print('y_predshape', y_pred.shape)
#y_predshape (2000, 4)
# # })

a = np.arange(10000,20000)
y_pred = pd.DataFrame(y_pred,a)
y_pred.to_csv('./data/dacon/comp1/sample_submission.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')

# submission.to_csv('./submit/submission_dnn.csv', index = False)
'''