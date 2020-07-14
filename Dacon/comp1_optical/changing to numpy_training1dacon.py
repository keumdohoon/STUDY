import numpy as np
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, LSTM , Flatten, Dropout, MaxPooling2D,Input
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np


train = np.genfromtxt('./data/dacon/comp1/train.csv',delimiter=',' )
print(train)
print(type(train))  


# train = pd.read_csv('./data/dacon/comp1/train.csv', header=0, index_col=0)
# test = pd.read_csv('./data/dacon/comp1/test.csv', header=0, index_col=0)
# submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header=0, index_col=0)

# print('train.shape : ', train.shape)         #x_train, test #train.shape :  (10000, 75)
# print('test.shape : ', test.shape)           #x_predict#test.shape :   (10000, 71)
# print('summit.shape : ', submission.shape)   #y_predict#summit.shape : (10000, 4)
# #test는 지금 xpredict밖에 되지 않음 

# # print(train.isnull().sum())
# train = train.interpolate() #보간법 //선형보간
# # print(train.isnull().sum())
# test = test.interpolate() #보간법 //선형보간
# #컬럼별로 보간이기때문에 옆에 컬럼에는 영향을 미치지 않는다. 

# #x의 


# # y_pred.to_csv(경로)
# # predict할 sample-submission파일을 만든다. 

# train = train.fillna(method = 'bfill')


x_train = train[1:,1:71]
y_train = train[1:,71:75]
#d이 슬라이싱 방식은 numpy형식일때 사용 된다, 우리가 이미 numpy로 바꾸어서 정보를 가져와서 이렇게 사용할수 있다.

# print(y_wine)

print(x_train.shape)#(10000, 70)
print(y_train.shape)#(10000, 4)

print(x_train)
print(y_train)
# y_wine = np_utils.to_categorical(y_wine)
'''

x_train,x_test, y_train,y_test = train_test_split(x_train,y_train,
                                                  random_state = 66, shuffle=True,
                                                  train_size=0.8)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape) # (8000, 70)
print(y_train.shape) # (8000, 4)
print(x_test.shape) # (2000, 70)
print(y_test.shape) # (2000, 4)

# 2. model
model = Sequential()
model.add(Dense(100, input_dim=70, activation='relu'))
model.add(Dense(100))
model.add(Dense(300))
model.add(Dense(4, activation='relu'))

# 3. compile, fit
model.compile(optimizer='adam',loss = 'mae', metrics = ['mae'])

model.fit(x_train,y_train,epochs=30,batch_size=64)

# 4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test,batch_size=3)

print("loss : ",loss)
print("acc : ",acc)
'''
