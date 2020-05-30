import numpy as np
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA



# 1. 데이터

dataset = load_diabetes()
print(dataset)
print(dataset.keys())
print(dataset['feature_names'])
# feature_names = dataset.feature_names
# print(f"feature_names : {feature_names}") 
#  #['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']  

x = dataset.data
y = dataset.target





from sklearn.preprocessing import MinMaxScaler
print(x.shape) 
 # (442, 10), 여기서의 10은 데이터가 age, sex, bm1, bp, s1, s2, s3, s4, s5, s6 총 10가지로 구성되어있기 때문이다. 
print(y.shape) 
 # (442,)

scaler = MinMaxScaler()
x = scaler.fit_transform(x)
print(x)
 #나중 데이터에서 0 이나 1인 데이터를 뽑아주기 위해서 minmax scaler을 사용하여 x를 변환해준다. 

x= x.reshape(x.shape[0], 2, 5)
#we bring the data of 353 and input it to 0 and from the model itself and we will split the 10 into 2 by5 this is inorder to fit in the data into the model we are working on which needs 3 diff type of num 


#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size = 0.8)

print(x_train.shape) #(353, 2, 5)
print(x_test.shape)  #(89, 2, 5)
print(y_train.shape) #(353,)
print(y_test.shape)  #(89,)


print(f"x_train.shape : {x_train.shape}") # x_train.shape :  (353, 2,5)
print(f"y_train : {y_train}")

# 2.  모델
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(100)(input_shape=(2,5)))
model.add(Dense(512,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()


# 3. 컴파일(훈련준비),실행(훈련)
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

es = EarlyStopping(monitor = 'val_loss', patience=100, mode = 'auto')

cp = ModelCheckpoint(filepath='./model/{epoch:02d}-{val_loss:.5f}.hdf5', monitor='val_loss', save_best_only=True, mode='auto')

tb = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)

model.compile(optimizer='adam',loss = 'mse', metrics = ['mse'])

hist = model.fit(x_train,y_train,epochs=12, batch_size=110, validation_split= 0.3, callbacks=[es, cp, tb])

plt.plot(hist.history['loss'])
plt.plot(hist.history['acc'])
plt.title('keras54 loss plot')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss','train acc'])
# plt.show()

### 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=32)
print('loss:', loss)
print('mse:', mse)

plt.subplot(2,1,1)
plt.plot(hist.history['loss'], c='black', label ='loss')
plt.plot(hist.history['val_loss'], c='blue', label ='loss')
plt.ylabel('loss')
plt.legend()

plt.subplot(2,1,2)
plt.plot(hist.history['acc'], c='blue', label ='loss')
plt.plot(hist.history['val_acc'], c='blue', label ='loss')
plt.ylabel('acc')
plt.legend()

plt.show()

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
