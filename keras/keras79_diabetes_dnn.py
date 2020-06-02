from sklearn.datasets import load_diabetes
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# import some data

dataset = load_diabetes()
print(dataset)
print(dataset.keys())
print(dataset['feature_names'])

x = dataset.data
y = dataset.target


# scatter graph
import matplotlib.pyplot as plt
plt.figure(figsize = (20, 10))
for i in range(np.size(x, 1)):
    plt.subplot(2, 5, i+1)
    plt.scatter(x[:, i], y)
    plt.title(dataset.feature_names[i])
plt.xlabel('columns')
plt.ylabel('target')
plt.axis('equal')
plt.legend()
plt.show()



from sklearn.preprocessing import MinMaxScaler
print(x.shape) 
 # (442, 10), 여기서의 10은 데이터가 age, sex, bm1, bp, s1, s2, s3, s4, s5, s6 총 10가지로 구성되어있기 때문이다. 
print(y.shape) 
 # (442,)

scaler = MinMaxScaler()
x = scaler.fit_transform(x)
print(x)
 #나중 데이터에서 0 이나 1인 데이터를 뽑아주기 위해서 minmax scaler을 사용하여 x를 변환해준다. 


# pca = PCA(n_components=2)
# pca.fit(x_scaled)
# x_pca = pca.transform(x_scaled)
# print(x_pca)
# print(x_pca.shape)  #(442, 2)



#train test split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size = 0.8)

print(x_train.shape) #(353, 10)
print(x_test.shape)  #(89, 10)
print(y_train.shape) #(353,)
print(y_test.shape)  #(89,)


### 2. 모델
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(5, input_shape= (10, )))#dnn모델이기에 위에서 가져온 10이랑 뒤에 ',' 가 붙는다.
model.add(Dense(10, activation= 'relu'))
model.add(Dropout(0.5))

model.add(Dense(20, activation= 'relu'))
model.add(Dropout(0.5))

model.add(Dense(400, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(400, activation= 'relu'))
model.add(Dropout(0.3))

model.add(Dense(200, activation= 'relu'))
model.add(Dropout(0.3))

model.add(Dense(400, activation= 'relu'))
model.add(Dropout(0.2))

model.add(Dense(100, activation= 'relu'))
model.add(Dense(500, activation= 'relu'))

model.add(Dense(40, activation= 'relu'))
model.add(Dropout(0.2))

model.add(Dense(300, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(1))

model.summary()


# EarlyStopping
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

es = EarlyStopping(monitor = 'loss', patience=50, mode = 'auto')

cp = ModelCheckpoint(filepath='./model/{epoch:02d}-{val_loss:.4f}.hdf5', monitor='val_loss', save_best_only=True, mode='auto')

tb = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

### 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
hist = model.fit(x_train, y_train, epochs=100, batch_size=70, verbose=1, validation_split=0.025, callbacks=[es, cp, tb])


### 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=32)
print('loss:', loss)
print('mse:', mse)

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

plt.subplot(2,1,1)
plt.plot(hist.history['loss'], c='black', label ='loss')
plt.plot(hist.history['val_loss'], c='yellow', label ='val_loss')
plt.ylabel('loss')
plt.legend()

plt.subplot(2,1,2)
plt.plot(hist.history['mse'], c='red', label ='mse')
plt.plot(hist.history['val_mse'], c='green', label ='val_mse')
plt.ylabel('mse')
plt.legend()

plt.show()

#loss: 3954.6966868196982
#mse: 3954.696533203125