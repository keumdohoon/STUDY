
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

# import some data

dataset = load_iris()
print("dataset", dataset)  #array([[5.1, 3.5, 1.4, 0.2]......계속 
print("keys", dataset.keys())
print(dataset['feature_names']) #['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
 #총 4개의 종류의 특성을 파악한다 위에 feature name 중에서, 그리고 각각의 길이와 넓이로 꽃을 어느꽃인지 분류하는 작업을 실시하는 데이터셋이다.
print("data: ",  dataset.data)
print('target: ', dataset.target)
x = dataset.data
y = dataset.target
print(x.shape) # (150, 4)
print(y.shape) # (150,)

y= np_utils.to_categorical(y)
 #여기서 utils는 원핫 인코딩을 사용하여 위에 지저분하게 나열된 y의 데이터를 0,0,1이나0,1,0이나1,0,0으로 정리 해준다.이것을 원핫 인코딩이라고한다.
 #print(y)

#
# scaler = StandardScaler()
# scaler.fit(x)
# x_scaled = scaler.transform(x)
# print(x_scaled)

print(x)
# pca = PCA(n_components=2)
# pca.fit(x_scaled)
# x_pca = pca.transform(x_scaled)
# print(x_pca)
# print(x_pca.shape)  #(150, 2)
scale = StandardScaler()
x = scale.fit_transform(x)
print(x)
#standaars scaler 은 각 값의 평균을 0으로 잡고서는 standard deviation을 1로 잡아주는 역할을 한다.

###
#train test split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, shuffle=True,
    train_size = 0.8)
 #데이터를 스플릿하여 train사이즈를 80프로를 주고 나머지는 테스트로 설정해둔다.
print(x_train.shape) #(120, 4)
print(x_test.shape)  #(30, 4)
print(y_train.shape) #(120,)
print(y_test.shape)  #(30,)
#총 150개의 데이터에서 120(80프로)인 120은train, 30(20프로)은 test로

### 2. 모델
from keras.models import Sequential
from keras.layers import Dense, Dropout
 #모델은 sequential을 사용해준다. 

model = Sequential()

model.add(Dense(50, input_shape= (4, )))
model.add(Dense(100, activation= 'elu'))
model.add(Dropout(0.2))

model.add(Dense(200, activation= 'elu'))
model.add(Dropout(0.2))

model.add(Dense(400, activation= 'elu'))
model.add(Dropout(0.2))
model.add(Dense(400, activation= 'elu'))
model.add(Dropout(0.2))

model.add(Dense(200, activation= 'elu'))
model.add(Dropout(0.2))

model.add(Dense(400, activation= 'elu'))
model.add(Dropout(0.2))

model.add(Dense(100, activation= 'elu'))
model.add(Dropout(0.2))

model.add(Dense(500, activation= 'elu'))
model.add(Dropout(0.2))

model.add(Dense(400, activation= 'elu'))
model.add(Dropout(0.2))

model.add(Dense(300, activation = 'softmax'))
model.add(Dropout(0.2))

model.add(Dense(3))

model.summary()

# EarlyStopping


### 3. 훈련
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
es = EarlyStopping(monitor = 'loss', patience=200, mode = 'auto')
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
tb_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_test, y_test, epochs=300, batch_size=32, validation_split=0.25,
          callbacks=[es, checkpoint, tb_hist])


### 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print('loss:', loss)
print('mse:', acc)

plt.subplot(2,1,1)
plt.plot(hist.history['loss'], c='black', label ='loss')
plt.plot(hist.history['val_loss'], c='yellow', label ='val_loss')
plt.ylabel('loss')
plt.legend()

plt.subplot(2,1,2)
plt.plot(hist.history['acc'], c='red', label ='acc')
plt.plot(hist.history['val_acc'], c='green', label ='val_acc')
plt.ylabel('acc')
plt.legend()

plt.show()



y_predict = model.predict(x_test)
print(y_predict)

# loss = hist.history['loss']
# acc = hist.history['acc']
# val_loss = hist.history['val_loss']
# val_acc = hist.history['val_acc']

# # RMSE 구하기
# from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# print("RMSE : ", RMSE(y_test, y_predict))

# # R2 구하기
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict)
# print("R2 : ", r2)
#loss: 1.1920930376163597e-07
#acc: 0.0455
