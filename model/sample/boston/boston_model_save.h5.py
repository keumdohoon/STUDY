
import numpy as np
from sklearn.datasets import load_boston
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from keras.callbacks import EarlyStopping, TensorBoard


dataset = load_boston()
x = dataset.data
y = dataset.target

print(x.shape) # (506, 13)
print(y.shape) # (506,)

###

scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)
print(x_scaled)


pca = PCA(n_components=2)
pca.fit(x_scaled)
x_pca = pca.transform(x_scaled)
print(x_pca)
print(x_pca.shape)


###
#train test split

x_train, x_test, y_train, y_test = train_test_split(
    x_pca, y, random_state=66, shuffle=True,
    train_size = 0.8)

print(x_train.shape) #(404, 2)
print(x_test.shape)  #(102, 2)
print(y_train.shape) #(404,)
print(y_test.shape)  #(102,)


### 2. 모델


model = Sequential()

model.add(Dense(100, input_shape= (2, )))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(400))
model.add(Dense(500))
model.add(Dense(400))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(1))

model.summary()
# model.save('./model/sample/boston/boston_model_save.h5')



# tb_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

### 3. 훈련

#3. 훈련

model.compile(loss = 'mse',
              optimizer = 'adam', metrics = ['mse'])
modelpath = './model/sample/boston/{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, save_weights_only=False, mode='auto')

early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto' )
hist = model.fit(x_train, y_train, epochs = 300, batch_size = 32, validation_split=0.25, verbose=1, callbacks=[cp, early_stopping])
model.save('./model/sample/boston/boston_model_save.h5')
model.save_weights('./model/sample/boston/boston_save_weights.h5')

### 4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc= hist.history['val_acc']


print("loss : {loss}", loss)
print("acc : {acc}", acc)
print("val_acc: ", val_acc)
print("loss_acc: ", loss_acc)

# # RMSE 구하기
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# print("RMSE : ", RMSE(y_test, y_predict))

# # R2 구하기
# r2 = r2_score(y_test, y_predict)
# print("R2 : ", r2)
