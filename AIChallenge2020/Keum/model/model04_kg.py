import cv2
import glob
import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, f1_score
from keras.layers import LeakyReLU, Activation, UpSampling2D
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception
import matplotlib.pyplot as plt
leaky = LeakyReLU(alpha = 0.2) 
leaky.__name__ = 'leaky'

#1. data
x_train = np.load('/tf/notebooks/Keum/data/x_train.npy')
x_pred = np.load('/tf/notebooks/Keum/data/x_test.npy')
x_val = np.load('/tf/notebooks/Keum/data/x_val.npy')

# print(x_train.shape)
# print(x_pred.shape)
# print(x_val.shape)
# x_train = x_train.reshape(x_train, [4])
# x_pred = x_pred.reshape(x_pred, [4])
# x_val = x_val.reshape(x_val, [4])
# print(x_train.shape)
from tensorflow.keras import regularizers

#1. data
x_train = np.load('/tf/notebooks/Keum/data/x_train.npy').reshape(-1, 384, 384, 1)
x_pred = np.load('/tf/notebooks/Keum/data/x_test.npy').reshape(-1, 384, 384, 1)
x_val = np.load('/tf/notebooks/Keum/data/x_val.npy').reshape(-1, 384, 384, 1)

print(x_train.shape)
print(x_pred.shape)
print(x_val.shape)

y_train = np.load('/tf/notebooks/Keum/data/y_train.npy')
y_val = np.load('/tf/notebooks/Keum/data/y_val.npy')
print(y_train.shape)
print(y_val.shape)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2,
                                                    shuffle= True, random_state = 66)


# #2. model
inputs = Input(shape=(384, 384, 1))
#encoder
net = Conv2D(32, kernel_size=3, activation='relu', padding='same')(inputs)
net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Conv2D(64, kernel_size=3, activation='relu', padding='same')(net)
net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Conv2D(128, kernel_size=3, activation='relu', padding='same')(net)
net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Dense(128, activation='relu')(net)

#decoder
net = UpSampling2D(size=2)(net)
net = Conv2D(128, kernel_size=3, activation='sigmoid', padding='same')(net)

net = UpSampling2D(size=2)(net)
net = Conv2D(64, kernel_size=3, activation='sigmoid', padding='same')(net)

net = UpSampling2D(size=2)(net)
outputs = Conv2D(1, kernel_size=3, activation='sigmoid', padding='same')(net)

model = Model(inputs=inputs, ouputs=outputs)




es = EarlyStopping(monitor = 'loss', patience = 20, verbose = 1)

modelpath = '/tf/notebooks/Keum/save_model/model_save{epoch:02d} - {val_loss:.4f}.hdf5'                         
checkpoint = ModelCheckpoint(filepath= modelpath, monitor='val_loss',
                            save_best_only = True, mode = 'auto', save_weights_only= False)    

#compile fit
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', 'mse'])
model.summary()

history = model.fit(x_train, y_train, validation_data = (x_val, y_val),
epochs=1, batch_size=32, callbacks=[ es,
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=1,
    mode='auto', min_lr=1e-05)])

fig,ax = plt.subplots(2, 2, figsize=(10,7))

ax[0, 0].set_title('loss')
ax[0, 0].plot(history.history['loss','r'])
ax[0, 0].set_title('acc')
ax[0, 0].plot(history.history['acc'], 'b')

ax[0, 0].set_title('val_loss')
ax[0, 0].plot(history.history['val_loss','r--'])
ax[0, 0].set_title('val_acc')
ax[0, 0].plot(history.history['val_acc'], 'b--')

preds = model.predict(x_val)

fig, ax = plt.subplots(len(x_val),3, figsize=(10, 100))

# model.save
model.save('/tf/notebooks/Keum/save_model/model04.h5')

#4. eavluate
loss_acc = model.evaluate(x_test, y_test, batch_size = 32)
print('loss_acc : ', loss_acc)

 

y_pred = model.predict(x_pred)
y_pred = y_pred.reshape(-1, )
y_predict = np.where(y_pred >= int(0.5), int(1), int(0))
print(y_predict)
# submission
def submission(y_sub):
    for i in range(len(y_sub)):
        filesnames = glob.glob('/tf/notebooks/Keum/data/test/*.png')
        f = open('/tf/notebooks/Keum/sub/submission_2.txt', 'a')
        f.write(filesnames[i]+' '+str(y_sub[i]) + '\n')
    print('complete')

submission(y_predict)
