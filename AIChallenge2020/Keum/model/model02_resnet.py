import cv2
import glob
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, f1_score
from keras.layers import LeakyReLU, Flatten
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception

leaky = LeakyReLU(alpha = 0.2) 
leaky.__name__ = 'leaky'

#1. data
x_train = np.load('/tf/notebooks/Keum/data/x_train.npy')
x_pred = np.load('/tf/notebooks/Keum/data/x_test.npy')
x_val = np.load('/tf/notebooks/Keum/data/x_val.npy')

print(x_train.shape)
print(x_pred.shape)
print(x_val.shape)
# x_train = x_train.reshape(x_train, [4])
# x_pred = x_pred.reshape(x_pred, [4])
# x_val = x_val.reshape(x_val, [4])
# print(x_train.shape)



y_train = np.load('/tf/notebooks/Keum/data/y_train.npy')
y_val = np.load('/tf/notebooks/Keum/data/y_test.npy')
print('y_train',y_train.shape) #y_train (546,)
print('y_val',y_val.shape) #y_val (100,)

######################
print(x_train.shape)  # (546, 384, 384)


x_train = np.repeat(x_train[..., np.newaxis], 3, -1)
x_pred = np.repeat(x_pred[..., np.newaxis], 3, -1)
# x_val = np.repeat(x_val[..., np.newaxis], 3, -1)
print(x_train.shape)  # (546, 384, 384, 3)
print(x_pred.shape)  # (100, 384, 384, 3)
print(x_val.shape)  # (100, 384, 384, 3)


y_train = np.repeat(y_train[..., np.newaxis], 3, -1)
y_val = np.repeat(y_val[..., np.newaxis], 3, -1)
print('y_train',y_train.shape) #y_train (546, 3)
print('y_val',y_val.shape) #y_val (100, 3)

print(x_val)
print(y_val)


conv_base = Xception(weights ='imagenet', include_top=False, 
                        input_shape =(384,384,3))

conv_base.summary()

from keras import models, layers
model = models.Sequential()
model.add(conv_base)
model.add(layers.Dense(256, activation ='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2,
#                                                     shuffle= True, random_state = 66)


# #2. model
# input1 = Input(shape = (384, 384, 1))
# x = Conv2D(50, (3, 3), padding= 'same', activation = 'elu')(input1)
# x = MaxPooling2D(pool_size= 2)(x)
# x = Dropout(0.3)(x)
# x = Conv2D(150, (3, 3), padding= 'same', activation = 'elu')(x)
# x = MaxPooling2D(pool_size= 2)(x)
# x = Dropout(0.3)(x)
# x = Conv2D(250, (3, 3), padding= 'same', activation = 'elu')(x)
# x = MaxPooling2D(pool_size= 2)(x)
# x = Dropout(0.3)(x)
# x = Conv2D(350, (3, 3), padding= 'same', activation = 'elu')(x)
# x = Dropout(0.3)(x)
# x = Conv2D(450, (3, 3), padding= 'same', activation = 'elu')(x)
# x = Dropout(0.3)(x)
# x = Faltten()(x)
# x = Dense(300, activation = 'elu')(x)
# x = Dropout(0.3)(x)
# x = Dense(250, activation = 'elu')(x)
# x = Dense(150, activation = 'elu')(x)
# x = Dense(100, activation = 'elu')(x)
# x = Dense(50, activation = 'elu')(x)
# output = Dense(1, activation = 'sigmoid')(x)
# model = Model(inputs = input1, outputs = output)

# callbacks
es = EarlyStopping(monitor = 'loss', patience = 50, verbose = 1)

modelpath = '/tf/notebooks/Keum/save_model/model_save{epoch:02d} - {val_loss:.4f}.hdf5'                         
checkpoint = ModelCheckpoint(filepath= modelpath, monitor='val_loss',
                            save_best_only = True, mode = 'auto', save_weights_only= False)                

#3. compile, fit
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 300, batch_size = 128, verbose = 2,
                            callbacks = [es, checkpoint],
                            validation_data = (x_val, y_val))

# model.save
model.save('/tf/notebooks/Keum/save_model/model02.h5')

#4. eavluate
loss_acc = model.evaluate(x_test, y_test, batch_size = 128)
print('loss_acc : ', loss_acc)

y_pred = model.predict(x_test)
f1_score = f1_score(y_test, y_pred)
print('f1_score : ', f1_score)


