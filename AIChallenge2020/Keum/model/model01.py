import cv2
import glob
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, f1_score
from keras.layers import LeakyReLU
leaky = LeakyReLU(alpha = 0.2) 

#1. data
x_train = np.load('/tf/notebooks/Keum/data/x_train.npy').reshape(-1, 384, 384, 1)
x_pred = np.load('/tf/notebooks/Keum/data/x_test.npy').reshape(-1, 384, 384, 1)
x_val = np.load('/tf/notebooks/Keum/data/x_val.npy').reshape(-1, 384, 384, 1)

print(x_train.shape)
print(x_pred.shape)
print(x_val.shape)

y_train = np.load('/tf/notebooks/Keum/data/y_train.npy')
y_val = np.load('/tf/notebooks/Keum/data/y_test.npy')
print(y_train.shape)
print(y_val.shape)

#2. model
def build_model(act = 'elu', drop = 0.3):
    input1 = Input(shape = (384, 384, 1))
    x = Conv2D(50, (3, 3), padding= 'same', activation = act)(input1)
    x = MaxPooling2D(pool_size= 2)(x)
    x = Dropout(drop)(x)
    x = Conv2D(150, (3, 3), padding= 'same', activation = act)(x)
    x = MaxPooling2D(pool_size= 2)(x)
    x = Dropout(drop)(x)
    x = Conv2D(250, (3, 3), padding= 'same', activation = act)(x)
    x = MaxPooling2D(pool_size= 2)(x)
    x = Dropout(drop)(x)
    x = Conv2D(150, (3, 3), padding= 'same', activation = act)(x)
    x = MaxPooling2D(pool_size= 2)(x)
    x = Dropout(drop)(x)
    x = Flatten()(x)
    x = Dense(100, activation = act)(x)
    x = Dropout(drop)(x)
    x = Dense(80, activation = act)(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation = act)(x)
    x = Dropout(drop)(x)
    output = Dense(1, activation = 'sigmoid')(x)

    model = Model(inputs = input1, outputs = output)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

    return model

seed = 7
np.random.seed(seed)

model = KerasClassifier(build_fn= build_model, epochs = 1, batch_size = 128, verbose = 2)

kfold = KFold(n_splits = 3, shuffle = True, random_state = seed)
result = cross_val_score(model, x_train, y_train, cv = kfold)

y_pred = cross_val_predict(model, x_val, y_val, cv = kfold)
score = search.score(x_test, y_test, verbose=0)
score = f1_score(y_val, y_pred)
print('score : ', score)

# submit_data
y_predict = model.predict(x_pred)
y_predict = np.where(y_predict >= 0.5, 1, 0)

# submission
def submission(y_sub):
    for i in range(len(y_sub)):
        filesnames = glob.glob('/tf/notebooks/Keum/data/test/*.png')
        f = open('tf/notebooks/Keum/sub/submission.txt', 'a')
        f.write(filesnames[i]+' '+str(y_sub[i]) + '\n')
    print('complete')

submission(y_predict)
