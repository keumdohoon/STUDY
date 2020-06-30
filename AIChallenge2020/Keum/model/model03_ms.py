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

#get image 에서 png파일을 npy형식으로 바꾸어준 파일들을 불러와준다. 
#각각의 파일들을 불러 와준다.


from tensorflow.keras import regularizers

#1. data
x_train = np.load('/tf/notebooks/Keum/data/x_train.npy').reshape(-1, 384, 384, 1)
x_pred = np.load('/tf/notebooks/Keum/data/x_test.npy').reshape(-1, 384, 384, 1)
x_val = np.load('/tf/notebooks/Keum/data/x_val.npy').reshape(-1, 384, 384, 1)

#각각의 불러와준 파일들을  reshapping해준다. 


print(x_train.shape)
print(x_pred.shape)
print(x_val.shape)

y_train = np.load('/tf/notebooks/Keum/data/y_train.npy')
y_val = np.load('/tf/notebooks/Keum/data/y_val.npy')
print(y_train.shape)
print(y_val.shape)
#get label에서txt-> npy 형태로 바꾸어준 파일들을 불러와준다. 
#각각의 레이블들을 불러와준다. 


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2,
                                                    shuffle= True, random_state = 66)


# #2. model
model = Sequential()
model.add(Conv2D(64, (3,3), padding="same", input_shape=x_train.shape[1:], activation="relu",kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(pool_size=(2,2)))
#x_train.shape 를 해주면 (384, 384, 1)이렇게 원하는 값이 나온다, 이는 (100, 384, 384, 1)에서 앞에 100을 빼준 값이다. 

model.add(Conv2D(60, (3,3), padding="same", activation="relu",kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(82, (3,3), padding="same", activation="relu",kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(128, (3,3), padding="same", activation="relu",kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), padding="same", activation="relu",kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(32, kernel_initializer='he_normal'))
model.add(Dense(1, activation = 'sigmoid'))

# callbacks
es = EarlyStopping(monitor = 'loss', patience = 3, verbose = 1)

modelpath = '/tf/notebooks/Keum/save_model/model_save{epoch:02d} - {val_loss:.4f}.hdf5'     
#save_model 이라는 폴더에 모델을 저장해준다.                     
checkpoint = ModelCheckpoint(filepath= modelpath, monitor='val_loss',
                            save_best_only = True, mode = 'auto', save_weights_only= False)                
#checkpoint 는 좋은 결과값만을 저장하게 해준다.ㅏ 
#3. compile, fit
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 10, batch_size = 32, verbose = 1,
                            callbacks = [es, checkpoint],
                            validation_data = (x_val, y_val))
#callbacks에 제대로 들어가있는지 봐주기
# model.save
model.save('/tf/notebooks/Keum/save_model/model02_1.h5')

#4. eavluate
loss_acc = model.evaluate(x_test, y_test, batch_size = 32)
print('loss_acc : ', loss_acc)

# y_pred = model.predict(x_test)
# y_pred = np.where(y_pred >= int(0.5), int(1), int(0))
# acc = accuracy(y_test, y_pred)
# print('acc : ', acc)
# # f1_score = f1_score(y_test, y_pred)
# # print('f1_score : ', f1_score)


# y_pred = model.predict(x_val)
# y_pred = np.where(y_pred >= int(0.5), int(1), int(0))
# score = accuracy(y_val, y_pred)
# print('acc : ', score)
# # score = f1_score(y_val, y_pred)
# # print('score : ', score)

y_pred = model.predict(x_pred)
y_pred = y_pred.reshape(-1, )
#이걸 해줘야지 결과에서 []가 없어진다. 
y_predict = np.where(y_pred >= int(0.5), int(1), int(0))
print(y_predict)
# submission
def submission(y_sub):
    for i in range(len(y_sub)):
        filesnames = glob.glob('/tf/notebooks/Keum/data/test/*.png')
        f = open('/tf/notebooks/Keum/sub/submission_final.txt', 'a')
        f.write(filesnames[i]+' '+str(y_sub[i]) + '\n')
    print('complete')

submission(y_predict)
