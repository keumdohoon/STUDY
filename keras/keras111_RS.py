#100번을 카피해서 lr을 넣고 묶기
#LSTM -> dense 로 바꿀것

from keras.optimizers import Adam, RMSprop, SGD, Adadelta 
from keras.optimizers import Adagrad, Adamax , Nadam
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, MaxPooling2D, Dense, LSTM
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier          
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV,  RandomizedSearchCV
from keras.layers import LeakyReLU
#1. data
(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train.shape)               # (60000, 28, 28)
print(x_test.shape)                # (10000, 28, 28)

x_train = x_train.reshape(x_train.shape[0], 28*28)/225
x_test = x_test.reshape(x_test.shape[0], 28*28)/225

# one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)                                    # (60000, 10)
print(y_train[0])


print(y_train.shape)  # (60000, 9)
print(y_test.shape)  # (10000, 9)

#2. model

# gridsearch에 넣기위한 모델(모델에 대한 명시 : 함수로 만듦)
def build_model(drop, optimizer, lr, activation ): # 여기에도 learning_rate,epoch 변수 추가 해준다 
    inputs = Input(shape= (784,), name = 'input')
    x = Dense(51, activation = activation, name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation = activation, name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation = activation, name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation = activation, name = 'output')(x)

    model = Model(inputs = inputs, outputs = outputs)
    model.compile( optimizer=optimizer(lr = lr), metrics = ['acc'],
                  loss = 'categorical_crossentropy')
    return model

import tensorflow as tf
sin = tf.math.sin
leaky = LeakyReLU(0.2)
#파라미터들
# adam = Adam(lr=0.001)
# rmsprop = RMSprop(lr=0.001)
# SGD = SGD(lr=0.001)
# adadelta = Adadelta(lr=0.001)
# Adagrad = Adagrad(lr=0.001)
# Adamax =  Adamax(lr=0.001)
# Nadam = Nadam(lr=0.001)

#adam도 경사하강법 중에 하나이다. 
from keras.activations import relu,selu,elu,sigmoid,softmax, tanh
# parameter
def create_hyperparameters(): # epochs, node, acivation 추가 가능
    batches = [64, 128, 256]
    optimizers = [ Adam, Adadelta, Adamax, Nadam, RMSprop, Adagrad, SGD]
    dropout = np.linspace(0.1, 0.5, 5).tolist()
    lr = np.linspace(0.001, 0.01, 10).tolist()
    activation = [elu, relu, selu, tanh, sigmoid, softmax, leaky, sin]                      
    
    return {'batch_size' : batches, 
            'optimizer': optimizers, 
            'drop': dropout,
            'lr': lr,
            'activation': activation}                   

# wrapper
model = KerasClassifier(build_fn = build_model, verbose = 1)
hyperparameters = create_hyperparameters()
search = RandomizedSearchCV(estimator = model, param_distributions = hyperparameters,
                             cv = 3)                        

# fit
search.fit(x_train, y_train)
acc = search.score(x_test,y_test)
# np.save('./keras/keras111_model.py', arr=search)
print("최고의 파람:", search.best_params_)
print( "acc : ", acc)
# 최고의 파람: {'optimizer': <class 'tensorflow.python.keras.optimizer_v2.adam.Adam'>, 
#             'lr': 0.006, 'drop': 0.1, 'batch_size': 256}
# acc :  0.9556000232696533

# 최고의 파람: {'optimizer': <class 'tensorflow.python.keras.optimizer_v2.adam.Adam'>, 'lr': 0.01, 'drop': 0.4, 'batch_size': 64, 'activation': <function sigmoid at 0x000002588E8F11F8>}
# acc :  0.9314000010490417
# PS D:\Study\Bitcamp>