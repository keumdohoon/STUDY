#iris 를 케라스 파이프라인으로 구성
#당연히 randomizedsearchcv 구성
#keras98 참고할것

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, Dense
from keras.layers import MaxPooling2D
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

#. 1. 데이터 

#1. DATA
# x = iris["data"]

iris = load_iris()
x = iris.data
y = iris.target
#싸이킷 런에서 땡기는 방식과 케라스에서 땡기는 방식이 따로 있다.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, shuffle = True, random_state = 43)

print(x_train.shape)#(120, 4)
print(x_test.shape)#(30, 4)



x_train = x_train.reshape(x_train.shape[0], 4, 1)
x_test = x_test.reshape(x_test.shape[0], 4, 1)



y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#캐라스에서 하는거는 라벨의 시작이 0부터이니 원핫 인코딩을 할때에는 y의 차원을 반드시확인하고 들어가야한다.
# 
print(y_train.shape)  #(120, 3)
print(y_test.shape)  #(30, 3)


# 2. 모델
# gridsearch 에 있는 parameter으의 순서를 보면 
# def build_model(drop=0.5, optimizer= 'adam'):
#     inputs = Input(shape=(4,), name = 'input')
#     x = Dense(512, activation = 'relu', name= 'hidden1')(inputs)
#     x= Dropout(drop)(x)
#     x = Dense(256, activation = 'relu', name= 'hidden2')(x)
#     x= Dropout(drop)(x)
#     x = Dense(128, activation = 'relu', name= 'hidden3')(x)
#     x= Dropout(drop)(x)
#     output  = Dense(3, activation = 'softmax', name= 'outputs')(x)
#     model = Model(inputs =inputs, outputs = output)
#     model.compile (optimizer, metrics =["acc"], loss = 'categorical_crossentropy')
#     return model


def build_model(drop=0.5, optimizer= 'adam'):
    model = Sequential()
    model.add(Conv2D(80, (1,1), activation='relu', padding='same', input_shape=(4,1)))
    model.add(Conv2D(70, (1,1), activation='relu', padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv2D(160, (1,1), activation='relu', padding='same'))
    model.add(Conv2D(100, (1,1), activation='relu', padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv2D(80, (1,1), activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size = 2))
    model.add(Conv2D(40, (1,1), activation='relu', padding='same'))
    model.add(Conv2D(30, (1,1), activation='relu', padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv2D(30, (1,1), activation='relu', padding='same'))
    model.add(Conv2D(100, (1,1), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))
    model.compile (optimizer, metrics =["acc"], loss = 'categorical_crossentropy')
    return model

#모델을 만들었고 이제 파라미터를 만들어준다. 

def create_hyperparameters():
    batches =[100]
    optimizers = ['adam']
    dropout = [10]
    return{"models__batch_size" : batches,
             "models__optimizer" : optimizers,
             "models__drop" :dropout}

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier

model= KerasClassifier(build_fn= build_model)

hyperparameters = create_hyperparameters()
pipe = Pipeline([("scaler", MinMaxScaler()), ("models", model)])
print('pipe', pipe)

#모델 핏. 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# pipe = Pipeline([("scaler", MinMaxScaler()), ('model', model)])
# pipe = make_pipeline(MinMaxScaler(), SVC())


'''

search = RandomizedSearchCV(pipe, hyperparameters, cv = 3)

try: 
    search.fit(x_train, y_train)
except:
    print("Error")


print('최적의 매개 변수 = ', search.best_estimator_)
print('최적의 parameters = ', search.best_params_)



# acc: 0.9311
# {'optimizer': 'rmsprop', 'drop': 0.1, 'batch_size': 50}

#score
#score 을 추가하여 작성 
acc = search.score(x_test, y_test)
print(search.best_params_)
print("acc :", acc)
# acc: 0.9143
# {'optimizer': 'adadelta', 'drop': 0.30000000000000004, 'batch_size': 20}  
# Traceback (most recent call last):
#   File "d:\Study\Bitcamp\keras\keras98_randomsearch.py", line 105, in <module>
#     print("최적의 매개변수 :", model.best_params_)
'''