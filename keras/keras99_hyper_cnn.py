#원래는 randomizedSearchCV로 변환, 파일 keras97불러오기. 


#score 을 추가해준다. 

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, Dense
from keras.layers import MaxPooling2D
import numpy as np



#. 1. 데이터 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#데이터를 불러옴과 동시에 x, y와 트레인과 테스트를 분리해준다. 

print(x_train.shape)#(60000, 28, 28)
print(x_test.shape)#(10000, 28, 28)

# x_train = x_train.reshape(x_train[0], 28,28,1)/255
# x_test = x_test.reshape(x_test[0], 28,28,1)/255
#0부터 255개의 데이터가 들어가있는데 이것은 결국 민맥스와 같은 결과를 가져다 준다.
x_train = x_train.reshape(x_train.shape[0], 28 , 28, 1)/255
x_test = x_test.reshape(x_test.shape[0], 28 , 28, 1)/255
#위에거는 dense모델을 위해서 만들어 준 것이다. 



y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#캐라스에서 하는거는 라벨의 시작이 0부터이니 원핫 인코딩을 할때에는 y의 차원을 반드시확인하고 들어가야한다.
# 
print(y_train.shape)  






#########################################################################################################
# model = GridSearchCV(modelthat we will use, Parameters that we will use, cv=kfold= u can just input a number)

###################################################################################################
#위에 모델을 만들어주기 위해서 모델, 파라미터, cv를 각각 만들어준다. 
#2. 모델
#gridsearch 에 있는 parameter으의 순서를 보면 
def build_model(drop=0.5, optimizer= 'adam'):#초기값이 없으면 돌아가질 않는다. 
    
    input1= Input(shape= (28,28,1))
    Conv2d1 = Conv2D(filters= 15, kernel_size=3, padding="same", activation="relu")(input1)

    Conv2d2= Conv2D(filters= 15, kernel_size=3, padding="same", activation="relu")(Conv2d1) 
    drop1 = Dropout(0.1)(Conv2d2)

    Conv2d3= Conv2D(filters= 22, kernel_size=3, padding="same", activation="relu")(drop1) 
    Conv2d4= Conv2D(filters= 21, kernel_size=3, padding="same", activation="relu")(Conv2d3) 
    drop2 = Dropout(0.3)(Conv2d4)

    Conv2d5= Conv2D(filters= 21, kernel_size=3, padding="same", activation="relu")(drop2) 
    drop3 = Dropout(0.3)(Conv2d5)
    Conv2d6= Conv2D(filters= 12, kernel_size=3, padding="same", activation="relu")(drop3) 
    drop4 = Dropout(0.3)(Conv2d6)

    Conv2d7= Conv2D(filters= 18, kernel_size=3, padding="same", activation="relu")(drop4) 

    drop5 = Dropout(0.1)(Conv2d7)

    Conv2d8= Conv2D(filters= 12, kernel_size=3, padding="same", activation="relu")(drop5) 
    drop6 = Dropout(0.1)(Conv2d8)

    output1 = (Flatten())(drop6)
    output2 = Dense(10, activation = 'softmax')(output1)
    model = Model(inputs=input1, outputs=output2)
    model.compile (optimizer, metrics =["acc"], loss = 'categorical_crossentropy')
    return model

print(x_train)
print(y_train)
#이렇게 직접 함수형을 만들어 줄 수도 있는 것이다. 
#그리드 서치를 사용하려면 맨처음에 들어가는 것이 모델이기 때문에 우리가 이미 모델을 만들어주고 그걸 사용하기 위해서 직접 모델을 만들어준다. 
#컴파일까지만 만들어주고 핏은 아직 안만들어준다 왜냐하면 핏은 나중에 랜덤서치나 그리드 서치에서 할것이기  때문이다. 

#모델을 만들었고 이제 두번째 파라미터를 만들어준다. 

def create_hyperparameters():
    batches =[10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1,0.5, 5)
    return{"batch_size" : batches, "optimizer" : optimizers,
        "drop" :dropout}
        #위에 딕셔너리형태이다. 파라미터에 들어가는 매개변수 형테는 딕셔너리 형태이다. 그래서 무조건 딕셔너리 형태로 맞춰줘야한다. 케라스가 아니라 싸이킷런에 맞는 모델로 래핑을 만들어주기 위해서 이런식으로 해준다. 
#k fold에서는 숫자만 들어가면 되는것이니 그것도 이미 준비 된것이다. 
#여기다가 에포도 넣을수 있고 노드의 갯수도 변수명을 넣어주고 하이퍼 파라미터에 넣을수 있고 activation도 넣어 줄 수 있다. 여기서 원하는건 다 넣을수 있음. 

#케라스를 그냥 사용하면 안되고 케라스에 보면 사이킷런에 사용할수 있는 wrapper이라는 것이 존재하고 사이킷런에 케라스를 쌓아 올리겠다는는 뜻이다.
from keras.wrappers.scikit_learn import KerasClassifier
#케라스건 사이킷런이건 분류와 회기를 항상 잃지 말고
#케라스에서 wrapping을 해주는 이유는 사이킷런에서 해주기 위해서 
model= KerasClassifier(build_fn= build_model, verbose= 1)
#우리가 만들어둔 모델을 wrapping 해 주는 것이다. kerasClassifier 모델을 이렇게 만들어주는 것이다. 
hyperparameters = create_hyperparameters()
#help buiild a hyper parameters , 위데짜놓은 create_hyperparameters()를 hyperparamer 에 대입시켜준다. 





#여기서 부터가 모델 핏 부분이 되는 것이다. 

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model, hyperparameters, cv = 3)

#여기서부터 fit
search.fit(x_train, y_train)

print(search.best_params_)

#이 폴더에서 항상 주의해야할것들은 소스와 하이퍼 파라미터




# acc: 0.9311
# {'optimizer': 'rmsprop', 'drop': 0.1, 'batch_size': 50}


#score
#score 을 추가하여 작성 
score = search.score(x_test, y_test, verbose=0)
print(search.best_params_)
print("acc :", score)
# acc: 0.9143
# {'optimizer': 'adadelta', 'drop': 0.30000000000000004, 'batch_size': 20}  
# Traceback (most recent call last):
#   File "d:\Study\Bitcamp\keras\keras98_randomsearch.py", line 105, in <module>
#     print("최적의 매개변수 :", model.best_params_)