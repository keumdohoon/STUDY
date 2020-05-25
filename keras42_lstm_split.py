#keras40_lstm_split1

import numpy as np          
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split




#1. 데이터
a= np.array(range(1,101))
size = 5    #사이즈가 5면 time_steps=4
#LSTM 모델을 완성하시오.
print(a.shape)

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):#100에서 사이즈 뺀다음에 더하기 1, 사이즈에서 빼기 하나 한게 의 크기이다 고로 96,5가 된다. 
        subset = seq[i : (i + size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)
#실습 1. train, test 분리할것(트레인과 테스트의 비율은 8:2)
#실습 2. 96행중에 나는 마지막 6행을 잘라서 그거를  predict로  하고 싶다, 제일 마지막에 여섯개 겠징.   
#실습 3. validation을 넣을 것 (train 의 20%)

data = split_x(a, size)
# 1-2-1. predict 데이터 분할
predict = data[90: , :4]
print(predict)
# predict = predict.reshape(6, 4, 1)
print(predict.shape)

#################x = np.reshape(94, 4, 1)###########



x = data[:90, :4]
y = data[:90, -1:]



print(x)
print(x.shape)
print(y)
print(y.shape)



# x = x.reshape(90, 4, 1)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True,
    random_state = 666)

print(x_train.shape)
print(x_test.shape)
print(y_train)
print(y_test)

x = np.reshape(94, 4, 1)

'''    
dataset = split_x(a, size) 
print("====================================")
print("dataset", dataset)  #
print(dataset.shape)
print(type(dataset)) #numpy.ndarray 함수에 보면 리턴 값이 numpyarray이다. 그래서 결과값이 이렇게 나오는 것이다. 

predict = dataset[ 90: , 0:4] 
y_train = dataset[0:90 , 4]
x_test = dataset[ 0:90 , 0:4] 
y_test = dataset[0:90 , 4]
x_predict = dataset[ 90:94 , 0:4] 

print("x_train:", x_train)
print("y_train:", y_train)
print("x_test:", x_test)
print("y_test:", y_test)
print("x_predict:", x_predict)

########################################################
#

######################################################

x = np.reshape(x, (6,4))#전체 행이6개 열이4개, 1개씩 자르니까 1 
#x = x.reshape(6, 4, 1)#위에것과 같은것을 의미함
print("x:", x)
print("y:", y)

#41번 스플릿 함수를 그대로 카피해온다.


'''
 
#2. 모델

model = Sequential()

model.add(LSTM(10, activation='relu', input_dim=(4,1)))
model.add(Dense(5))
model.add(Dense(1))

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience=5, mode = 'auto')
#3. 훈련, 컴파일
model.compile(optimizer='adam', loss='mse', metrics= ['mse'])
model.fit(x_train, y_train, epochs=120, batch_size=1, verbose=1,
           callbacks=[early_stopping], validation_split=0.2, shuffle=True) 




#4. 평가와 예측
loss, mse = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("mse : ", mse)


y_predict = model.predict(predict, batch_size = 1)


print("y_predict : \n", y_predict)
print(y_predict.shape)






