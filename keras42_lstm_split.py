#keras40_lstm_split1

import numpy as np          
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping




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
predict = predict.reshape(6, 4, 1)
print(predict.shape)


#################x = np.reshape(94, 4, 1)###########



x = data[:90, :4]
y = data[:90, -1:]

#x = np.reshape(94, 4, 1)
print(x)
print(x.shape)
print(y)
print(y.shape)



x = x.reshape(90, 4, 1)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True,
    random_state = 666)

print(x_train.shape)
print(x_test.shape)
print(y_train)
print(y_test)



 
#2. 모델

input1 = Input(shape = (4, 1))
dense1 = LSTM(10, activation='relu', return_sequences = True)(input1)
dense2 = LSTM(10, activation='relu', return_sequences = True)(dense1)
dense3 = LSTM(8, activation = 'relu')(dense2)
dense4 = Dense(8, activation = 'relu')(dense3)

output1 = Dense(3)(dense4)
output2 = Dense(2)(output1)
output3 = Dense(5)(output2)
output4 = Dense(15)(output3)
output5 = Dense(3)(output4)
output6 = Dense(2)(output5)
output7 = Dense(1)(output6)
model = Model(inputs = input1, outputs = output7)
model.summary()

#3. 훈련, 컴파일

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience=5, mode = 'auto')
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






