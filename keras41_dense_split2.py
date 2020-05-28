#keras40_lstm_split1

import numpy as np          
from keras.models import Sequential
from keras.layers import Dense, LSTM





#1. 데이터
a= np.array(range(1,11))
size = 5    #time_steps=4
#LSTM 모델을 완성하시오.

print(a)#[ 1  2  3  4  5  6  7  8  9 10]
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i + size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    
    return np.array(aaa)
dataset = split_x(a, size) #6,5
print("======================")
print("dataset", dataset)  #6,5
print(dataset.shape)
print(type(dataset)) #numpy.ndarray 함수에 보면 리턴 값이 numpyarray이다. 그래서 결과값이 이렇게 나오는 것이다. 

x = dataset[ : , 0:4] 
y = dataset[ : , 4]


x = np.reshape(x, (6,4))#전체 행이6개 열이4개, 1개씩 자르니까 1 
#x = x.reshape(6, 4, 1)#위에것과 같은것을 의미함
print("x:", x)
print("y:", y)

#40번 스플릿 함수를 그대로 카피해온다.



model = Sequential()

model.add(Dense(10, activation='relu', input_shape=(4,)))#원래는 (6,4,1)인데 행을 무시해서 없다. 
model.add(Dense(5))
model.add(Dense(1))

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience=5, mode = 'auto')

model.compile(optimizer='adam', loss='mse', metrics= ['mse'])
model.fit(x, y, epochs=120, batch_size=1, verbose=1,
           callbacks=[early_stopping]) 

loss, mse = model.evaluate(x, y)

y_predict = model.predict(x)

print('loss :', loss)
print('mse :', mse)
print('y_predict :' , y_predict )



'''
loss : 0.06504759192466736
mse : 0.06504759192466736
y_predict : [[ 4.538972 ]
 [ 5.6982903]
 [ 6.828315 ]
 [ 7.9583373]
 [ 9.088359 ]
 [10.218383 ]]
'''