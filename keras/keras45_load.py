#keras45_load.py
#keras44 번을 그대로 가져온것
#keras40_lstm_split1

import numpy as np          
from keras.models import Sequential
from keras.layers import Dense, LSTM





#1. 데이터
a= np.array(range(1,11))
size = 5    #time_steps=4
#LSTM 모델을 완성하시오.


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

x = dataset[ : , 0:4] #컴마가 나오면 몇바이 몇이냐, 똔똔하고 양옆에 아무것도 안했으면 모든행이라는 뜻이고, 0부터 4라고 하면 시작인덱느는 그대로 가고 0,1,2,3
#결론적으로는 [:=모든행 을 가져오겠다=행 ,0:4=열= 0부터 3열까지만 가져오겠다.] 
y = dataset[ : , 4]
#[:=모든행을 가져오겠다, 4= 4번째의 것을 4번째의 열을 가져오겠다는 뜻이다.즉 결과 값은 밑에처럼 된다.]
#사이즈를 5로 자르게 된다면
#1,2,3,4,5,
#2,3,4,5,6
#3.4.5.6.7
#4.5.6.7.8
#5.6.7.8.9
#6.7.8.9.10

#x는
#1,2,3,4
#2,3,4,5
#3.4.5.6
#4.5.6.7
#5.6.7.8
#6.7.8.9

#y는
#5,
#6,
#7,
#8,
#9,
#10
#[:] x와 y를 분리하는 방법은 리스트에서 슬라이싱을 통해서 하게 되면 된다. 

x = np.reshape(x, (6, 4, 1))#전체 행이6개 열이4개, 1개씩 자르니까 1 
#x = x.reshape(6, 4, 1)#위에것과 같은것을 의미함
print("x:", x)
print("y:", y)

#39번 스플릿 함수를 그대로 카피해온다.

#a라는 데이터를 가지고 이 함수에 대입해서  사이즈 5니깐  타임 스텝스는 4, 6,5 로 나누어지게 된다.
# shape의 행무시로 들어가는 배치 사이즈는 행의 수 를 말하는거고 compile에 나오는 배치사이즈는 총 잘라주는 갯수인것이다. 


#2. 모델

from keras.models import load_model

#다른 모델을 땡겨와도 되는데 항상 와꾸가 맞는지랑 땡겨온 모델이랑 네이밍에서 충돌이 이루어지지 않는 것인지 주의해야한다. 
model = load_model(".//model//save_44.h5")

model.add(Dense(1, name='new'))
model.summary()

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience=5, mode = 'auto')

model.compile(optimizer='adam', loss='mse', metrics= ['mse'])
model.fit(x, y, epochs=120, batch_size=1, verbose=1,
           callbacks=[early_stopping])#변수를 이미 만들어 놓으면 나중에 리스트에서 푸펀으로  earlystopping이 이미 나오게 되어있다. 

loss, mse = model.evaluate(x, y)

y_predict = model.predict(x)

print('loss :', loss)
print('mse :', mse)
print('y_predict :' , y_predict )


