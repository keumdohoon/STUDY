#keras47_tensorboard.py
#keras46 번을 그대로 가져온것
#loss값을 fit에서 보여주는 거를 커멘드 창에서 보여주는 것이다, 이것을 save할 수 있을까?
#tensorboard 는 이미지로 보여줄때 더 이쁘게 나오고 웹을 기반으로 하기때문에 이쁘게 그림이 나온다. 

import numpy as np          
from keras.models import Sequential
from keras.layers import Dense, LSTM





#1. 데이터
a= np.array(range(1,101))
size = 5    #time_steps=4


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


x = np.reshape(x, (96, 4, 1))#전체 행이6개 열이4개, 1개씩 자르니까 1 
#x = x.reshape(6, 4, 1)#위에것과 같은것을 의미함
print("x:", x)
print("y:", y)

#46번 스플릿 함수를 그대로 카피해온다.

#2. 모델

from keras.models import load_model

#model = load_model(".//model//save_44.h5")

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(4,1)))#원래는 (6,4,1)인데 행을 무시해서 없다. 
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(3))

model.add(Dense(1))

model.summary()
from keras.callbacks import EarlyStopping, TensorBoard
#Tensorboard란 
tb_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
#주의해야할 부분은 경로이다, 파일과 폴더를 항상 표준화시키는 것이 경로 에러이다. 텐서보드에도 경로를 넣는 부분이 있다. 
#텐서보드에도 lod dir에다가 경로를 넣어줘야한다. 텐서보드 를 저장하기 위한  graph폴더를 한다. 실행할때 저장해준다.  
#'graph'를 적어주면 그래프 폴더에다가 트레인과 밸리데이션을 만들어 준다. 
early_stopping = EarlyStopping(monitor = 'loss', patience=5, mode = 'auto')

model.compile(optimizer='adam', loss='mse', metrics= ['acc'])
hist = model.fit(x, y, epochs=100, validation_split=0.2, batch_size=1, verbose=1,
           callbacks=[early_stopping, tb_hist])

print(hist) 
print(hist.history.keys())

import matplotlib.pyplot as plt#그래프를 그려주는 것을  plt라고 하겠다

plt.plot(hist.history['loss'])#y값만 넣었다는 뜻이다 , loss값만을 하겠다. hist에 history에 로스 값만을 하겠다. 
plt.plot(hist.history['acc'])#y값만 넣었다는 뜻이다 , loss값만을 하겠다. hist에 history에 로스 값만을 하겠다. 
plt.plot(hist.history['val_loss'])#y값만 넣었다는 뜻이다 , loss값만을 하겠다. hist에 history에 로스 값만을 하겠다. 
plt.plot(hist.history['val_acc'])#y값만 넣었다는 뜻이다 , loss값만을 하겠다. hist에 history에 로스 값만을 하겠다. 
#plt.plot(hist.history['val_loss'])#validation을 지금 지정안해뒀기 때문에 빼주는것
plt.title('loss & acc')#제목
plt.ylabel('loss & acc')#y라벨
plt.xlabel('epoch')#x라벨
plt.legend(['train loss', 'test loss', 'train acc', 'test acc'])
#plt.show()



'''
#4. 평가 예측
loss, mse = model.evaluate(x, y)

y_predict = model.predict(x)

print('loss :', loss)
print('mse :', mse)
print('y_predict :' , y_predict )
'''
x
