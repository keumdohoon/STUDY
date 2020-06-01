#keras44_save.py
#keras 40 을 카피해서 복붙

import numpy as np          
from keras.models import Sequential
from keras.layers import Dense, LSTM

#2. 모델
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(4,1)))#원래는 (6,4,1)인데 행을 무시해서 없다. 
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(10))

# model.summary()

model.save(".//model//save_44.h5")#.은 현재 폴더를 뜻해준다. 그하단에 keras

#모델을 저장하는 법은 h5확장자를 쓴다.     
print("저장 잘됐다.")