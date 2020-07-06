#뉴스 기사를 가지고 46개의 카테고리로 나누어지는 예제이다. 
from keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib import history

#keras에서 데이터를 가져오고
(x_train, y_train), (x_test, y_test) =reuters.load_data(num_words=1000, test_split=0.2)

print(x_train.shape, x_test.shape) #(8982,) (2246,)
print(y_train.shape, y_test.shape) #(8982,) (2246,)
#이런식으로 가져와 주면 되는 것이다 .
#총 만 1258개의 데이터 8982+2246=11228

print(x_train[0])  # [1, 2, 2, 8, 43, 10, 447, ... , 30, 32, 132, 6, 109, 15, 17, 12]
print(y_train[0])  # 3


# print(x_train[0].shape)------->AttributeError: 'list' object has no attribute 'shape'
print(len(x_train[0]))  #87
#이것들 데이터의 갯수는 일정하지 않다고 우리는 판단할수 있다 하지만 우리는 이제 이것의 와꾸를 맞춰주어야 한다는 결과를 도출 해 낼 수 있다. 

#이를 처리해 주기 위해서 우리는 pad_sequence를 사용하여 앞뒤의 빈자리를 채워주면서 맞춰 줄 수 있다는 것이다. 
#y가 총 몇개인지 모르고 있는 상황이다. 명확하게 여기서 확인을 해볼것이다. 


#y의 카테고리 갯수 출력
category = np.max(y_train) + 1
#제일 많은 글자수를 가지고 있는 것이 무엇인지 모르기 때문에 y_train에서 가장 큰것을 골라 준다는 np.max를 사용해 주었다. 
print("카테고리 :" , category)  #카테고리 : 46-->0~45까지 있다는 뜻이다. 

#y의 유니크한 값들 출력 
y_scatter = np.unique(y_train)
print(y_scatter)
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
#위처럼 0~45까지 총 46개가 있다. 

y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count()
print(bbb)
'''0       55
1      432
2       74
3     3159
4     1949
5       17
6       48
7       16
8      139
9      101
10     124
11     390
12      49
13     172
14      26
15      20
16     444
17      39
18      66
19     549
20     269
21     100
22      15
23      41
24      62
25      92
26      24
27      15
28      48
29      19
30      45
31      39
32      32
33      11
34      50
35      10
36      49
37      19
38      19
39      24
40      36
41      30
42      13
43      21
44      12
45      18
Name: 0, dtype: int64'''
print(bbb.shape)
# (46,)

##################################################################################
# x_train에 대한 크기를 맞춰줘야한다.

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical  

x_train = pad_sequences(x_train, maxlen=100, padding='pre')

x_test = pad_sequences(x_test, maxlen=100, padding='pre')

#y 값이 총 46개이고 이는 다중 분류라고 우리가 쳐줄수 있다 이를 우리는 원핫인코딩을 해 주어야 한다. 
# print(len(x_train[0]))
# print(len(x_train[-1]))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


print("ss",y_test.shape)
print(x_train.shape, x_test.shape) #(8982, 100) (2246, 100)


#모델
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten

model = Sequential() 

model.add(Embedding(1000,128, input_length = 100))
#word_size= 1000
#여기서 123 즉 아웃풋은 우리 마음대로 정할 수 있다. 
#워드 사이즈도 꼭 1000을 하지 않고 5000을 해도 상관은 없지만 파라미터의 갯수가 50만개가 되게 된다. 이것은 즉슨 파라미터의 낭비가 될 수 있다. 
model.add(LSTM(64))
model.add(Dense(46, activation= 'softmax'))

model.summary()

#compile
model.compile(loss = 'categorical_crossentropy', optimizer ='adam', metrics= ['acc'])
history = model.fit(x_train, y_train, batch_size=100, epochs=10,
                    validation_split = 0.2)

acc = model.evaluate(x_test, y_test)[-1]
print("acc: ", acc)


#그림을 그리자

y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker='.', c= 'red', label = 'TestSet Loss')
plt.plot(y_loss, marker='.', c= 'red', label = 'TestSet Loss')
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

