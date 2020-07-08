##뉴스 기사를 가지고 46개의 카테고리로 나누어지는 예제이다. 
from keras.datasets import reuters, imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib import history


word_index = imdb.get_word_index()


#keras에서 데이터를 가져오고
(x_train, y_train), (x_test, y_test) = imdb.load_data()

print(x_train.shape, x_test.shape) #(25000,) (25000,)
print(y_train.shape, y_test.shape) #(25000,) (25000,)

print('훈련용 리뷰 개수 : {}'.format(len(x_train))) #훈련용 리뷰 개수 : 25000
print('테스트용 리뷰 개수 : {}'.format(len(x_test))) #테스트용 리뷰 개수 : 25000
num_classes = max(y_train) + 1
print('카테고리 : {}'.format(num_classes)) #카테고리 : 2
print(x_train[0])
print(y_train[0])


len_result = [len(s) for s in x_train]

print('리뷰의 최대 길이 : {}'.format(np.max(len_result))) #리뷰의 최대 길이 : 2494
print('리뷰의 평균 길이 : {}'.format(np.mean(len_result))) #리뷰의 평균 길이 : 238.71364

plt.subplot(1,2,1)
plt.boxplot(len_result)
plt.subplot(1,2,2)
plt.hist(len_result, bins=50)
# plt.show()

unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("각 레이블에 대한 빈도수:") #[[    0     1] [12500 12500]]
print(np.asarray((unique_elements, counts_elements)))

word_to_index = imdb.get_word_index()
index_to_word={}
for key, value in word_to_index.items():
    index_to_word[value] = key

print('빈도수 상위 1번 단어 : {}'.format(index_to_word[1])) #빈도수 상위 1번 단어 : the  가장 많이 쓰인 단어를 뜻한다. 

print('빈도수 상위 3941번 단어 : {}'.format(index_to_word[3941])) #빈도수 상위 3941번 단어 : journalist 즉 제일 많이 안 쓰인 단어를 뜻한다. 

for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
      index_to_word[index]=token

print(' '.join([index_to_word[index] for index in x_train[0]]))


from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 5000)
#훈련 데이터느느 빈도 1~500 인 단어들로만 가지고 오게 되고 단어 집합의 크기를 5000으로 제한하게 된다는 뜻이다. 

max_len = 500
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)
#리뷰의 최대 길이는 500 자인것으로 제한하고 각리뷰의 문장이 다르기 때문에 모델이 처리할 수 있도록 길이를 동일하게 해주어야 한다.
model = Sequential()
model.add(Embedding(5000, 120))
model.add(LSTM(120))
model.add(Dense(1, activation='sigmoid'))  
#embedding은 두개의 인자를 받는데 첫번째 인자는 단어 집합의 크기이며 두번째 인자는 임베딩 후의 벡터의 크기이다.
# 이 예제에서는 120을 선택하였는데 입력데이터에서 모든 단어는 120 차원을 가진 임베딩 백터로 표현됩니다.  

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64, callbacks=[es, mc])


# loss: 0.1762 - acc: 0.9321 - val_loss: 0.3666 - val_acc: 0.8654

# Epoch 00008: val_acc did not improve from 0.87252
# Epoch 00008: early stopping