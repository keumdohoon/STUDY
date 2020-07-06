#122번 embedding3 파일을 conv1d로ㅓ 바꿔주어라 
from keras.preprocessing.text import Tokenizer
import numpy as np

docs = ["너무 재밌어요", "최고에요", '참잘 만든 영화에요', '추천하고 싶은 영화입니다'
        , '한 번 더 보고싶네요', '글쎄요', '별로에요', '재미없어요', '생각보다 지루해요', '연기가 어색해요', '너무 재미없다',
         '참 재밋네요']


labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,1])

token= Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)
# [[1, 2], [3], [4, 5, 6], [7, 8, 9], [10, 11], [12], [13], [14], [15, 16], [17, 18], [1, 19], [20, 21]]
#이렇게 하니 문제가 되는 넋이 셰이프가 달라져서 머신에 집어넣을때 곤란하다. 
#그래서 이를 보완하기 위해서 우리는 cnn모델에서 빈 공간을 채워주는 역할을 해주는 padding_same을 적용하게 해준다 
#먼저는 가장 큰 수치의 갯수는 구해서 위와 같은 경우는 



#여기서 post, pre를 하면 0을 앞으로 넣을건지 아니면 원레 데이터의 뒤에 넣을건지를 결정하게 되는 것이다. 
from keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='post', value=1.0) #벨류를 1로 넣어 주었기 때문에 빈자리에는 다 1로 채워주게 된다. 

print(pad_x)
# [[ 1  2  1  1]
#  [ 3  1  1  1]
#  [ 4  5  6  1]
#  [ 7  8  9  1]
#  [10 11 12 13]
#  [14  1  1  1]
#  [15  1  1  1]
#  [16  1  1  1]
#  [17 18  1  1]
#  [19 20  1  1]
#  [ 1 21  1  1]
#  [22 23  1  1]]


#Conv1D 에 넣어 주기 위해서는 reshape를 해줘야 한다. 
print(pad_x.shape)  #(12, 4)
pad_x = pad_x.reshape(pad_x.shape[0], pad_x.shape[1], 1)
print(pad_x.shape)  #(12, 4, 1)
#임베딩은 단어별로 이쓴ㄴ 유사점이나 공통점으 ㄹ벡터화 한다. 결론적으로는 one hot encoding의 친구라고 보면 된다.

#현재 가장 긴 길의의 단어 5개의 단어이다. 한 번 더 보고싶네요.
word_size = len(token.word_index)+1
print('전체 토큰 사이즈', word_size)  #전체 토큰 사이즈 24

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, LSTM, Conv1D


model= Sequential()

model.add(Conv1D(1, 1, input_shape=(4,1)))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))

model.summary()

#만약 enbedding을 사용하지 않는다면 LSTM 도 구현 가능하다. 


# model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer ='adam', loss = "binary_crossentropy", metrics= ['acc'])
model.fit(pad_x, labels, epochs=30)

acc = model.evaluate(pad_x, labels)[1]
print('acc :', acc)


#워드 사이즈의 크니 우리가 가장 많이 하게 될 1만개의 데이터를 가지고 할 것이다. 사이즈는 우리가 전체 데이터의 크기를 잡지만 아무거나 줘도 돌아는가지만 파라미터에 영향을 주니 항상 명심할것. 
#인풋 랭스는 현재 들어가는 값12,4 열에 대해서 명시를 해 준 것이다. 