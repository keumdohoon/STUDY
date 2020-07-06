#keras122_embedding3을 가져다가 conv1d로 구성
#conv1d로 구성
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

pad_x = pad_x.reshape(pad_x.shape[0], pad_x.shape[1])


#현재 가장 긴 길의의 단어 4개의 단어이다. '한 번 더 보고싶네요'
word_size = len(token.word_index)+1
print('전체 토큰 사이즈', word_size)  #전체 토큰 사이즈 24

from keras.models import Sequential
from keras.layers import Dense,Conv1D, Embedding, Flatten, LSTM


model= Sequential()
model.add(Embedding(word_size, 10, input_length =4))   #(None, 4, 10) 
#embedding에 처음에 들어가 줘야하는 단어의 갯수, 10, 이랗 , 제일 큰것의 길이.  
# model.add(Embedding(24, 10))
#Conv1d, 나 LSTM을 사용해주면 된다. 이렇게 사용해주게 되면 밑에 Flatten을 생략해줘도 된다. 
model.add(Conv1D(1, 1, input_shape=(4,1)))
#conv1D는 3차원으로 받아서 3차원으로 내어 주기 때문에 우리는 밑에 Dense레이어에 진입하기 이전에 flatten을 한번 해 주어야 한다. 
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.summary()

#우리의 최종출력은 1, 이냐 0, 이냐 긍정인지 부정인지를 정하는 것이다. 

model.compile(optimizer ='adam', loss = "binary_crossentropy", metrics= ['acc'])
model.fit(pad_x, labels, epochs=30)

acc = model.evaluate(pad_x, labels)[1]
print('acc :', acc)


