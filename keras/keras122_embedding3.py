
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
#임베딩은 단어별로 이쓴ㄴ 유사점이나 공통점으 ㄹ벡터화 한다. 결론적으로는 one hot encoding의 친구라고 보면 된다.
#  

#현재 가장 긴 길의의 단어 5개의 단어이다. 한 번 더 보고싶네요.
word_size = len(token.word_index)+1
print('전체 토큰 사이즈', word_size)  #전체 토큰 사이즈 24

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, LSTM


model= Sequential()
# model.add(Embedding(word_size, 10, input_length =4))   #(None, 4, 10)   
model.add(Embedding(24, 10))
#Conv1d, 나 LSTM을 사용해주면 된다. 이렇게 사용해주게 되면 밑에 Flatten을 생략해줘도 된다. 
model.add(LSTM(3))
#윗 라인 계산 embedding의 아웃풋이 여기서 인풋이 된다 , bias하나 인풋이 3개 아웃풋이 3개 곱하기 LSTM*4
#x에 input값을 명시해주지 않았을 경우에 즉 embedding과 lstm을 같이 써주게 된다면 flatten을 사용하지 않아도 된다. 
#그래서 명시를 안해주는 경우가 있다. 
#Enbedding에 가장 많이 사용해주는것은 LSTM이다. 
#embedding레이어에서 5의 값은 값크기에 반영되지 않는다. 
#embedding은 3차원으로 받아주니까 Conv1d 로도 받아줄 수 있다 
# model.add(Flatten())#Flatten 하면 현재 벡터의 형식인 것은 Dense로 평평하게 붙여줄수 있게 된다. 


#만약 enbedding을 사용하지 않는다면 LSTM 도 구현 가능하다. 

model.add(Dense(25))
model.add(Dense(1, activation='sigmoid'))

model.summary()
#우리의 최종출력은 1, 이냐 0, 이냐 긍정인지 부정인지를 정하는 것이다. 

model.compile(optimizer ='adam', loss = "binary_crossentropy", metrics= ['acc'])
model.fit(pad_x, labels, epochs=30)

acc = model.evaluate(pad_x, labels)[1]
print('acc :', acc)

# acc : 0.8333333134651184
