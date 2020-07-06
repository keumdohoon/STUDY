from keras.preprocessing.text import Tokenizer
import numpy as np

docs = ["너무 재밌어요", "최고에요", '참잘 만든 영화에요', '추천하고 싶은 영화입니다',
         '한 번 더 보고싶네요', '글쎄요', '별로에요', '재미없어요', '생각보다 지루해요',
          '연기가 어색해요', '너무 재미없다','참 재밋네요']


#은 긍정을 나타내고 2는 부정을 나타내게 된다.
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,1])

#docs 는 x labels 는 y라고 칭해 줄것이다
#각각의 x는 y를 가지고 있고 우리는 각각을 레이블을 주었다. 

#첫번째 라인으 '너무' 라는 단어랑 마지막줄에 있는 '너무'와 '참'은 두번이상 나온 애들로서 중복된 부분은 빼주게되고 
#반복도 가 많은 단어일수록 인덱스 앞쪽번호로 자동으로 빼주게 된다.  

token= Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
x = token.texts_to_sequences(docs)
print(x)
# [[1, 2], [3], [4, 5, 6], [7, 8, 9], [10, 11], [12], [13], [14], [15, 16], [17, 18], [1, 19], [20, 21]]
#위와 같은 결과는 셰이프가 달라져서 곤란하다. 


#이를 보완하기 위해서 우리는 cnn모델에서 빈 공간을 채워주는 역할을 해주는 padding_same을 적용하게 해준다 
#여기서 post, pre를 하면 0을 앞으로 넣을건지 아니면 원레 데이터의 뒤에 넣을건지를 결정하게 되는 것이다. 
from keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='post', value=0) #벨류를 0로 넣어 주었기 때문에 빈자리에는 다 0로 채워주게 된다. 
print(pad_x)
#[[ 1  2  0  0]
#  [ 3  0  0  0]
#  [ 4  5  6  0]
#  [ 7  8  9  0]
#  [10 11 12 13]
#  [14  0  0  0]
#  [15  0  0  0]
#  [16  0  0  0]
#  [17 18  0  0]
#  [19 20  0  0]
#  [ 1 21  0  0]
#  [22 23  0  0]]
#임베딩은 단어별로 이쓴ㄴ 유사점이나 공통점으 ㄹ벡터화 한다. 결론적으로는 one hot encoding의 친구라고 보면 된다.
#  

#현재 가장 긴 길의의 단어 4개의 단어이다. 한 번 더 보고싶네요.
word_size = len(token.word_index)+1
print('전체 토큰 사이즈', word_size)  #전체 토큰 사이즈 24/전체 단어의 수

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten

model= Sequential()
model.add(Embedding(word_size, 10, input_length =4))   #(None, 4, 10)   
                # (전체 단어의 갯수, embedding 에서는 10= 아웃풋(특별하게 두번째단에다가 둔다), input_length= input값이다.)
                    # 위의 10은 다음 레이어에 출력하게 되는 노드의 갯수이다. 변동해도 상관없음. 

#이부분은 와꾸를 맞춰주는 부분이다. 
# model.add(Embedding(24, 10, 4))

#앞에 크기를 word size를 설정해줄때는 그 크기를 우리가 임의로 넣어줘도 된다
#신문에서 만개가 있던 천개가 있던 다 상관없으니 우리가 워드의 갯수를 어떻게 할 것인지와 연산의 갯수만 달라지지만 전체 크기를 잘 잡아준다., 

model.add(Flatten())#Flatten 하면 현재 벡터의 형식인 것은 Dense로 평평하게 붙여줄수 있게 된다. 

model.add(Dense(1, activation='sigmoid'))

model.summary()
#우리의 최종출력은 1, 이냐 0, 이냐 긍정인지 부정인지를 정하는 것이다. 

model.compile(optimizer ='adam', loss = "binary_crossentropy", metrics= ['acc'])
model.fit(pad_x, labels, epochs=30)

acc = model.evaluate(pad_x, labels)[1]
print('acc :', acc)


#워드 사이즈의 크니 우리가 가장 많이 하게 될 1만개의 데이터를 가지고 할 것이다. 사이즈는 우리가 전체 데이터의 크기를 잡지만 아무거나 줘도 돌아는가지만 파라미터에 영향을 주니 항상 명심할것. 
#인풋 랭스는 현재 들어가는 값12,4 열에 대해서 명시를 해 준 것이다. 