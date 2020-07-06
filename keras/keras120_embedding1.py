from keras.preprocessing.text import Tokenizer
#문자 전처리 방식 중에 케라스에 Tokenizer이라는 것이 있다.


#자연어 처리에서는 단어나 글자별로 분류하게 된다. 

#tokenizer은 임베딩에 들어가기 전 단계에서 실행해 준다. 
text = '나는 맛있는 밥을 먹었다'
token = Tokenizer()
token.fit_on_texts([text])
print(token.word_index)
#{'나는': 1, '맛있는': 2, '밥을': 3, '먹었다': 4}



# print(token.index_word)
# {1: '나는', 2: '맛있는', 3: '밥을', 4: '먹었다'}

#index_word하게 되면 인덱스가 앞에 오게 되고 
#word_index을 하게 되면 단어가 먼저 오게 된다.
 

#자연어 처리의 기본은 위와 같이 각 단어별로 인덱스를 주어서 수취화 해 주는 것이다. 
#위의 단계를 fit이라고 보면 되고 아래의 단계를 transform 단계라고 보면 된다. 

x= token.texts_to_sequences([text])
print(x)
# [[1, 2, 3, 4]]


#모든 단어를 수취화 시켜준다음에 모델을 돌려줄수 있지만 문제가 하나 있는데 
#글자 수의 차이 때문에 1의 '나는' 과 2, 의  '맛있는' 글자 수가 다르기때문에 다른 가중치를-
#다른 가중치를 줄 수 있게 된다. 그렇게 되면 연산이 흐트러지게 된다. 인덱싱일 뿐이기 때문이다. 
#우리는 이 와 같은 상황을 이미 해결해본 경험이 있다. one hot encoding 을 사용하여 mnist에서 
#  [0,0,1,0], [1,0,0,0],와 같은식으로 0과 1로 표현해 주었다. 

from keras.utils import to_categorical


# print(len(token.word_index))#4
#token.word_index는 dict형태로 작성되어있기 때문에 len으로 해주어야한다.
word_size = len(token.word_index)+1

#아래에서 x는 matrix으로 변환하게 될 수 를 말해준다. 
#numclasses는 4+1이다. 통 4개의 word가 있으니
x = to_categorical(x, num_classes=word_size)
print(x)
# [[[0. 1. 0. 0. 0.]
#   [0. 0. 1. 0. 0.]
#   [0. 0. 0. 1. 0.]
#   [0. 0. 0. 0. 1.]]]
#이러한 방식으로 하면 안 좋은점이 너무 배보다 보꼽이 커지고 글자 수가 많아질수록 숫자가 너무 기하급수적으로 늘어나게 된다. 


#그래서 이를 해결하고자 나오게 된 것이 임베팅이다. 

