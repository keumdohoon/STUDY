#20200526정리
 #카테고리컬의 숫자는 0부터 시작한다. 그래서 어제 과제에서 0을 없애주라고 할때 OneHotIncoding 이나 슬라이싱을 사용하여 없앨수도 있다. 
 #Argmax설명 이해하기
#2번의 첫번째 답1


import numpy as np
y= np.array([1,2,3,4,5,1,2,3,4,5])

#from keras.utils import np_utils
#y = np_utils.to_categorical(y)
#print(y)
#print(y.shape)
#to categorical같은 경우는 시작이 0이여서 플러스 1을 해줘야 우리가 원하는 수가 나온다. 
y= y.reshape(-1,1)
#y = y.reshape(10,1)
#2번째의 답2
from sklearn.preprocessing import OneHotEncoder
aaa = OneHotEncoder()
aaa.fit(y)
y = aaa.transform(y).toarray()
#왜냐하면 들어가고 나오는 값이 달라지기 때문엗. 와꾸를 맞춰줘야한다. 
print(y)
print(y.shape)