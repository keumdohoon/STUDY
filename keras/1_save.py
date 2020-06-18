#코스피 파일을 불러온다.

import numpy as np
import pandas as pd

datasets1 = pd.read_csv("./data/csv/kospi200.csv", 
                        index_col = 0, 
                        header = 0, encoding = 'cp949',sep = ',')

print(datasets1)       
print(datasets1.shape)      

#삼성주가 파일을 불러온다.

import numpy as np
import pandas as pd

datasets2 = pd.read_csv("./data/csv/samsung.csv", 
                        index_col = 0, 
                        header = 0, encoding = 'cp949',sep = ',')

print(datasets2)       
print(datasets2.shape)        

#거래량str->int 변경
#kospi200
for i in range(len(datasets1.index)): 
    datasets1.iloc[i,4] = int(datasets1.iloc[i,4].replace(',',''))
#samsung
for i in range(len(datasets2.index)): 
    for j in range(len(datasets2.iloc[i])):
        datasets2.iloc[i,j]= int(datasets2.iloc[i,j].replace(',',''))

datasets1 = datasets1.sort_values(['일자'], ascending=[True])
datasets2 = datasets2.sort_values(['일자'], ascending=[True])
#일자순으로 깔끔하게 나오게 분류해준다.
print('datasets1: ', datasets1)  #(426, 5)
print('datasets2', datasets2)

datasets1 = datasets1.values
datasets2 = datasets2.values
#pandas를 numpy로 바꿔준다


print(datasets1.shape)  #(426, 5)
print(datasets2.shape)  #(426, 5)
print(type(datasets1), type(datasets2))
#<class 'numpy.ndarray'> <class 'numpy.ndarray'>로 우리가 원하는 방향으로 변경된걸 알 수 있다.

#데이터를 넘파이로 저장하기
np.save('./data/kospi.npy', arr = datasets1) 
np.save('./data/samsung.npy', arr = datasets2) 
print(datasets1)
#실행시키면 저장까지 다 한것이다.
 
