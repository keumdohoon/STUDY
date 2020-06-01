import numpy as np
import pandas as pd
# 판다스를pd 로 가져온다

datasets = pd.read_csv("./data/csv/iris.csv",
                        index_col = None,
                        header=0, sep=',')
                        #저위에있는 파일명과 경로에서 저 데이터를 가져온다.
                        #인덱스 컬럼은 들어가있지 않다, 자동인덱스 생성
                        #해더는 0, 헤더, 첫번때라인을 해더로 했기때문에 실제 데이터는 그 밑에부터이다. 
                        #헤더를 none으로 하면 veronica를 데이터로 인식해버리는 경우가 있다 그래서 우리는 첫번째라인을 표시하는 0을 헤더로 정해주는 것이다.
                         
print(datasets)
print(datasets.head())
#위에서 부터5개정도만 보여주는 것이다
print(datasets.tail())
#아래에서부터 5개

print('======================================')
print(datasets.values)
#.values= 머신을 돌리기 위해서는 pandas로csv-> numpy로 바꿔줘야하는데 그걸 바꿔주는 것이 .values 이다.

aaa= datasets.values
print(type(aaa))
#type를 프린트해보면 numpy로 바꿔줬다는게 티가 나게 된다. 

#넘파이로 저장하시오

np.save('./data/iris_data.npy', arr=aaa)

