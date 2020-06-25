import pandas as pd
import matplotlib.pyplot as plt

#와인 데이터 읽기 
wine = pd.read_csv("./data/csv/winequality-white.csv", sep=';', header=0)
#컬럼 안에 들어가 있는걸 그룹별로 모아주는 것

count_data = wine.groupby('quality')['quality'].count()
#3,4,5,6,7,8,9, 을 행별로 세겠다는 뜻이다 pandas groupby 검색해보기
#지금 현재 보면 그룹의 형태가 3부터 9까지 너무 다양하고 대부분의 클라스는 6과 7에 몰려 있다.
# 그래서 이를 다음 파일에서는 3개의 그룹으로 나누어줄 것이다 . 
print(count_data)
 # quality
 # 3      20
 # 4     163
 # 5    1457
 # 6    2198
 # 7     880
 # 8     175
 # 9       5

count_data.plot()
plt.show()
#Quality 의 분포도가 한곳에 집중되어서 정확한 값을 찾기가 어려워짐
#y값이 5와6에 집중되어서 머신이 학습을 하게 된다면 대부분의 경우에는 맞을수도 있지만 전체적으로는 정확도가 떨어진다. 
#일정 정확도 이상은 올리기가 어려워진다. 