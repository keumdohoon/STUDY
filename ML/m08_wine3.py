import pandas as pd
import matplotlib.pyplot as plt

#와인 데이터 읽기 
wine = pd.read_csv("./data/csv/winequality-white.csv", sep=';', header=0)
#컬럼 안에 들어가 있는걸 그룹별로 모아주는 것

count_data = wine.groupby('quality')['quality'].count()
#3,4,5,6,7,8,9, 을 행별로 세겠다는 뜻이다 pandas groupby 검색해복시
#지금 현재 보면 그룹의 형태가 3부터 9까지 너무 다양하고 대부분의 클라스는 6과 7에 몰려 있다.
# 그래서 이를 다음 파일에서는 3개의 그룹으로 나누어줄 것이다 . 
print(count_data)

count_data.plot()
plt.show()