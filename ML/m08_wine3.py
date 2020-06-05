import pandas as pd
import matplotlib.pyplot as plt

#와인 데이터 읽기 
wine = pd.read_csv("./data/csv/winequality-white.csv", sep=';', header=0)
#컬럼 안에 들어가 있는걸 그룹별로 모아주는 것

count_data = wine.groupby('quality')['quality'].count()
#1,2,3,4,5,6,7,8,9, 을 행별로 세겠다는 뜻이다 pandas groupby 검색해복시
print(count_data)

count_data.plot()
plt.show()