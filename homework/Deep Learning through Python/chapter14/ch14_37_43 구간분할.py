#구간 분할은 이산 범위의 데이터를 분할하여 집계할 경우에 사용하는 편리한 기능이다. 
import pandas as pd
from pandas import DataFrame

attri_data1 = {"ID": ["100", "101", "102", "103", "104", "106", "108", "110", "111", "113"],
               "city": ["서울", "부산", "대전", "광주", "서울", "서울", "부산", "대전", "광주", "서울"],
               "birth_year" :[1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name" :["영이", "순돌", "짱구", "태양", " 션", " 유리", "현아", "태식", "민수", "호식"]}
attri_data_frame1 = DataFrame(attri_data1)

print(attri_data_frame1)
#14_38
birth_year_bins =[1980,1985,1990,1995,2000]

#구간 분할
birth_year_cut_data = pd.cut(attri_data_frame1.birth_year, birth_year_bins)
print(birth_year_cut_data)

# 0    (1985, 1990]
# 1    (1985, 1990]
# 2    (1990, 1995]
# 3    (1995, 2000]
# 4    (1980, 1985]
# 5    (1990, 1995]
# 6    (1985, 1990]
# 7    (1985, 1990]
# 8    (1990, 1995]
# 9    (1980, 1985]
# Name: birth_year, dtype: category     
# Categories (4, interval[int64]): [(1980, 1985] < (1985, 1990] < (1990, 1995] < (1995, 2000]]
#14_39
counting = pd.value_counts(birth_year_cut_data)
print(counting)
# (1985, 1990]    4
# (1990, 1995]    3
# (1980, 1985]    2
# (1995, 2000]    1
# Name: birth_year, dtype: int64   
group_names = ["first1980", "second1980", "first1990", "second1990"]
birth_year_cut_data = pd.cut(attri_data_frame1.birth_year,birth_year_bins,labels = group_names)
countingnames=pd.value_counts(birth_year_cut_data)
#이런식으로 각구릅에 이름을 붙여줄수도 있고 만약 이름을 붙이게 되면 갯수가 많은 순으로 프린트되게 된다. 
print(countingnames)
# second1980    4
# first1990     3
# first1980     2
# second1990    1
# Name: birth_year, dtype: int64 
분할수지정=pd.cut(attri_data_frame1.birth_year, 2)
#birthyear을 두 구간으로 지정하여 프린트해주는 것이다. 
print(분할수지정)
# Name: birth_year, dtype: int64        
# 0      (1989.0, 1997.0]
# 1    (1980.984, 1989.0]
# 2      (1989.0, 1997.0]
# 3      (1989.0, 1997.0]
# 4    (1980.984, 1989.0]
# 5      (1989.0, 1997.0]
# 6    (1980.984, 1989.0]
# 7      (1989.0, 1997.0]
# 8      (1989.0, 1997.0]
# 9    (1980.984, 1989.0]
# Name: birth_year, dtype: category     
# Categories (2, interval[float64]): [(1980.984, 1989.0] < (1989.0, 1997.0]]
#이번에는 id를 2개로 분할해주어서 프린트 할것이다. 

attri_data_frame1 = DataFrame(attri_data1)
아이디나누기=pd.cut(attri_data_frame1.ID, 2)
print(아이디나누기)