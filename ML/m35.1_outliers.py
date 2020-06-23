# 실습 : 행렬을 입력해서 컬럼별로 이상치 발견하는 함수를 구하시오.
#------------------------------------------------------------------
import numpy as np
import pandas as pd 
def outliers(data_out):
    outliers = []
    for i in range(data_out.shape[1]):
        data = data_out[:, i]
        quartile_1, quartile_3 = np.percentile(data, [25, 75])
        print("1사 분위 : ",quartile_1)                                       
        print("3사 분위 : ",quartile_3)                                        
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        out = np.where((data > upper_bound) | (data < lower_bound)) #나오는 값들
        outliers.append(out)   #나오는 값들을 더해준다 결국엔 각 컬럼마다를 구하는거
    return outliers

a2 = np.array([[1, 5000],[200, 8],[2, 4],[3, 7],[8, 2]])
print(a2)
# [[   1 5000]
#  [ 200    8]
#  [   2    4]
#  [   3    7]
#  [   8    2]]

b2 = outliers(a2)
print('b2',b2)
# [(array([1], dtype=int64),), (array([0], dtype=int64),)]

# -------------------------------------------------------------------
# pandas
def outliers(data_out):
        quartile_1 = data_out.quantile(.25)
        quartile_3 = data_out.quantile(.75)
        print("1사 분위 : ",quartile_1)                                       
        print("3사 분위 : ",quartile_3)                                        
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        return np.where((data_out > upper_bound) | (data_out < lower_bound))
         
a3 = pd.DataFrame({'a' : [1, 3, 5, 200, 100, 8],
                    'b' : [300, 100, 6, 8, 2, 3]})
print(a3)
#      a    b
# 0    1  300
# 1    3  100
# 2    5    6
# 3  200    8
# 4  100    2
# 5    8    3

b3 = outliers(a3)
print('b3',b3)
# 1사 분위 :  a    3.50
#             b    3.75
# Name: 0.25, dtype: float64
# 3사 분위 :  a    77.0
#             b    77.0
# Name: 0.75, dtype: float64
# (array([0, 3], dtype=int64), array([1, 0], dtype=int64))