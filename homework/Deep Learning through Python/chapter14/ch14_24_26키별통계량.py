#키별 통계량 산출. 
import pandas as pd

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)
df.columns=["", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium","Total phenols", "Flavanoids", "Nonflavanoid phenols", 
            "Proanthocyanins","Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
print(df["Alcohol"].mean())
#이렇게 하면 각 키 중에 alcohol의 평균을 나타내어준다.
# avg of wine alcohol 13.00061797752809  

import pandas as pd

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)
df.columns=["", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium","Total phenols", "Flavanoids", "Nonflavanoid phenols", 
            "Proanthocyanins","Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]

# 이렇게 하면 magnesium에 들어가있는 값들의 평균을 구해준다. 
#만약 magnesium에 nan값이 들어가 있다면 계산이 안되기 때문에 앞서 배운 결측치제거를 통하여 읻를 보안해주자. 
print(df["Magnesium"].mean())