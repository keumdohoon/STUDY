import pandas as pd
import numpy as np
from numpy import nan as NA
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)

df.columns=["", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash",
            "Magnesium", "Total phenols", "Flavanoids",
            "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue",
            "OD280/OD315 of diluted wines","Proline"]

df_ten = df.head(10)
print(df_ten)

df_ten.iloc[1,0] = NA
df_ten.iloc[2,3] = NA
df_ten.iloc[4,8] = NA
df_ten.iloc[7,3] = NA
print(df_ten)

df_ten.fillna(df_ten.mean())
print(df_ten)

print(df_ten["Alcohol"].mean())

df_ten.append(df_ten.loc[3])
df_ten.append(df_ten.loc[6])
df_ten.append(df_ten.loc[9])
df_ten = df_ten.drop_duplicates()
print(df_ten)

alcohol_bins = [0,5,10,15,20,25]
alcoholr_cut_data = pd.cut(df_ten["Alcohol"],alcohol_bins)

print(pd.value_counts(alcoholr_cut_data))
#        ...  Proline
# 0  1.0  ...     1065
# 1  NaN  ...     1050
# 2  1.0  ...     1185
# 3  1.0  ...     1480
# 4  1.0  ...      735
# 5  1.0  ...     1450
# 6  1.0  ...     1290
# 7  1.0  ...     1295
# 8  1.0  ...     1045
# 9  1.0  ...     1045

# [10 rows x 14 columns]
#         ...  Proline
# 0  1.0  ...     1065
# 1  NaN  ...     1050
# 2  1.0  ...     1185
# 3  1.0  ...     1480
# 4  1.0  ...      735
# 5  1.0  ...     1450
# 6  1.0  ...     1290
# 7  1.0  ...     1295
# 8  1.0  ...     1045
# 9  1.0  ...     1045

# [10 rows x 14 columns]
# 13.954000000000002
#         ...  Proline
# 0  1.0  ...     1065
# 1  NaN  ...     1050
# 2  1.0  ...     1185
# 3  1.0  ...     1480
# 4  1.0  ...      735
# 5  1.0  ...     1450
# 6  1.0  ...     1290
# 7  1.0  ...     1295
# 8  1.0  ...     1045
# 9  1.0  ...     1045

# [10 rows x 14 columns]
# (10, 15]    10
# (20, 25]     0
# (15, 20]     0
# (5, 10]      0
# (0, 5]       0
# Name: Alcohol, dtype: int64