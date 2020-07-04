import pandas as pd

data_1 = pd.read_csv('./dacon/comp4_jeju/data/201901-202003.csv')

print(data_1.head(10))
print('#'*100)
print(data_1.tail(10))

# ./dacon/comp4_jeju/data/201901-202003.csv
# D:\Study\Bitcamp\Dacon\comp4_jeju\data\201901-202003.csv

print('#'*100)


data_2 = pd.read_csv('./dacon/comp4_jeju/data/submission.csv')

print(data_2.head(10))
print('#'*100)
print(data_2.tail(10))
